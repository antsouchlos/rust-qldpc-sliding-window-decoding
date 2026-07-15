use std::ops::Range;

use crate::decoders::engine::{
    AccessEngineInternals, BpComputeEngine, EdgeId, ParityCheckMatrix,
};

#[derive(Clone)]
pub struct PhiTable {
    table: Vec<f64>,
    dx_inv: f64,
    x_max: f64,
}

/// Lookup table for the function phi(x) = -ln(tanh(x/2))
impl PhiTable {
    pub fn new(n: usize, x_max: f64) -> Self {
        let dx = x_max / (n - 1) as f64;

        let table = (0..n)
            .map(|i| {
                let x = i as f64 * dx;
                if x < 1e-10 {
                    x_max
                } else {
                    -1.0 * (x / 2.0).tanh().ln()
                }
            })
            .collect();

        PhiTable {
            table,
            dx_inv: 1.0 / dx,
            x_max,
        }
    }

    /// Look up phi(x) using linear interpolation between table entries.
    #[inline]
    pub fn lookup(&self, x: f64) -> f64 {
        assert!(x >= 0.0);

        if x >= self.x_max {
            return 0.0;
        }

        let frac_idx = x * self.dx_inv;
        let idx = frac_idx as usize;

        let diff = frac_idx - idx as f64;
        let i1 = (idx + 1).min(self.table.len() - 1);

        self.table[idx] + diff * (self.table[i1] - self.table[idx])
    }
}

#[derive(Clone)]
pub struct Edge {
    pub row: usize,
    pub col: usize,
    pub msg_vn_to_cn: f64,
    pub msg_cn_to_vn: f64,
}

#[derive(Clone)]
struct State {
    edges: Vec<Edge>,
    cn_ranges: Vec<Range<usize>>,
    vn_indices: Vec<Vec<usize>>,
    channel_llrs: Vec<f64>,
    total_llrs: Vec<f64>,
    num_vns: usize,
    num_cns: usize,
}

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct SpaComputeEngine {
    state: State,
    /// Lookup table used instead of computing tanh and arctanh directly
    phi_table: PhiTable,
    /// VN->CN messages are clipped to [-K,+K]
    K: f64,
}

impl BpComputeEngine for SpaComputeEngine {
    fn new(pcm: &ParityCheckMatrix) -> Self {
        let h_csr = pcm.h.to_csr();
        let num_cns = h_csr.rows();
        let num_vns = h_csr.cols();
        let nnz = h_csr.nnz();

        let mut edges = Vec::with_capacity(nnz);
        for (row, row_vec) in h_csr.outer_iterator().enumerate() {
            for (col, _) in row_vec.iter() {
                edges.push(Edge {
                    row,
                    col,
                    msg_vn_to_cn: 0.0,
                    msg_cn_to_vn: 0.0,
                });
            }
        }

        let mut cn_ranges = Vec::with_capacity(num_cns);
        for j in 0..num_cns {
            let start = pcm.h.indptr().index(j);
            let end = pcm.h.indptr().index(j + 1);
            cn_ranges.push(start..end);
        }

        let mut vn_indices = vec![Vec::new(); num_vns];
        for (k, edge) in edges.iter().enumerate() {
            vn_indices[edge.col].push(k);
        }

        Self {
            state: State {
                edges,
                cn_ranges,
                vn_indices,
                channel_llrs: vec![0.0; num_vns],
                total_llrs: vec![0.0; num_vns],
                num_vns,
                num_cns,
            },
            phi_table: PhiTable::new(2usize.pow(16), 25.0),
            K: 25.0,
        }
    }

    fn set_channel_llrs(&mut self, llrs: &[f64]) {
        self.state.channel_llrs.copy_from_slice(llrs);
    }

    fn reset(&mut self) {
        for edge in self.state.edges.iter_mut() {
            edge.msg_vn_to_cn = 0.0;
            edge.msg_cn_to_vn = 0.0;
        }
    }

    /// Perform check node update using the log-domain sum-product
    /// algorithm [1]. Uses the self-inverse function phi(x) = -ln(tanh(x/2))
    /// to replace tanh/atanh with table lookups.
    ///
    /// [1] H. Yao et al., "Belief Propagation Decoding of Quantum LDPC Codes
    ///     with Guided Decimation," arXiv:2312.10950, 2024.
    fn cn_update(&mut self, s: &[u8]) {
        for j in 0..self.state.num_cns {
            let neighboring_edge_indices = self.state.cn_ranges[j].clone();
            let syndrome_sign = 1.0 - 2.0 * s[j] as f64;

            let mut total_phi = 0.0;
            let mut total_sign = syndrome_sign;

            // Combine all incoming messages

            for edge_idx in neighboring_edge_indices.clone() {
                let msg_i_to_j = self.state.edges[edge_idx].msg_vn_to_cn;
                total_sign *= msg_i_to_j.signum();

                let phi = self.phi_table.lookup(msg_i_to_j.abs());
                total_phi += phi;
            }

            // Assign outgoing messages

            for edge_idx in neighboring_edge_indices {
                let msg_i_to_j = self.state.edges[edge_idx].msg_vn_to_cn;
                let phi = self.phi_table.lookup(msg_i_to_j.abs());
                let extrinsic_phi = total_phi - phi;
                let extrinsic_sign = total_sign * msg_i_to_j.signum();

                let out = extrinsic_sign * self.phi_table.lookup(extrinsic_phi);
                self.state.edges[edge_idx].msg_cn_to_vn = out;
            }
        }
    }

    /// Perform variable node update [1].
    ///
    /// [1] H. Yao et al., "Belief Propagation Decoding of Quantum LDPC Codes
    ///     with Guided Decimation," arXiv:2312.10950, 2024.
    fn vn_update(&mut self) {
        for i in 0..self.state.num_vns {
            let num_neighbors = self.state.vn_indices[i].len();

            // Combine all incoming messages

            self.state.total_llrs[i] = self.state.channel_llrs[i];
            for j in 0..num_neighbors {
                let edge_idx = self.state.vn_indices[i][j];
                self.state.total_llrs[i] +=
                    self.state.edges[edge_idx].msg_cn_to_vn;
            }

            // Assign outgoing messages

            for j in 0..num_neighbors {
                let edge_idx = self.state.vn_indices[i][j];
                let msg = self.state.total_llrs[i]
                    - self.state.edges[edge_idx].msg_cn_to_vn;

                self.state.edges[edge_idx].msg_vn_to_cn =
                    msg.clamp(-self.K, self.K);
            }
        }
    }

    fn total_llrs(&self) -> &[f64] {
        &self.state.total_llrs
    }
}

impl SpaComputeEngine {
    fn edge_id_to_edge_idx(&self, edge_id: EdgeId) -> usize {
        let i = (edge_id.0 & 0xFFFF) as usize;
        let j = (edge_id.0 >> 16) as usize;

        self.state.vn_indices[i]
            .iter()
            .find(|&&k| self.state.edges[k].row == j)
            .expect("Invalid edge id")
            .clone()
    }
}

impl AccessEngineInternals for SpaComputeEngine {
    fn get_cn_to_vn_msg(
        &self,
        edge_id: crate::decoders::engine::EdgeId,
    ) -> f64 {
        self.state.edges[self.edge_id_to_edge_idx(edge_id)].msg_cn_to_vn
    }

    fn get_vn_to_cn_msg(
        &self,
        edge_id: crate::decoders::engine::EdgeId,
    ) -> f64 {
        self.state.edges[self.edge_id_to_edge_idx(edge_id)].msg_vn_to_cn
    }

    fn get_channel_llr(&self, i: usize) -> f64 {
        self.state.channel_llrs[i]
    }

    fn set_cn_to_vn_msg(
        &mut self,
        edge_id: crate::decoders::engine::EdgeId,
        msg: f64,
    ) {
        let edge_idx = self.edge_id_to_edge_idx(edge_id);
        self.state.edges[edge_idx].msg_cn_to_vn = msg;
    }

    fn set_vn_to_cn_msg(
        &mut self,
        edge_id: crate::decoders::engine::EdgeId,
        msg: f64,
    ) {
        let edge_idx = self.edge_id_to_edge_idx(edge_id);
        self.state.edges[edge_idx].msg_vn_to_cn = msg;
    }

    fn set_channel_llr(&mut self, i: usize, llr: f64) {
        self.state.channel_llrs[i] = llr;
    }
}

#[cfg(test)]
mod tests {
    use sprs::{CsMat, TriMat};

    use super::*;

    #[test]
    fn test_phi_self_inverse() {
        let table = PhiTable::default();
        // Tolerance is looser for large x because phi(x) is tiny (≈ 2e^{-x}) and
        // falls in the steep near-zero region of the table where phi'' is large.
        for &(x, tol) in &[
            (0.5f64, 1e-4),
            (1.0, 1e-4),
            (2.0, 1e-3),
            (3.0, 1e-2),
            (5.0, 5e-2),
        ] {
            let y = table.lookup(x);
            let z = table.lookup(y);

            assert!(
                (z - x).abs() < tol,
                "phi(phi({x})) = {z}, expected {x} (tol {tol})"
            );
        }
    }

    #[test]
    fn test_phi_accuracy() {
        let table = PhiTable::default();

        for &x in &[0.5f64, 1.0, 2.0, 5.0] {
            let exact = -(x / 2.0).tanh().ln();
            let approx = table.lookup(x);

            assert!(
                (approx - exact).abs() < 1e-7,
                "phi({x}): got {approx}, exact {exact}"
            );
        }
    }

    #[test]
    fn test_phi_boundary() {
        let table = PhiTable::default();

        assert_eq!(table.lookup(25.0), 0.0);
        assert_eq!(table.lookup(100.0), 0.0);
    }

    fn get_hamming_h() -> CsMat<u8> {
        let mut h = TriMat::<u8>::new((3, 7));

        h.add_triplet(0, 0, 1);
        h.add_triplet(1, 1, 1);
        h.add_triplet(2, 2, 1);
        h.add_triplet(0, 3, 1);
        h.add_triplet(2, 3, 1);
        h.add_triplet(0, 4, 1);
        h.add_triplet(1, 4, 1);
        h.add_triplet(1, 5, 1);
        h.add_triplet(2, 5, 1);
        h.add_triplet(0, 6, 1);
        h.add_triplet(1, 6, 1);
        h.add_triplet(2, 6, 1);

        h.to_csr()
    }

    #[test]
    fn test_vn_update() {
        let h = get_hamming_h();
        let pcm = ParityCheckMatrix::new(&h.clone());

        let mut engine = SpaComputeEngine::new(&pcm);
        for edge in &mut engine.state.edges {
            edge.msg_cn_to_vn = 1.0;
        }

        engine.vn_update();

        //     1 0 0 1 1 0 1
        // H = 0 1 0 0 1 1 1
        //     0 0 1 1 0 1 1
        //
        //            1 0 0 1 1 0 1
        // L_{i<-j} = 0 1 0 0 1 1 1
        //            0 0 1 1 0 1 1

        let expected =
            vec![0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0];
        let got: Vec<f64> =
            engine.state.edges.iter().map(|e| e.msg_vn_to_cn).collect();
        assert_eq!(expected, got);
    }

    #[test]
    fn test_cn_update() {
        #[allow(non_snake_case)]
        let h = get_hamming_h();
        let pcm = ParityCheckMatrix::new(&h.clone());

        let mut engine = SpaComputeEngine::new(&pcm);
        for edge in &mut engine.state.edges {
            edge.msg_vn_to_cn = 1.0;
        }

        let s = vec![0, 1, 0];
        engine.cn_update(&s);

        //     1 0 0 1 1 0 1
        // H = 0 1 0 0 1 1 1
        //     0 0 1 1 0 1 1
        //
        //            1 0 0 1 1 0 1
        // L_{i->j} = 0 1 0 0 1 1 1
        //            0 0 1 1 0 1 1

        // The negative signs are due to the syndrome
        let v = 2.0 * f64::tanh(0.5).powi(3).atanh();
        let expected = vec![v, v, v, v, -v, -v, -v, -v, v, v, v, v];

        let got: Vec<f64> =
            engine.state.edges.iter().map(|e| e.msg_cn_to_vn).collect();

        assert_eq!(expected.len(), got.len());
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {a}, got {b}");
        }
    }

    #[test]
    fn test_total_llrs() {
        #[allow(non_snake_case)]
        let h = get_hamming_h();
        let pcm = ParityCheckMatrix::new(&h.clone());

        let channel_llrs = vec![1.0; h.cols()];
        let mut engine = SpaComputeEngine::new(&pcm);
        engine.set_channel_llrs(&channel_llrs);
        for edge in &mut engine.state.edges {
            edge.msg_cn_to_vn = 1.0;
        }

        engine.vn_update();

        //     1 0 0 1 1 0 1
        // H = 0 1 0 0 1 1 1
        //     0 0 1 1 0 1 1
        //
        //            1 0 0 1 1 0 1
        // L_{i<-j} = 0 1 0 0 1 1 1, L_ch = 1 1 1 1 1 1 1
        //            0 0 1 1 0 1 1

        let expected = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        assert_eq!(expected, engine.state.total_llrs);
    }

    #[test]
    fn test_access_engine_internals() {
        let h = get_hamming_h();

        let pcm = ParityCheckMatrix { h };
        let mut engine = SpaComputeEngine::new(&pcm);
        for (num, edge) in (engine.state.edges).iter_mut().enumerate() {
            edge.msg_vn_to_cn = (num + 1) as f64;
            edge.msg_cn_to_vn = (num + 1) as f64 * 10.0;
        }
        for i in 0..engine.state.num_vns {
            engine.set_channel_llr(i, (i + 1) as f64 * 100.0);
        }

        for (flat_idx, (j, i)) in [
            (0, 0),
            (0, 3),
            (0, 4),
            (0, 6),
            (1, 1),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 2),
            (2, 3),
            (2, 5),
            (2, 6),
        ]
        .iter()
        .enumerate()
        {
            let edge_id = pcm.get_edge_id(*j, *i).unwrap();

            assert_eq!(
                engine.get_vn_to_cn_msg(edge_id.clone()),
                (flat_idx as f64 + 1.0)
            );
            assert_eq!(
                engine.get_cn_to_vn_msg(edge_id.clone()),
                (flat_idx as f64 + 1.0) * 10.0
            );
        }

        for i in 0..engine.state.num_vns {
            assert_eq!(engine.get_channel_llr(i), (i + 1) as f64 * 100.0);
        }
    }
}
