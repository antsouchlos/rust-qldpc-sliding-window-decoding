use std::ops::Range;

use crate::decoders::engine::{
    AccessEngineInternals, BpComputeEngine, EdgeId, ParityCheckMatrix,
};

#[derive(Clone)]
pub struct Edge {
    pub row: usize,
    pub col: usize,
    pub msg_vn_to_cn: f64,
    pub msg_cn_to_vn: f64,
}

pub struct MinSumComputeEngine {
    pub edges: Vec<Edge>,
    pub cn_ranges: Vec<Range<usize>>,
    pub vn_indices: Vec<Vec<usize>>,
    pub channel_llrs: Vec<f64>,
    pub total_llrs: Vec<f64>,
    pub num_vns: usize,
    pub num_cns: usize,
}

impl BpComputeEngine for MinSumComputeEngine {
    type Llr = f64;

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
            edges,
            cn_ranges,
            vn_indices,
            channel_llrs: vec![0.0; num_vns],
            total_llrs: vec![0.0; num_vns],
            num_vns,
            num_cns,
        }
    }

    fn set_channel_llrs(&mut self, llrs: &[Self::Llr]) {
        self.channel_llrs.copy_from_slice(llrs);
    }

    fn reset(&mut self) {
        for edge in self.edges.iter_mut() {
            edge.msg_vn_to_cn = 0.0;
            edge.msg_cn_to_vn = 0.0;
        }
    }

    /// Perform check node update using the min-sum approximation.
    ///
    /// To avoid having to do two passes to account for the extrinsic
    /// principle, the two minimum values are found in one pass.
    fn cn_update(&mut self, syndrome: &[u8]) {
        for j in 0..self.num_cns {
            let range = self.cn_ranges[j].clone();
            let syndrome_sign = 1.0 - 2.0 * syndrome[j] as f64;
            let mut total_sign = syndrome_sign;
            let mut min1 = f64::INFINITY;
            let mut min2 = f64::INFINITY;
            let mut min1_idx = range.start;

            // Combine all incoming messages

            for edge_idx in range.clone() {
                let msg = self.edges[edge_idx].msg_vn_to_cn;
                let abs_msg = msg.abs();

                total_sign *= msg.signum();

                if abs_msg < min1 {
                    min2 = min1;
                    min1 = abs_msg;
                    min1_idx = edge_idx;
                } else if abs_msg < min2 {
                    min2 = abs_msg;
                }
            }

            // Assign outgoing messages

            for edge_idx in range {
                let msg = self.edges[edge_idx].msg_vn_to_cn;

                let extrinsic_sign = total_sign * msg.signum();
                let extrinsic_mag =
                    if edge_idx == min1_idx { min2 } else { min1 };

                self.edges[edge_idx].msg_cn_to_vn =
                    extrinsic_sign * extrinsic_mag;
            }
        }
    }

    /// Perform variable node update [1].
    ///
    /// This function also computes the total LLRs.
    ///
    /// [1] H. Yao et al., "Belief Propagation Decoding of Quantum LDPC Codes
    ///     with Guided Decimation," arXiv:2312.10950, 2024.
    fn vn_update(&mut self) {
        for i in 0..self.num_vns {
            let num_neighbors = self.vn_indices[i].len();

            // Combine all incoming messages

            self.total_llrs[i] = self.channel_llrs[i];
            for j_idx in 0..num_neighbors {
                let idx = self.vn_indices[i][j_idx];
                self.total_llrs[i] += self.edges[idx].msg_cn_to_vn;
            }

            // Assign outgoing messages

            for j_idx in 0..num_neighbors {
                let edge_idx = self.vn_indices[i][j_idx];
                self.edges[edge_idx].msg_vn_to_cn =
                    self.total_llrs[i] - self.edges[edge_idx].msg_cn_to_vn;
            }
        }
    }

    fn total_llrs(&self) -> &[Self::Llr] {
        &self.total_llrs
    }
}

impl MinSumComputeEngine {
    fn edge_id_to_edge_idx(&self, edge_id: EdgeId) -> usize {
        let i = (edge_id.0 & 0xFFFF) as usize;
        let j = (edge_id.0 >> 16) as usize;

        self.vn_indices[i]
            .iter()
            .find(|&&k| self.edges[k].row == j)
            .expect("Invalid edge id")
            .clone()
    }
}

impl AccessEngineInternals for MinSumComputeEngine {
    fn get_cn_to_vn_msg(
        &self,
        edge_id: crate::decoders::engine::EdgeId,
    ) -> Self::Llr {
        self.edges[self.edge_id_to_edge_idx(edge_id)].msg_cn_to_vn
    }

    fn get_vn_to_cn_msg(
        &self,
        edge_id: crate::decoders::engine::EdgeId,
    ) -> Self::Llr {
        self.edges[self.edge_id_to_edge_idx(edge_id)].msg_vn_to_cn
    }

    fn get_channel_llr(&self, i: usize) -> Self::Llr {
        self.channel_llrs[i]
    }

    fn set_cn_to_vn_msg(
        &mut self,
        edge_id: crate::decoders::engine::EdgeId,
        msg: Self::Llr,
    ) {
        let edge_idx = self.edge_id_to_edge_idx(edge_id);
        self.edges[edge_idx].msg_cn_to_vn = msg;
    }

    fn set_vn_to_cn_msg(
        &mut self,
        edge_id: crate::decoders::engine::EdgeId,
        msg: Self::Llr,
    ) {
        let edge_idx = self.edge_id_to_edge_idx(edge_id);
        self.edges[edge_idx].msg_vn_to_cn = msg;
    }

    fn set_channel_llr(&mut self, i: usize, llr: Self::Llr) {
        self.channel_llrs[i] = llr;
    }
}

#[cfg(test)]
mod tests {
    use sprs::{CsMat, TriMat};

    use super::*;

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

        let mut engine = MinSumComputeEngine::new(&ParityCheckMatrix { h });
        for edge in &mut engine.edges {
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
            engine.edges.iter().map(|e| e.msg_vn_to_cn).collect();
        assert_eq!(expected, got);
    }

    #[test]
    fn test_cn_update() {
        let h = get_hamming_h();

        let mut engine = MinSumComputeEngine::new(&ParityCheckMatrix { h });
        for (num, edge) in (engine.edges).iter_mut().enumerate() {
            edge.msg_vn_to_cn = (num + 1) as f64;
        }

        let s = vec![0, 1, 0];
        engine.cn_update(&s);

        //     1 0 0 1 1 0 1
        // H = 0 1 0 0 1 1 1
        //     0 0 1 1 0 1 1
        //
        //            1 0 0 2  3 0  4
        // L_{i->j} = 0 5 0 0  6 7  8
        //            0 0 9 10 0 11 12

        // The negative signs are due to the syndrome
        let expected = vec![
            2.0, 1.0, 1.0, 1.0, -6.0, -5.0, -5.0, -5.0, 10.0, 9.0, 9.0, 9.0,
        ];
        let got: Vec<f64> =
            engine.edges.iter().map(|e| e.msg_cn_to_vn).collect();
        assert_eq!(expected, got);
    }

    #[test]
    fn test_total_llrs() {
        let h = get_hamming_h();

        let channel_llrs = vec![1.0; h.cols()];
        let mut engine = MinSumComputeEngine::new(&ParityCheckMatrix { h });
        engine.set_channel_llrs(&channel_llrs);
        for edge in &mut engine.edges {
            edge.msg_cn_to_vn = 1.0;
        }

        engine.vn_update(); //< Implicitly computes total llrs

        //     1 0 0 1 1 0 1
        // H = 0 1 0 0 1 1 1
        //     0 0 1 1 0 1 1
        //
        //            1 0 0 1 1 0 1
        // L_{i<-j} = 0 1 0 0 1 1 1, L_ch = 1 1 1 1 1 1 1
        //            0 0 1 1 0 1 1

        let expected = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        assert_eq!(expected, engine.total_llrs);
    }

    #[test]
    fn test_access_engine_internals() {
        let h = get_hamming_h();

        let pcm = ParityCheckMatrix { h };
        let mut engine = MinSumComputeEngine::new(&pcm);
        for (num, edge) in (engine.edges).iter_mut().enumerate() {
            edge.msg_vn_to_cn = (num + 1) as f64;
            edge.msg_cn_to_vn = (num + 1) as f64 * 10.0;
        }
        for i in 0..engine.num_vns {
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

        for i in 0..engine.num_vns {
            assert_eq!(engine.get_channel_llr(i), (i + 1) as f64 * 100.0);
        }
    }
}
