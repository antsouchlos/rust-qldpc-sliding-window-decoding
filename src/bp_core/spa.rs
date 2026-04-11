use crate::bp_core::{Edge, SyndromeBpCore, SyndromeBpStrategy};
use crate::phi_table::PhiTable;
use sprs::CsMat;

pub struct SyndromeSpaCore {
    pub state: SyndromeBpCore,
    phi_table: PhiTable,
}

impl SyndromeBpStrategy for SyndromeSpaCore {
    #[allow(non_snake_case)]
    fn new(H: &CsMat<u8>, channel_llrs: &[f64]) -> Self {
        let H_csr = H.to_csr();
        let num_cns = H_csr.rows();
        let num_vns = H_csr.cols();
        let nnz = H_csr.nnz();

        let mut edges = Vec::with_capacity(nnz);
        for (row, row_vec) in H_csr.outer_iterator().enumerate() {
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
            let start = H_csr.indptr().index(j);
            let end = H_csr.indptr().index(j + 1);
            cn_ranges.push(start..end);
        }

        let mut vn_indices = vec![Vec::new(); num_vns];
        for (k, edge) in edges.iter().enumerate() {
            vn_indices[edge.col].push(k);
        }

        Self {
            state: SyndromeBpCore {
                edges,
                cn_ranges,
                vn_indices,
                channel_llrs: channel_llrs.to_vec(),
                total_llrs: vec![0.0; num_vns],
                num_vns,
                num_cns,
            },
            phi_table: PhiTable::default(),
        }
    }

    /// Perform variable node update [1].
    ///
    /// For each variable node `i`, the outgoing message to each connected check
    /// node is the sum of the channel LLR and all incoming CN messages, minus
    /// the message from that specific check node.
    ///
    /// # References
    ///
    /// [1] H. Yao et al., "Belief Propagation Decoding of Quantum LDPC Codes
    ///     with Guided Decimation," arXiv:2312.10950, 2024.
    fn vn_update(&mut self) {
        for i in 0..self.state.num_vns {
            let mut total = self.state.channel_llrs[i];
            let num_edges = self.state.vn_indices[i].len();
            for ki in 0..num_edges {
                let k = self.state.vn_indices[i][ki];
                total += self.state.edges[k].msg_cn_to_vn;
            }
            for ki in 0..num_edges {
                let k = self.state.vn_indices[i][ki];
                self.state.edges[k].msg_vn_to_cn =
                    total - self.state.edges[k].msg_cn_to_vn;
            }
        }
    }

    /// Perform check node update using the log-domain sum-product algorithm [1].
    ///
    /// Uses the identity phi(x) = -ln(tanh(x/2)) (self-inverse) to replace
    /// tanh/atanh with table lookups:
    ///
    ///   |L_{j→i}| = phi(∑_{k≠i} phi(|L_{k→j}|))
    ///   sign(L_{j→i}) = syndrome_sign_j · ∏_{k≠i} sign(L_{k→j})
    ///
    /// Two passes per check node: accumulate phi-sum and sign in pass 1,
    /// subtract each edge's contribution and look up the result in pass 2.
    /// This is O(d) per check node with no transcendental function calls.
    ///
    /// # References
    ///
    /// [1] H. Yao et al., "Belief Propagation Decoding of Quantum LDPC Codes
    ///     with Guided Decimation," arXiv:2312.10950, 2024.
    fn cn_update(&mut self, s: &[u8]) {
        for j in 0..self.state.num_cns {
            let range = self.state.cn_ranges[j].clone();
            let syndrome_sign = 1.0 - 2.0 * s[j] as f64;

            let mut total_phi = 0.0;
            let mut total_sign = syndrome_sign;

            for k in range.clone() {
                let msg = self.state.edges[k].msg_vn_to_cn;
                let phi_k = self.phi_table.lookup(msg.abs());
                total_phi += phi_k;
                total_sign *= msg.signum();
            }

            for k in range {
                let msg = self.state.edges[k].msg_vn_to_cn;
                let phi_k = self.phi_table.lookup(msg.abs());
                let excl_phi = total_phi - phi_k;
                let excl_sign = total_sign * msg.signum();
                let out = excl_sign * self.phi_table.lookup(excl_phi);
                self.state.edges[k].msg_cn_to_vn = out;
            }
        }
    }

    fn total_llrs(&mut self) {
        for i in 0..self.state.num_vns {
            self.state.total_llrs[i] = self.state.channel_llrs[i];
            let num_edges = self.state.vn_indices[i].len();
            for ki in 0..num_edges {
                let k = self.state.vn_indices[i][ki];
                self.state.total_llrs[i] += self.state.edges[k].msg_cn_to_vn;
            }
        }
    }

    fn get_state(&mut self) -> &mut SyndromeBpCore {
        &mut self.state
    }
}

#[cfg(test)]
mod tests {
    use sprs::TriMat;

    use super::*;

    #[allow(non_snake_case)]
    fn get_hamming_H() -> CsMat<u8> {
        #[allow(non_snake_case)]
        let mut H = TriMat::<u8>::new((3, 7));

        H.add_triplet(0, 0, 1);
        H.add_triplet(1, 1, 1);
        H.add_triplet(2, 2, 1);
        H.add_triplet(0, 3, 1);
        H.add_triplet(2, 3, 1);
        H.add_triplet(0, 4, 1);
        H.add_triplet(1, 4, 1);
        H.add_triplet(1, 5, 1);
        H.add_triplet(2, 5, 1);
        H.add_triplet(0, 6, 1);
        H.add_triplet(1, 6, 1);
        H.add_triplet(2, 6, 1);

        H.to_csr()
    }

    #[test]
    fn test_vn_update() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();

        let channel_llrs = vec![0.0; H.cols()];
        let mut decoder = SyndromeSpaCore::new(&H, &channel_llrs);
        for edge in &mut decoder.state.edges {
            edge.msg_cn_to_vn = 1.0;
        }

        decoder.vn_update();

        // Edges are in CSR order: (0,0),(0,3),(0,4),(0,6),(1,1),(1,4),(1,5),(1,6),(2,2),(2,3),(2,5),(2,6)
        // col degrees: 0→1, 1→1, 2→1, 3→2, 4→2, 5→2, 6→3
        // msg = (channel_llr + sum_incoming) - this_incoming = degree(col)-1
        let expected =
            vec![0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0];
        let got: Vec<f64> =
            decoder.state.edges.iter().map(|e| e.msg_vn_to_cn).collect();
        assert_eq!(expected, got);
    }

    #[test]
    fn test_cn_update() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();

        let channel_llrs = vec![0.0; H.cols()];
        let mut decoder = SyndromeSpaCore::new(&H, &channel_llrs);
        for edge in &mut decoder.state.edges {
            edge.msg_vn_to_cn = 1.0;
        }

        let s = vec![0, 1, 0];
        decoder.cn_update(&s);

        // Edges in CSR order: rows 0,0,0,0 then 1,1,1,1 then 2,2,2,2
        // syndrome_sign: row 0 = +1, row 1 = -1, row 2 = +1
        // Magnitude = 2*arctanh(tanh(0.5)^3) (product of 3 tanh values, 4 edges per row)
        let v = 2.0 * f64::tanh(0.5).powi(3).atanh();
        let expected = vec![v, v, v, v, -v, -v, -v, -v, v, v, v, v];

        let got: Vec<f64> =
            decoder.state.edges.iter().map(|e| e.msg_cn_to_vn).collect();
        assert_eq!(expected.len(), got.len());
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {a}, got {b}");
        }
    }

    #[test]
    fn test_total_llrs() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();

        let channel_llrs = vec![1.0; H.cols()];
        let mut decoder = SyndromeSpaCore::new(&H, &channel_llrs);
        for edge in &mut decoder.state.edges {
            edge.msg_cn_to_vn = 1.0;
        }

        decoder.total_llrs();

        // total_llrs[i] = channel_llrs[i] + degree(col i)
        // degrees: cols 0-2 have degree 1, cols 3-5 degree 2, col 6 degree 3
        let expected = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        assert_eq!(expected, decoder.state.total_llrs);
    }
}
