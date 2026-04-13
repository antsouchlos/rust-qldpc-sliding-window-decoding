use crate::decoders::core::{Edge, SyndromeBpCore, SyndromeBpStrategy};
use sprs::CsMat;

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct SyndromeNaiveSpaCore {
    state: SyndromeBpCore,
}

// TODO: Don't hardcode clipping value
impl SyndromeBpStrategy for SyndromeNaiveSpaCore {
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

            let mut total = self.state.channel_llrs[i];
            for j in 0..num_neighbors {
                let edge_idx = self.state.vn_indices[i][j];
                total += self.state.edges[edge_idx].msg_cn_to_vn;
            }

            // Assign outgoing messages

            for j in 0..num_neighbors {
                let edge_idx = self.state.vn_indices[i][j];
                let msg = total - self.state.edges[edge_idx].msg_cn_to_vn;

                self.state.edges[edge_idx].msg_vn_to_cn = msg;
            }
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

            let mut total_prod = 1.0;

            // Combine all incoming messages

            for edge_idx in neighboring_edge_indices.clone() {
                let msg_i_to_j = self.state.edges[edge_idx].msg_vn_to_cn;
                total_prod *= (msg_i_to_j / 2.0).tanh();
            }

            // Assign outgoing messages

            for edge_idx in neighboring_edge_indices {
                let msg_i_to_j = self.state.edges[edge_idx].msg_vn_to_cn;

                let mut extrinsic_prod = total_prod / (msg_i_to_j / 2.0).tanh();

                extrinsic_prod = extrinsic_prod.clamp(-1.0 + 1e-7, 1.0 - 1e-7);

                self.state.edges[edge_idx].msg_cn_to_vn =
                    2.0 * syndrome_sign * extrinsic_prod.atanh();
            }
        }
    }

    fn total_llrs(&mut self) {
        for i in 0..self.state.num_vns {
            self.state.total_llrs[i] = self.state.channel_llrs[i];

            let num_neighbors = self.state.vn_indices[i].len();
            for j in 0..num_neighbors {
                let edge_idx = self.state.vn_indices[i][j];
                self.state.total_llrs[i] +=
                    self.state.edges[edge_idx].msg_cn_to_vn;
            }
        }
    }

    fn get_state(&mut self) -> &mut SyndromeBpCore {
        &mut self.state
    }

    fn get_state_ref(&self) -> &SyndromeBpCore {
        &self.state
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
        let mut decoder = SyndromeNaiveSpaCore::new(&H, &channel_llrs);
        for edge in &mut decoder.state.edges {
            edge.msg_cn_to_vn = 1.0;
        }

        decoder.vn_update();

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
            decoder.state.edges.iter().map(|e| e.msg_vn_to_cn).collect();
        assert_eq!(expected, got);
    }

    #[test]
    fn test_cn_update() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();

        let channel_llrs = vec![0.0; H.cols()];
        let mut decoder = SyndromeNaiveSpaCore::new(&H, &channel_llrs);
        for edge in &mut decoder.state.edges {
            edge.msg_vn_to_cn = 1.0;
        }

        let s = vec![0, 1, 0];
        decoder.cn_update(&s);

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
        let mut decoder = SyndromeNaiveSpaCore::new(&H, &channel_llrs);
        for edge in &mut decoder.state.edges {
            edge.msg_cn_to_vn = 1.0;
        }

        decoder.total_llrs();

        //     1 0 0 1 1 0 1
        // H = 0 1 0 0 1 1 1
        //     0 0 1 1 0 1 1
        //
        //            1 0 0 1 1 0 1
        // L_{i<-j} = 0 1 0 0 1 1 1, L_ch = 1 1 1 1 1 1 1
        //            0 0 1 1 0 1 1

        let expected = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        assert_eq!(expected, decoder.state.total_llrs);
    }
}
