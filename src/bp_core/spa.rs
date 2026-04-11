use crate::bp_core::{SyndromeBpCore, SyndromeBpStrategy};
use sprs::{CsMat, DenseVector};

pub struct SyndromeSpaCore(SyndromeBpCore);

// TODO: Clean this up
#[allow(non_snake_case)]
fn get_csc_to_csr(H: &CsMat<u8>) -> Vec<usize> {
    #[allow(non_snake_case)]
    let H_csc = H.to_csc();
    #[allow(non_snake_case)]
    let H_csr = H.to_csr();
    let nnz = H.nnz();

    // cols_csc[i] = which column the i-th CSC nonzero belongs to
    let mut cols_csc = Vec::with_capacity(nnz);
    for col in 0..H_csc.cols() {
        let count = H_csc.indptr().index(col + 1) - H_csc.indptr().index(col);
        cols_csc.extend(std::iter::repeat(col).take(count));
    }
    // rows_csc = H_csc.indices (inner indices of CSC are row indices)
    let rows_csc: Vec<usize> = H_csc.indices().to_vec();

    // rows_csr[i] = which row the i-th CSR nonzero belongs to
    let mut rows_csr = Vec::with_capacity(nnz);
    for row in 0..H_csr.rows() {
        let count = H_csr.indptr().index(row + 1) - H_csr.indptr().index(row);
        rows_csr.extend(std::iter::repeat(row).take(count));
    }
    // cols_csr = H_csr.indices (inner indices of CSR are column indices)
    let cols_csr: Vec<usize> = H_csr.indices().to_vec();

    // csc_order = argsort of CSC nonzeros by (row, col)
    // np.lexsort((cols_csc, rows_csc)) sorts primarily by rows_csc, secondarily by cols_csc
    let mut csc_order: Vec<usize> = (0..nnz).collect();
    csc_order.sort_by_key(|&i| (rows_csc[i], cols_csc[i]));

    // csr_order = argsort of CSR nonzeros by (row, col)
    let mut csr_order: Vec<usize> = (0..nnz).collect();
    csr_order.sort_by_key(|&i| (rows_csr[i], cols_csr[i]));

    // csr_to_csc[csr_order[k]] = csc_order[k]
    // i.e. for the k-th nonzero in (row,col) order, map its CSR index to its CSC index
    let mut csr_to_csc = vec![0usize; nnz];
    for k in 0..nnz {
        csr_to_csc[csr_order[k]] = csc_order[k];
    }

    csr_to_csc
}

impl SyndromeBpStrategy for SyndromeSpaCore {
    #[allow(non_snake_case)]
    fn new(H: &CsMat<u8>, channel_llrs: &[f64]) -> Self {
        Self {
            0: SyndromeBpCore {
                H_csc: H.to_csc(),
                H_csr: H.to_csr(),
                csr_to_csc: get_csc_to_csr(&H),
                channel_llrs: Vec::<f64>::from(channel_llrs),
                msg_cn_to_vn: Vec::<f64>::zeros(H.nnz()),
                msg_vn_to_cn: Vec::<f64>::zeros(H.nnz()),
                total_llrs: Vec::<f64>::zeros(H.cols()),
            },
        }
    }

    /// Perform variable node update [1].
    ///
    /// For each variable node `i`, the outgoing message to each connected check
    /// node is the sum of the channel LLR and all incoming CN messages, minus
    /// the message from that specific check node. Messages are stored in CSC
    /// order; see [2] for the index structure.
    ///
    /// # References
    ///
    /// [1] H. Yao et al., "Belief Propagation Decoding of Quantum LDPC Codes
    ///     with Guided Decimation," arXiv:2312.10950, 2024.
    /// [2] https://stackoverflow.com/a/52299730
    fn vn_update(&mut self) {
        for i in 0..self.0.H_csc.cols() {
            let col_i_indices = self.0.H_csc.indptr().index(i)
                ..self.0.H_csc.indptr().index(i + 1);

            let mut total = self.0.channel_llrs[i];
            for idx in col_i_indices.clone() {
                total += self.0.msg_cn_to_vn[idx];
            }

            for idx in col_i_indices {
                self.0.msg_vn_to_cn[idx] = total - self.0.msg_cn_to_vn[idx]
            }
        }
    }

    /// Perform check node update using the sum-product
    /// algorithm (SPA) [1, Eq. 21].
    ///
    /// For each check node `j`, the outgoing message to each connected
    /// variable node is computed via the tanh rule. `csr_to_csc` maps CSR
    /// positions to CSC positions when looking up incoming VN messages.
    /// Messages are stored in CSC order; see [2] for the index structure.
    ///
    /// # Arguments
    ///
    /// - `syndrome` - Measured syndrome bits, one per check node.
    ///
    /// # References
    ///
    /// [1] H. Yao et al., "Belief Propagation Decoding of Quantum LDPC Codes
    ///     with Guided Decimation," arXiv:2312.10950, 2024.
    /// [2] https://stackoverflow.com/a/52299730
    // TODO: Fix clipping
    fn cn_update(&mut self, s: &[u8]) {
        for j in 0..self.0.H_csr.rows() {
            let start = self.0.H_csr.indptr().index(j);
            let end = self.0.H_csr.indptr().index(j + 1);

            let syndrome_sign = 1.0 - 2.0 * s[j] as f64;

            let mut proc_total = 1.0;
            for idx in start..end {
                let msg = self.0.msg_vn_to_cn[self.0.csr_to_csc[idx]];
                proc_total *= (msg / 2.0).tanh();
            }

            for idx in start..end {
                let msg = self.0.msg_vn_to_cn[self.0.csr_to_csc[idx]];
                let tanh_val = (msg / 2.0).tanh();

                // Guard against division by zero
                let prod_extrinsic = if tanh_val.abs() < 1e-15 {
                    // Recompute without this edge
                    let mut p = 1.0;
                    for idx_prime in start..end {
                        if idx_prime == idx {
                            continue;
                        }
                        p *= (self.0.msg_vn_to_cn
                            [self.0.csr_to_csc[idx_prime]]
                            / 2.0)
                            .tanh();
                    }
                    p
                } else {
                    proc_total / tanh_val
                };

                let clamped = prod_extrinsic.clamp(-1.0 + 1e-7, 1.0 - 1e-7);
                self.0.msg_cn_to_vn[self.0.csr_to_csc[idx]] =
                    2.0 * syndrome_sign * clamped.atanh();
            }
        }
    }
    fn total_llrs(&mut self) {
        for i in 0..self.0.H_csc.cols() {
            let col_i_indices = self.0.H_csc.indptr().index(i)
                ..self.0.H_csc.indptr().index(i + 1);

            self.0.total_llrs[i] = self.0.channel_llrs[i];
            for idx in col_i_indices {
                self.0.total_llrs[i] += self.0.msg_cn_to_vn[idx];
            }
        }
    }

    fn get_state(&mut self) -> &mut SyndromeBpCore {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use sprs::{CsMat, TriMat};

    use super::*;

    #[test]
    fn test_get_csc_to_csr() {
        let mut m = TriMat::<u8>::new((3, 7));
        m.add_triplet(0, 0, 1);
        m.add_triplet(1, 1, 2);
        m.add_triplet(2, 2, 3);
        m.add_triplet(0, 3, 4);
        m.add_triplet(2, 3, 5);
        m.add_triplet(0, 4, 6);
        m.add_triplet(1, 4, 7);
        m.add_triplet(1, 5, 8);
        m.add_triplet(2, 5, 9);
        m.add_triplet(0, 6, 10);
        m.add_triplet(1, 6, 11);
        m.add_triplet(2, 6, 12);
        let m_csc: CsMat<u8> = m.to_csc();
        let m_csr: CsMat<u8> = m.to_csr();

        let csr_to_csc = get_csc_to_csr(&m_csc);

        for idx in 0..m_csc.nnz() {
            assert_eq!(m_csr.data()[idx], m_csc.data()[csr_to_csc[idx]]);
        }
    }

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
        let nnz = H.nnz();

        let channel_llrs = Vec::<f64>::zeros(H.cols());
        let mut decoder = SyndromeSpaCore::new(&H, &channel_llrs[..]);
        decoder.0.msg_cn_to_vn = (0..nnz).map(|_| 1.0).collect::<Vec<f64>>();

        decoder.vn_update();

        let msg_vn_to_cn_expected =
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        assert_eq!(msg_vn_to_cn_expected, decoder.0.msg_vn_to_cn);
    }

    #[test]
    fn test_cn_update() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();
        let nnz = H.nnz();

        let channel_llrs = Vec::<f64>::zeros(H.cols());
        let mut decoder = SyndromeSpaCore::new(&H, &channel_llrs[..]);
        decoder.0.msg_vn_to_cn = vec![1.0; nnz];

        let s = vec![0, 1, 0];
        decoder.cn_update(&s);

        // Expected values in CSC order. Magnitude = 2*arctanh(tanh(0.5)^3).
        // Sign follows (-1)^syndrome[check_node].
        let v = 2.0 * f64::tanh(0.5).powi(3).atanh();
        let expected = vec![v, -v, v, v, v, v, -v, -v, v, v, -v, v];

        assert_eq!(expected.len(), decoder.0.msg_cn_to_vn.len());
        for (a, b) in expected.iter().zip(decoder.0.msg_cn_to_vn.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {a}, got {b}");
        }
    }

    #[test]
    fn test_total_llrs() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();
        let nnz = H.nnz();

        let channel_llrs = vec![1.0; H.cols()];
        let mut decoder = SyndromeSpaCore::new(&H, &channel_llrs[..]);
        decoder.0.msg_cn_to_vn = vec![1.0; nnz];

        decoder.total_llrs();

        // total_llrs[i] = channel_llrs[i] + degree(col i)
        // degrees: cols 0-2 have degree 1, cols 3-5 degree 2, col 6 degree 3
        let expected = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        assert_eq!(expected, decoder.0.total_llrs);
    }
}
