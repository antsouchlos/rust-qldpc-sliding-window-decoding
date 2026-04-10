use std::iter::zip;

use crate::Decoder;
use sprs::{CsMat, DenseVector};

#[derive(PartialEq)]
enum BpMethod {
    Spa,
    MinSum,
}

pub struct Settings {
    max_iter: usize,
    bp_method: BpMethod,
}

#[allow(non_snake_case)]
struct State {
    H_csc: CsMat<u8>,
    H_csr: CsMat<u8>,
    csr_to_csc: Vec<usize>,
    channel_llrs: Vec<f64>,
    msg_cn_to_vn: Vec<f64>,
    msg_vn_to_cn: Vec<f64>,
    total_llrs: Vec<f64>,
}

pub struct BpDecoder {
    settings: Settings,
    state: State,
}

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

impl BpDecoder {
    #[allow(non_snake_case)]
    pub fn new(
        settings: Settings,
        H: &CsMat<u8>,
        channel_llrs: &[f64],
    ) -> Self {
        let state = State {
            H_csc: H.to_csc(),
            H_csr: H.to_csr(),
            csr_to_csc: get_csc_to_csr(&H),
            channel_llrs: Vec::<f64>::from(channel_llrs),
            msg_cn_to_vn: Vec::<f64>::zeros(H.nnz()),
            msg_vn_to_cn: Vec::<f64>::zeros(H.nnz()),
            total_llrs: Vec::<f64>::zeros(H.cols()),
        };

        Self { settings, state }
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
        for i in 0..self.state.H_csc.cols() {
            let col_i_indices = self.state.H_csc.indptr().index(i)
                ..self.state.H_csc.indptr().index(i + 1);

            let mut total = self.state.channel_llrs[i];
            for idx in col_i_indices.clone() {
                total += self.state.msg_cn_to_vn[idx];
            }

            for idx in col_i_indices {
                self.state.msg_vn_to_cn[idx] =
                    total - self.state.msg_cn_to_vn[idx]
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
    fn cn_update_spa(&mut self, s: &[u8]) {
        for j in 0..self.state.H_csc.rows() {
            let row_j_indices_csr = self.state.H_csr.indptr().index(j)
                ..self.state.H_csr.indptr().index(j + 1);

            for idx in row_j_indices_csr.clone() {
                let mut prod = 1.0;

                for idx_prime in row_j_indices_csr.clone() {
                    if idx_prime == idx {
                        continue;
                    }
                    prod *= (self.state.msg_vn_to_cn
                        [self.state.csr_to_csc[idx_prime]]
                        / 2.0)
                        .tanh();
                }

                if prod > 1.0 - 1e-7 {
                    prod = 1.0 - 1e-7;
                } else if prod < -1.0 + 1e-7 {
                    prod = -1.0 + 1e-7;
                }

                self.state.msg_cn_to_vn[self.state.csr_to_csc[idx]] =
                    2.0 * (1.0 - 2.0 * s[j] as f64) * prod.atanh();
            }
        }
    }

    fn cn_update_min_sum(&mut self, syndrome: &[u8]) {
        for j in 0..self.state.H_csr.rows() {
            let row_j_indices_csr = self.state.H_csr.indptr().index(j)
                ..self.state.H_csr.indptr().index(j + 1);

            for idx in row_j_indices_csr.clone() {
                let mut sign = 1.0 - (2 * syndrome[j]) as f64;
                let mut min_mag = f64::INFINITY;

                for idx_prime in row_j_indices_csr.clone() {
                    if idx_prime == idx {
                        continue;
                    }
                    let msg = self.state.msg_vn_to_cn
                        [self.state.csr_to_csc[idx_prime]];
                    sign *= msg.signum();
                    min_mag = min_mag.min(msg.abs());
                }

                self.state.msg_cn_to_vn[self.state.csr_to_csc[idx]] =
                    sign * min_mag;
            }
        }
    }

    fn total_llrs(&mut self) {
        for i in 0..self.state.H_csc.cols() {
            let col_i_indices = self.state.H_csc.indptr().index(i)
                ..self.state.H_csc.indptr().index(i + 1);

            self.state.total_llrs[i] = self.state.channel_llrs[i];
            for idx in col_i_indices {
                self.state.total_llrs[i] += self.state.msg_cn_to_vn[idx];
            }
        }
    }
}

#[allow(non_snake_case)]
fn compute_syndrome(H: &CsMat<u8>, e_hat: &[u8]) -> Vec<u8> {
    let mut syndrome = vec![0u8; H.rows()];
    for (row_idx, row) in H.to_csr().outer_iterator().enumerate() {
        for (col_idx, _) in row.iter() {
            syndrome[row_idx] ^= e_hat[col_idx];
        }
    }
    syndrome
}

// TODO: Doc comments
impl Decoder for BpDecoder {
    fn decode(&mut self, s: &[u8]) -> Vec<u8> {
        let mut e_hat: Vec<u8> = self
            .state
            .channel_llrs
            .iter()
            .map(|&v| if v < 0.0 { 1 } else { 0 })
            .collect();

        for _ in 0..self.settings.max_iter {
            self.vn_update();
            if self.settings.bp_method == BpMethod::Spa {
                self.cn_update_spa(s);
            } else if self.settings.bp_method == BpMethod::MinSum {
                self.cn_update_min_sum(s);
            }
            self.total_llrs();

            e_hat = self
                .state
                .channel_llrs
                .iter()
                .map(|&v| if v < 0.0 { 1 } else { 0 })
                .collect();

            let s_hat = compute_syndrome(&self.state.H_csc, &e_hat);
            if s_hat == s {
                break;
            }
        }

        e_hat
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

        let settings = Settings {
            max_iter: 100,
            bp_method: BpMethod::Spa,
        };

        let channel_llrs = Vec::<f64>::zeros(H.cols());
        let mut decoder = BpDecoder::new(settings, &H, &channel_llrs[..]);
        decoder.state.msg_cn_to_vn =
            (0..nnz).map(|_| 1.0).collect::<Vec<f64>>();

        decoder.vn_update();

        let msg_vn_to_cn_expected =
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        assert_eq!(msg_vn_to_cn_expected, decoder.state.msg_vn_to_cn);
    }

    #[test]
    fn test_cn_update_spa() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();
        let nnz = H.nnz();

        let settings = Settings {
            max_iter: 100,
            bp_method: BpMethod::Spa,
        };

        let channel_llrs = Vec::<f64>::zeros(H.cols());
        let mut decoder = BpDecoder::new(settings, &H, &channel_llrs[..]);
        decoder.state.msg_vn_to_cn = vec![1.0; nnz];

        let s = vec![0, 1, 0];
        decoder.cn_update_spa(&s);

        // Expected values in CSC order. Magnitude = 2*arctanh(tanh(0.5)^3).
        // Sign follows (-1)^syndrome[check_node].
        let v = 2.0 * f64::tanh(0.5).powi(3).atanh();
        let expected = vec![v, -v, v, v, v, v, -v, -v, v, v, -v, v];

        assert_eq!(expected.len(), decoder.state.msg_cn_to_vn.len());
        for (a, b) in expected.iter().zip(decoder.state.msg_cn_to_vn.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {a}, got {b}");
        }
    }

    #[test]
    fn test_cn_update_min_sum() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();
        let nnz = H.nnz();

        let settings = Settings {
            max_iter: 100,
            bp_method: BpMethod::MinSum,
        };

        let channel_llrs = Vec::<f64>::zeros(H.cols());
        let mut decoder = BpDecoder::new(settings, &H, &channel_llrs[..]);
        decoder.state.msg_vn_to_cn = vec![1.0; nnz];

        let s = vec![0, 1, 0];
        decoder.cn_update_min_sum(&s);

        // Magnitude = min(|vn_msgs|) = 1.0; sign follows
        // (-1)^syndrome[check_node].
        let expected = vec![
            1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
        ];

        assert_eq!(expected, decoder.state.msg_cn_to_vn);
    }

    #[test]
    fn test_total_llrs() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();
        let nnz = H.nnz();

        let settings = Settings {
            max_iter: 100,
            bp_method: BpMethod::Spa,
        };

        let channel_llrs = vec![1.0; H.cols()];
        let mut decoder = BpDecoder::new(settings, &H, &channel_llrs[..]);
        decoder.state.msg_cn_to_vn = vec![1.0; nnz];

        decoder.total_llrs();

        // total_llrs[i] = channel_llrs[i] + degree(col i)
        // degrees: cols 0-2 have degree 1, cols 3-5 degree 2, col 6 degree 3
        let expected = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        assert_eq!(expected, decoder.state.total_llrs);
    }
}
