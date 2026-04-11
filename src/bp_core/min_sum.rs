use crate::bp_core::{Edge, SyndromeBpCore, SyndromeBpStrategy};
use sprs::CsMat;

pub struct SyndromeMinSumCore(pub SyndromeBpCore);

impl SyndromeBpStrategy for SyndromeMinSumCore {
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

        Self(SyndromeBpCore {
            edges,
            cn_ranges,
            vn_indices,
            channel_llrs: channel_llrs.to_vec(),
            total_llrs: vec![0.0; num_vns],
            num_vns,
            num_cns,
        })
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
        for i in 0..self.0.num_vns {
            let mut total = self.0.channel_llrs[i];
            let num_edges = self.0.vn_indices[i].len();
            for ki in 0..num_edges {
                let k = self.0.vn_indices[i][ki];
                total += self.0.edges[k].msg_cn_to_vn;
            }
            for ki in 0..num_edges {
                let k = self.0.vn_indices[i][ki];
                self.0.edges[k].msg_vn_to_cn =
                    total - self.0.edges[k].msg_cn_to_vn;
            }
        }
    }

    fn cn_update(&mut self, syndrome: &[u8]) {
        for j in 0..self.0.num_cns {
            let range = self.0.cn_ranges[j].clone();
            let syndrome_sign = 1.0 - 2.0 * syndrome[j] as f64;
            let mut total_sign = syndrome_sign;
            let mut min1 = f64::INFINITY;
            let mut min2 = f64::INFINITY;
            let mut min1_idx = range.start;

            // First pass: accumulate sign, find two smallest magnitudes
            for k in range.clone() {
                let msg = self.0.edges[k].msg_vn_to_cn;
                total_sign *= msg.signum();
                let abs_msg = msg.abs();
                if abs_msg < min1 {
                    min2 = min1;
                    min1 = abs_msg;
                    min1_idx = k;
                } else if abs_msg < min2 {
                    min2 = abs_msg;
                }
            }

            // Second pass: assign outgoing messages
            for k in range {
                let msg = self.0.edges[k].msg_vn_to_cn;
                let sign_excl = total_sign * msg.signum(); // divide out this edge's sign
                let mag_excl = if k == min1_idx { min2 } else { min1 };
                self.0.edges[k].msg_cn_to_vn = sign_excl * mag_excl;
            }
        }
    }

    fn total_llrs(&mut self) {
        for i in 0..self.0.num_vns {
            self.0.total_llrs[i] = self.0.channel_llrs[i];
            let num_edges = self.0.vn_indices[i].len();
            for ki in 0..num_edges {
                let k = self.0.vn_indices[i][ki];
                self.0.total_llrs[i] += self.0.edges[k].msg_cn_to_vn;
            }
        }
    }

    fn get_state(&mut self) -> &mut SyndromeBpCore {
        &mut self.0
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
        let mut decoder = SyndromeMinSumCore::new(&H, &channel_llrs);
        for edge in &mut decoder.0.edges {
            edge.msg_cn_to_vn = 1.0;
        }

        decoder.vn_update();

        // Edges in CSR order: (0,0),(0,3),(0,4),(0,6),(1,1),(1,4),(1,5),(1,6),(2,2),(2,3),(2,5),(2,6)
        // col degrees: 0→1, 1→1, 2→1, 3→2, 4→2, 5→2, 6→3
        let expected =
            vec![0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0];
        let got: Vec<f64> =
            decoder.0.edges.iter().map(|e| e.msg_vn_to_cn).collect();
        assert_eq!(expected, got);
    }

    #[test]
    fn test_cn_update() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();

        let channel_llrs = vec![0.0; H.cols()];
        let mut decoder = SyndromeMinSumCore::new(&H, &channel_llrs);
        for edge in &mut decoder.0.edges {
            edge.msg_vn_to_cn = 1.0;
        }

        let s = vec![0, 1, 0];
        decoder.cn_update(&s);

        // Edges in CSR order: rows 0,0,0,0 then 1,1,1,1 then 2,2,2,2
        // syndrome_sign: row 0 = +1, row 1 = -1, row 2 = +1
        // Magnitude = min(|vn_msgs|) = 1.0; sign follows syndrome
        let expected = vec![
            1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let got: Vec<f64> =
            decoder.0.edges.iter().map(|e| e.msg_cn_to_vn).collect();
        assert_eq!(expected, got);
    }

    #[test]
    fn test_total_llrs() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();

        let channel_llrs = vec![1.0; H.cols()];
        let mut decoder = SyndromeMinSumCore::new(&H, &channel_llrs);
        for edge in &mut decoder.0.edges {
            edge.msg_cn_to_vn = 1.0;
        }

        decoder.total_llrs();

        // total_llrs[i] = channel_llrs[i] + degree(col i)
        // degrees: cols 0-2 have degree 1, cols 3-5 degree 2, col 6 degree 3
        let expected = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        assert_eq!(expected, decoder.0.total_llrs);
    }
}
