use crate::decoders::{
    Decoder,
    core::{Edge, SyndromeBpDecoder, SyndromeBpStrategy},
};

pub fn compute_syndrome(
    edges: &[Edge],
    num_cns: usize,
    e_hat: &[u8],
) -> Vec<u8> {
    let mut syndrome = vec![0u8; num_cns];

    for edge in edges {
        syndrome[edge.row] ^= e_hat[edge.col];
    }

    syndrome
}

#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Settings {
    pub max_iter: usize,
    /// Number of BP iterations before decimating
    pub T: usize,
}

#[derive(Clone)]
pub struct SyndromeBpGdDecoder<Core: SyndromeBpStrategy> {
    pub settings: Settings,
    pub core: Core,
    pub original_channel_llrs: Vec<f64>,
}

// TODO: Doc comments
// TODO: This implementation of BPGD is only compatible with the naive SPA core
impl<Core: SyndromeBpStrategy> Decoder for SyndromeBpGdDecoder<Core> {
    type Settings = Settings;

    // TODO: Should a reset happen here?
    fn decode(&mut self, s: &[u8]) -> Vec<u8> {
        let mut e_hat: Vec<u8> = self
            .core
            .get_state()
            .channel_llrs
            .iter()
            .map(|&v| if v < 0.0 { 1 } else { 0 })
            .collect();

        let mut iter_idx = 0;
        while (iter_idx < self.settings.max_iter)
            && (iter_idx < self.core.get_state().num_vns)
        {
            for _ in 0..self.settings.T {
                self.core.vn_update();
                self.core.cn_update(s);
            }

            self.core.total_llrs();

            e_hat = self
                .core
                .get_state()
                .total_llrs
                .iter()
                .map(|&v| if v < 0.0 { 1 } else { 0 })
                .collect();

            let state = self.core.get_state();
            let s_hat = compute_syndrome(&state.edges, state.num_cns, &e_hat);
            if s_hat == s {
                break;
            }

            let max_llr_index = self
                .core
                .get_state()
                .total_llrs
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|(_, a), (_, b)| {
                    a.abs().partial_cmp(&b.abs()).expect("Comparing NaN value")
                })
                .map(|(index, _)| index)
                .expect("Empty iterator: no finite LLRs found");

            let sign = self.core.get_state().total_llrs[max_llr_index].signum();

            self.core.get_state().channel_llrs[max_llr_index] =
                sign * f64::INFINITY;

            iter_idx += self.settings.T;
        }

        e_hat
    }
}

impl<Core: SyndromeBpStrategy> SyndromeBpDecoder for SyndromeBpGdDecoder<Core> {
    #[allow(non_snake_case)]
    fn new(
        settings: <Self as Decoder>::Settings,
        H: &sprs::CsMat<u8>,
        channel_llrs: &[f64],
    ) -> Self {
        Self {
            settings: settings,
            core: Core::new(H, channel_llrs),
            original_channel_llrs: channel_llrs.to_vec(),
        }
    }

    fn reset(&mut self) {
        for edge in &mut self.core.get_state().edges {
            edge.msg_vn_to_cn = 0.0;
            edge.msg_cn_to_vn = 0.0;
        }
        self.core.get_state().channel_llrs = self.original_channel_llrs.clone();
    }
}

#[cfg(test)]
mod tests {
    use sprs::{CsMat, TriMat};

    use super::*;
    use crate::decoders::core::SyndromeBpCore;
    use crate::decoders::core::spa::SyndromeSpaCore;

    #[allow(non_snake_case)]
    fn get_hamming_H() -> sprs::CsMat<u8> {
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

    fn csr_from_dense(rows: &[&[u8]]) -> CsMat<u8> {
        let nrows = rows.len();
        let ncols = rows[0].len();
        let mut tri = TriMat::new((nrows, ncols));
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0 {
                    tri.add_triplet(i, j, val);
                }
            }
        }
        tri.to_csr()
    }

    #[test]
    fn test_compute_syndrome() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();

        let mut edges = Vec::new();
        for (row, row_vec) in H.outer_iterator().enumerate() {
            for (col, _) in row_vec.iter() {
                edges.push(Edge {
                    row,
                    col,
                    msg_vn_to_cn: 0.0,
                    msg_cn_to_vn: 0.0,
                });
            }
        }

        let e = vec![0, 1, 0, 1, 0, 1, 0];
        let s = vec![1, 0, 0];

        assert_eq!(s, compute_syndrome(&edges, H.rows(), &e));
    }

    #[allow(non_snake_case)]
    fn remove_column(H: &sprs::CsMat<u8>, col_idx: usize) -> sprs::CsMat<u8> {
        let mut tri = TriMat::new((H.rows(), H.cols() - 1));
        for (val, (row, col)) in H.iter() {
            if col != col_idx {
                tri.add_triplet(
                    row,
                    if col > col_idx { col - 1 } else { col },
                    *val,
                );
            }
        }
        tri.to_csr()
    }

    fn abs_argmax(v: &[f64]) -> usize {
        v.iter()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .max_by(|(_, a), (_, b)| {
                a.abs().partial_cmp(&b.abs()).expect("Comparing NaN")
            })
            .map(|(i, _)| i)
            .expect("Empty iterator: no finite values")
    }

    fn copy_messages(dst: &mut SyndromeBpCore, src: &SyndromeBpCore) {
        assert_eq!(dst.edges.len(), src.edges.len());

        for (d, s) in dst.edges.iter_mut().zip(src.edges.iter()) {
            d.msg_cn_to_vn = s.msg_cn_to_vn;
            d.msg_vn_to_cn = s.msg_vn_to_cn;
        }
    }

    fn remove_variable(
        core: &SyndromeBpCore,
        dec_idx: usize,
    ) -> SyndromeBpCore {
        // Map old edge index to new edge index

        let mut old_to_new = vec![None::<usize>; core.edges.len()];
        let mut new_idx = 0;
        for (old_idx, edge) in core.edges.iter().enumerate() {
            if edge.col != dec_idx {
                old_to_new[old_idx] = Some(new_idx);
                new_idx += 1;
            }
        }

        // Build new edge list

        let new_edges: Vec<Edge> = core
            .edges
            .iter()
            .filter(|e| e.col != dec_idx)
            .map(|e| Edge {
                row: e.row,
                col: if e.col > dec_idx { e.col - 1 } else { e.col },
                msg_vn_to_cn: e.msg_vn_to_cn,
                msg_cn_to_vn: e.msg_cn_to_vn,
            })
            .collect();

        // Rebuild cn_ranges (each CN's surviving edges form a contiguous block)

        let mut new_cn_ranges = Vec::with_capacity(core.num_cns);
        let mut cur = 0usize;
        for j in 0..core.num_cns {
            let count = core.cn_ranges[j]
                .clone()
                .filter(|&edge_idx| core.edges[edge_idx].col != dec_idx)
                .count();
            new_cn_ranges.push(cur..cur + count);
            cur += count;
        }

        // Rebuild vn_indices (skip dec_idx entry, remap surviving edge indices)

        let new_vn_indices: Vec<Vec<usize>> = core
            .vn_indices
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != dec_idx)
            .map(|(_, v)| {
                v.iter()
                    .map(|&edge_idx| old_to_new[edge_idx].unwrap())
                    .collect()
            })
            .collect();

        let new_channel_llrs: Vec<f64> = core
            .channel_llrs
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != dec_idx)
            .map(|(_, &v)| v)
            .collect();

        let new_total_llrs: Vec<f64> = core
            .total_llrs
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != dec_idx)
            .map(|(_, &v)| v)
            .collect();

        SyndromeBpCore {
            edges: new_edges,
            cn_ranges: new_cn_ranges,
            vn_indices: new_vn_indices,
            channel_llrs: new_channel_llrs,
            total_llrs: new_total_llrs,
            num_vns: core.num_vns - 1,
            num_cns: core.num_cns,
        }
    }

    #[test]
    fn test_remove_variable() {
        #[allow(non_snake_case)]
        let H = csr_from_dense(&[
            &[0, 1, 1, 1, 1, 0, 0],
            &[1, 0, 1, 1, 0, 1, 0],
            &[1, 1, 0, 1, 0, 0, 1],
        ]);

        let channel_llrs: Vec<f64> = (0..7).map(|i| i as f64).collect();

        let cn_to_vn_original = csr_from_dense(&[
            &[0, 1, 2, 3, 4, 0, 0],
            &[7, 0, 9, 10, 0, 12, 0],
            &[14, 15, 0, 17, 0, 0, 20],
        ]);
        let vn_to_cn_original = csr_from_dense(&[
            &[0, 1, 2, 3, 4, 0, 0],
            &[7, 0, 9, 10, 0, 12, 0],
            &[14, 15, 0, 17, 0, 0, 20],
        ]);

        let channel_probs_expected = vec![0.0, 1.0, 2.0, 3.0, 5.0, 6.0];

        let cn_to_vn_expected = csr_from_dense(&[
            &[0, 1, 2, 3, 0, 0],
            &[7, 0, 9, 10, 12, 0],
            &[14, 15, 0, 17, 0, 20],
        ]);
        let vn_to_cn_expected = csr_from_dense(&[
            &[0, 1, 2, 3, 0, 0],
            &[7, 0, 9, 10, 12, 0],
            &[14, 15, 0, 17, 0, 20],
        ]);

        let mut spa = SyndromeSpaCore::new(&H, &channel_llrs);

        for (edge_idx, (msg_cn_to_vn, msg_vn_to_cn)) in cn_to_vn_original
            .data()
            .iter()
            .zip(vn_to_cn_original.data())
            .enumerate()
        {
            spa.get_state().edges[edge_idx].msg_cn_to_vn = *msg_cn_to_vn as f64;
            spa.get_state().edges[edge_idx].msg_vn_to_cn = *msg_vn_to_cn as f64;
        }

        let result = remove_variable(spa.get_state_ref(), 4);

        assert_eq!(result.channel_llrs, channel_probs_expected);

        assert_eq!(
            result
                .edges
                .iter()
                .map(|e| e.msg_cn_to_vn)
                .collect::<Vec::<f64>>(),
            cn_to_vn_expected
                .data()
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec::<f64>>(),
        );
        assert_eq!(
            result
                .edges
                .iter()
                .map(|e| e.msg_vn_to_cn)
                .collect::<Vec::<f64>>(),
            vn_to_cn_expected
                .data()
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec::<f64>>(),
        );
    }

    fn flip_syndrome_bits_for_decimated_vn(
        s: &mut Vec<u8>,
        core: &SyndromeBpCore,
        dec_idx: usize,
        total_llr: f64,
    ) {
        if total_llr < 0.0 {
            for edge in &core.edges {
                if edge.col == dec_idx {
                    s[edge.row] ^= 1;
                }
            }
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bpgd_decimation() {
        let H = csr_from_dense(&[
            &[0, 1, 1, 1, 0, 0],
            &[1, 0, 1, 0, 1, 0],
            &[1, 1, 0, 0, 0, 1],
        ]);

        let p = 0.003_f64;
        let channel_llr = ((1.0 - p) / p).ln();
        let channel_llrs = vec![channel_llr; 6];
        let s = vec![1u8, 1, 1];

        let mut s_manual = s.clone();

        //
        // Manual decimation using BP
        //

        // First round of BP

        let mut spa1 = SyndromeSpaCore::new(&H, &channel_llrs);

        spa1.vn_update();
        spa1.cn_update(&s_manual);
        spa1.total_llrs();

        let dec_idx = abs_argmax(&spa1.get_state_ref().total_llrs);
        flip_syndrome_bits_for_decimated_vn(
            &mut s_manual,
            spa1.get_state_ref(),
            dec_idx,
            spa1.get_state_ref().total_llrs[dec_idx],
        );

        // Round 1 of decimation

        let state2 = remove_variable(spa1.get_state_ref(), dec_idx);
        let h2 = remove_column(&H, dec_idx);
        let mut spa2 = SyndromeSpaCore::new(&h2, &state2.channel_llrs);
        copy_messages(spa2.get_state(), &state2);

        spa2.vn_update();
        spa2.cn_update(&s_manual);
        spa2.total_llrs();

        let dec_idx_2 = abs_argmax(&spa2.get_state_ref().total_llrs);
        flip_syndrome_bits_for_decimated_vn(
            &mut s_manual,
            spa2.get_state_ref(),
            dec_idx_2,
            spa2.get_state_ref().total_llrs[dec_idx_2],
        );

        // Round 2 of decimation

        let state3 = remove_variable(spa2.get_state_ref(), dec_idx_2);
        let h3 = remove_column(&h2, dec_idx_2);
        let mut spa3 = SyndromeSpaCore::new(&h3, &state3.channel_llrs);
        copy_messages(spa3.get_state(), &state3);

        spa3.vn_update();
        spa3.cn_update(&s_manual);
        spa3.total_llrs();

        //
        // Decimation built into `SimpleSyndromeBpGdDecoder`
        //

        let mut bpgd = SyndromeBpGdDecoder::<SyndromeSpaCore>::new(
            Settings { max_iter: 3, T: 1 },
            &H,
            &channel_llrs,
        );
        bpgd.decode(&s);

        //
        // Compare the results of the two approaches
        //

        let bpgd_non_decmiated =
            remove_variable(bpgd.core.get_state_ref(), dec_idx);
        let bpgd_non_decmiated =
            remove_variable(&bpgd_non_decmiated, dec_idx_2);

        assert!(
            spa3.get_state_ref()
                .channel_llrs
                .iter()
                .all(|&v| (v - channel_llr).abs() < 1e-15),
        );

        let manual = spa3.get_state_ref();
        assert_eq!(manual.edges.len(), bpgd_non_decmiated.edges.len());

        let tol = 1e-8_f64;
        for (manual_edge, bpgd_edge) in
            manual.edges.iter().zip(&bpgd_non_decmiated.edges)
        {
            assert!(
                (manual_edge.msg_cn_to_vn - bpgd_edge.msg_cn_to_vn).abs() < tol,
                "msg_cn_to_vn mismatch: {} vs {} (row={} col={})",
                manual_edge.msg_cn_to_vn,
                bpgd_edge.msg_cn_to_vn,
                manual_edge.row,
                manual_edge.col,
            );
            assert!(
                (manual_edge.msg_vn_to_cn - bpgd_edge.msg_vn_to_cn).abs() < tol,
                "msg_vn_to_cn mismatch: {} vs {} (row={} col={})",
                manual_edge.msg_vn_to_cn,
                bpgd_edge.msg_vn_to_cn,
                manual_edge.row,
                manual_edge.col,
            );
        }
    }
}
