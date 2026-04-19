use num_traits::Float;
use sprs::{CsMat, DenseVector};

use crate::decoders::{
    Decoder,
    bp::{VanillaBpDecoder, VanillaBpSettings},
    engine::{
        AccessEngineInternals, EdgeId, ParityCheckMatrix,
        min_sum::MinSumComputeEngine, spa::SpaComputeEngine,
    },
    meta::split_windows::{
        OverlapInfo, get_overlap_info, get_window_borders, split_channel_llrs,
        split_pcm,
    },
};

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct SlidingWindowSettings {
    pub warm_start: bool,
    pub F: usize,
    pub W: usize,
}

pub trait InnerWindowDecoder: Decoder {
    type Llr: Copy;

    fn new(
        settings: Self::Settings,
        pcm: &ParityCheckMatrix,
        channel_llrs: &[Self::Llr],
    ) -> Self;

    fn get_cn_to_vn_msg(&self, edge_id: EdgeId) -> Self::Llr;
    fn get_vn_to_cn_msg(&self, edge_id: EdgeId) -> Self::Llr;
    fn get_channel_llr(&self, i: usize) -> Self::Llr;

    fn set_cn_to_vn_msg(&mut self, edge_id: EdgeId, msg: Self::Llr);
    fn set_vn_to_cn_msg(&mut self, edge_id: EdgeId, msg: Self::Llr);
    fn set_channel_llr(&mut self, i: usize, llr: Self::Llr);

    fn reset(&mut self);
}

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct SlidingWindowDecoder<InnerDecoder>
where
    InnerDecoder: InnerWindowDecoder,
{
    settings: SlidingWindowSettings,
    window_decoders: Vec<InnerDecoder>,
    window_borders: Vec<((usize, usize), (usize, usize))>,
    overlap_info: OverlapInfo,
    win_pcms: Vec<ParityCheckMatrix>,
    e_hat_total: Vec<u8>,
}

impl<InnerDecoder> SlidingWindowDecoder<InnerDecoder>
where
    InnerDecoder: InnerWindowDecoder,
{
    // Comitting certain columns of the error estimate may change the parity
    // of some syndrome bits for the next window. This function computes
    // that change in parity.
    //
    // The figure below depicts two overlapping windows, the borders of which
    // are represented by a dashed and a dotted line respectively.
    //
    // comitted
    // rows
    //    |
    // |-----|
    //
    // 1 1 0 0   0 0 | 0 0
    // 1 1 0 0   0 0 | 0 0
    // 0 1 1 1   0 0 | 0 0
    // 0 1 1 1   0 0 | 0 0
    //         ...........  -
    // 0 0 0 1 . 1 1 | 0 0  | rows whose
    // 0 0 0 1 . 1 1 | 0 0  | parity changes
    // --------. ---------  -
    // 0 0 0 0 . 0 1 | 1 1
    // 0 0 0 0 . 0 1 | 1 1
    #[allow(non_snake_case)]
    fn get_next_window_syndrome_diff(
        &self,
        e_hat: &[u8],
        win_idx: usize,
    ) -> Vec<u8> {
        if win_idx + 1 >= self.window_decoders.len() {
            return vec![];
        }

        let next_syndrome_len = self.win_pcms[win_idx + 1].rows();
        let overlap_start = self.overlap_info.begin_positions[win_idx];

        let mut e_hat_comitted = e_hat.to_vec();
        for e_i in &mut e_hat_comitted[overlap_start.1..] {
            *e_i = 0u8;
        }

        let s = self.win_pcms[win_idx].compute_syndrome(&e_hat_comitted);

        let mut result = vec![0u8; next_syndrome_len];
        for (i, s_i) in s[overlap_start.0..].iter().enumerate() {
            result[i] = *s_i;
        }

        result
    }

    fn get_commited_e_hat<'a>(
        &self,
        e_hat: &'a [u8],
        win_idx: usize,
    ) -> &'a [u8] {
        let end_idx = if win_idx < self.window_decoders.len() - 1 {
            self.overlap_info.begin_positions[win_idx].1
        } else {
            e_hat.len()
        };

        &e_hat[0..end_idx]
    }

    fn cut_out_current_window_syndrome(
        &self,
        s: &[u8],
        win_idx: usize,
    ) -> Vec<u8> {
        let (row_begin, _) = self.window_borders[win_idx].0;
        let (row_end, _) = self.window_borders[win_idx].1;

        s[row_begin..=row_end].to_vec()
    }

    fn transfer_soft_info_from_previous_window(&mut self, win_idx: usize) {
        assert!(win_idx >= 1);

        let prev_win_overlap_start =
            self.overlap_info.begin_positions[win_idx - 1];
        let curr_win_overlap_end = self.overlap_info.end_positions[win_idx - 1];

        let pcm1 = &self.win_pcms[win_idx - 1];
        let pcm2 = &self.win_pcms[win_idx];

        let prev_win_rows = prev_win_overlap_start.0..;
        let prev_win_cols = prev_win_overlap_start.1..;
        let src_edges = pcm1.slice(&prev_win_rows, &prev_win_cols);

        let curr_win_rows = ..=curr_win_overlap_end.0;
        let curr_win_cols = ..=curr_win_overlap_end.1;
        let dest_edges = pcm2.slice(&curr_win_rows, &curr_win_cols);

        for (e_src, e_dst) in src_edges.zip(dest_edges) {
            let msg = self.window_decoders[win_idx - 1].get_cn_to_vn_msg(e_src);
            self.window_decoders[win_idx].set_cn_to_vn_msg(e_dst, msg);
        }

        for i in 0..=curr_win_overlap_end.1 {
            let llr = self.window_decoders[win_idx - 1]
                .get_channel_llr(prev_win_overlap_start.1 + i);
            self.window_decoders[win_idx].set_channel_llr(i, llr);
        }
    }
}

#[allow(non_snake_case)]
impl<InnerDecoder> SlidingWindowDecoder<InnerDecoder>
where
    InnerDecoder: InnerWindowDecoder,
    InnerDecoder::Llr: Float,
{
    pub fn new(
        settings: SlidingWindowSettings,
        inner_settings: <InnerDecoder as Decoder>::Settings,
        H: &CsMat<u8>,
        m: usize,
        num_rounds: usize,
        channel_llrs: &[InnerDecoder::Llr],
    ) -> Self {
        let window_borders =
            get_window_borders(&H, m, num_rounds, settings.W, settings.F);

        let win_hs = split_pcm(&H, &window_borders);
        let win_llrs = split_channel_llrs(&channel_llrs, &window_borders);
        let overlap_info = get_overlap_info(&window_borders);

        let mut win_pcms =
            Vec::<ParityCheckMatrix>::with_capacity(win_hs.len());
        for win_h in win_hs {
            let pcm = ParityCheckMatrix::new(&win_h);
            win_pcms.push(pcm);
        }

        let window_decoders = win_pcms
            .iter()
            .zip(win_llrs)
            .map(|(pcm, channel_llrs)| {
                InnerDecoder::new(inner_settings.clone(), &pcm, &channel_llrs)
            })
            .collect();

        win_pcms.push(ParityCheckMatrix::new(H));

        Self {
            e_hat_total: Vec::<u8>::with_capacity(H.cols()),
            settings,
            window_decoders,
            window_borders,
            overlap_info,
            win_pcms: win_pcms,
        }
    }

    pub fn reset(&mut self) {
        self.e_hat_total.clear();
        for decoder in self.window_decoders.iter_mut() {
            decoder.reset();
        }
    }
}

fn vec_add_inplace(a: &mut [u8], b: &[u8]) {
    assert_eq!(a.len(), b.len());

    for (x, &y) in a.iter_mut().zip(b.iter()) {
        *x ^= y;
    }
}

impl<Engine> Decoder for SlidingWindowDecoder<Engine>
where
    Engine: InnerWindowDecoder,
{
    type Settings = SlidingWindowSettings;

    fn decode(&mut self, s: &[u8]) -> &[u8] {
        self.e_hat_total.clear();

        let mut s_diff = Vec::<u8>::zeros(self.window_borders[0].1.0 + 1);

        for win_idx in 0..self.window_decoders.len() {
            let mut s_win = self.cut_out_current_window_syndrome(&s, win_idx);
            vec_add_inplace(&mut s_win, &s_diff);

            if self.settings.warm_start && win_idx >= 1 {
                self.transfer_soft_info_from_previous_window(win_idx);
            }

            let e_hat: Vec<u8> =
                self.window_decoders[win_idx].decode(&s_win).to_vec();

            self.e_hat_total
                .extend(self.get_commited_e_hat(&e_hat, win_idx));
            s_diff = self.get_next_window_syndrome_diff(&e_hat, win_idx);
        }

        &self.e_hat_total
    }
}

#[cfg(test)]
mod tests {
    use crate::decoders::{
        bp::{VanillaBpDecoder, VanillaBpSettings},
        engine::min_sum::MinSumComputeEngine,
    };

    use super::*;

    use sprs::TriMat;

    fn get_hamming_h() -> sprs::CsMat<u8> {
        #[allow(non_snake_case)]
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
    fn test_soft_init_get() {
        #[allow(non_snake_case)]
        let h = get_hamming_h();
        let channel_llrs: Vec<f64> = (0..h.cols()).map(|_| 0.0).collect();

        let pcm = ParityCheckMatrix::new(&h);

        let mut decoder = VanillaBpDecoder::<MinSumComputeEngine>::new(
            VanillaBpSettings { max_iter: 32 },
            &pcm,
            &channel_llrs,
        );

        let edge_ids: Vec<EdgeId> = pcm.slice(&(..), &(..)).collect();
        for (idx, eid) in edge_ids.into_iter().enumerate() {
            decoder.set_cn_to_vn_msg(eid, (idx + 1) as f64);
        }

        //     1 0 0 1 1 0 1
        // H = 0 1 0 0 1 1 1
        //     0 0 1 1 0 1 1
        //
        //            1 0 0 2  3 0  4
        // L_{i<-j} = 0 5 0 0  6 7  8
        //            0 0 9 10 0 11 12

        let expected1 = vec![6.0, 7.0, 8.0, 10.0, 11.0, 12.0];

        let edge_ids: Vec<EdgeId> = pcm.slice(&(1..), &(3..)).collect();
        let got1: Vec<f64> = edge_ids
            .iter()
            .map(|&e| decoder.get_cn_to_vn_msg(e))
            .collect();

        let expected2 = vec![3.0, 4.0, 6.0, 7.0, 8.0, 11.0, 12.0];
        let edge_ids: Vec<EdgeId> = pcm.slice(&(0..), &(4..)).collect();
        let got2: Vec<f64> = edge_ids
            .iter()
            .map(|&e| decoder.get_cn_to_vn_msg(e))
            .collect();

        let expected3 = vec![9.0, 10.0, 11.0, 12.0];
        let edge_ids: Vec<EdgeId> = pcm.slice(&(2..), &(0..)).collect();
        let got3: Vec<f64> = edge_ids
            .iter()
            .map(|&e| decoder.get_cn_to_vn_msg(e))
            .collect();

        assert_eq!(expected1, got1);
        assert_eq!(expected2, got2);
        assert_eq!(expected3, got3);
    }

    #[test]
    fn test_window_result_manipulation() {
        #[allow(non_snake_case)]
        let h = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1],
            &[0, 0, 0, 0, 0, 1, 1, 1],
        ]);

        let channel_llrs = Vec::<f64>::zeros(h.cols());

        let decoder =
            SlidingWindowDecoder::<VanillaBpDecoder<MinSumComputeEngine>>::new(
                SlidingWindowSettings {
                    warm_start: false,
                    F: 2,
                    W: 3,
                },
                VanillaBpSettings { max_iter: 32 },
                &h,
                2,
                4 - 2,
                &channel_llrs,
            );

        // 1 1 0 0   0 0 | 0 0
        // 1 1 0 0   0 0 | 0 0
        // 0 1 1 1   0 0 | 0 0
        // 0 1 1 1   0 0 | 0 0
        //         ...........
        // 0 0 0 1 . 1 1 | 0 0
        // 0 0 0 1 . 1 1 | 0 0
        // --------. ---------
        // 0 0 0 0 . 0 1 | 1 1
        // 0 0 0 0 . 0 1 | 1 1

        let s: Vec<u8> = (0..h.rows()).map(|v| v as u8).collect();

        // Window 1

        let e_hat: Vec<u8> = (0..6).map(|v| v as u8).collect();
        let got = decoder.get_commited_e_hat(&e_hat, 0);
        let expected = vec![0, 1, 2, 3];
        assert_eq!(expected, got);

        let got = decoder.cut_out_current_window_syndrome(&s, 0);
        let expected = vec![0, 1, 2, 3, 4, 5];
        assert_eq!(expected, got);

        let e_hat = vec![1, 0, 0, 1, 0, 0];
        let got = decoder.get_next_window_syndrome_diff(&e_hat, 0);
        let expected = vec![1, 1, 0, 0];
        assert_eq!(expected, got);

        // Window 1

        let e_hat: Vec<u8> = (0..4).map(|v| v as u8).collect();
        let got = decoder.get_commited_e_hat(&e_hat, 1);
        let expected = vec![0, 1, 2, 3];
        assert_eq!(expected, got);

        let got = decoder.cut_out_current_window_syndrome(&s, 1);
        let expected = vec![4, 5, 6, 7];
        assert_eq!(expected, got);

        let e_hat = vec![0, 0, 0, 0];
        let got = decoder.get_next_window_syndrome_diff(&e_hat, 1);
        let expected = Vec::<u8>::new();
        assert_eq!(expected, got);
    }

    #[test]
    fn test_soft_info_passing() {
        #[allow(non_snake_case)]
        let h = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1],
            &[0, 0, 0, 0, 0, 1, 1, 1],
        ]);

        let channel_llrs =
            (0..h.cols()).map(|v| v as f64).collect::<Vec<f64>>();

        let mut decoder =
            SlidingWindowDecoder::<VanillaBpDecoder<MinSumComputeEngine>>::new(
                SlidingWindowSettings {
                    warm_start: false,
                    F: 2,
                    W: 3,
                },
                VanillaBpSettings { max_iter: 32 },
                &h,
                2,
                4 - 2,
                &channel_llrs,
            );

        // 1 2 0 0    0  0  | 0 0
        // 3 4 0 0    0  0  | 0 0
        // 0 5 6 7    0  0  | 0 0
        // 0 8 9 10   0  0  | 0 0
        //          ...........
        // 0 0 0 11 . 12 13 | 0 0
        // 0 0 0 14 . 15 16 | 0 0
        // -------- . ---------
        // 0 0 0 0  . 0  1  | 1 1
        // 0 0 0 0  . 0  1  | 1 1

        let edge_ids: Vec<EdgeId> =
            decoder.win_pcms[0].slice(&(..), &(..)).collect();
        for (idx, eid) in edge_ids.into_iter().enumerate() {
            decoder.window_decoders[0].set_cn_to_vn_msg(eid, (idx + 1) as f64);
        }

        for i in 0..6 {
            decoder.window_decoders[0].set_channel_llr(i, i as f64);
        }

        let edge_ids: Vec<EdgeId> =
            decoder.win_pcms[1].slice(&(..), &(..)).collect();
        for (idx, eid) in edge_ids.into_iter().enumerate() {
            decoder.window_decoders[1].set_cn_to_vn_msg(eid, (idx + 1) as f64);
        }

        for i in 0..4 {
            decoder.window_decoders[1].set_channel_llr(i, i as f64);
        }

        decoder.transfer_soft_info_from_previous_window(1);

        let expected =
            vec![12.0, 13.0, 15.0, 16.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let got = decoder.win_pcms[1]
            .slice(&(..), &(..))
            .map(|e| decoder.window_decoders[1].get_cn_to_vn_msg(e))
            .collect::<Vec<f64>>();
        assert_eq!(expected, got);

        let channel_llrs: Vec<f64> = (0..decoder.win_pcms[1].cols())
            .map(|i| decoder.window_decoders[1].get_channel_llr(i))
            .collect();

        assert_eq!(channel_llrs, vec![4.0, 5.0, 2.0, 3.0]);
    }

    #[test]
    fn test_soft_init_set() {
        #[allow(non_snake_case)]
        let h = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1],
            &[0, 0, 0, 0, 0, 1, 1, 1],
        ]);

        let channel_llrs: Vec<f64> = (0..h.cols()).map(|_| 0.0).collect();

        let pcm = ParityCheckMatrix::new(&h);

        let mut decoder1 = VanillaBpDecoder::<MinSumComputeEngine>::new(
            VanillaBpSettings { max_iter: 32 },
            &pcm,
            &channel_llrs,
        );

        let edge_ids: Vec<EdgeId> = pcm.slice(&(..), &(..)).collect();
        for (idx, eid) in edge_ids.into_iter().enumerate() {
            decoder1.set_cn_to_vn_msg(eid, (idx + 1) as f64);
        }

        let mut decoder2 = VanillaBpDecoder::<MinSumComputeEngine>::new(
            VanillaBpSettings { max_iter: 32 },
            &pcm,
            &channel_llrs,
        );

        let edge_ids: Vec<EdgeId> = pcm.slice(&(..), &(..)).collect();
        for (idx, eid) in edge_ids.into_iter().enumerate() {
            decoder2.set_cn_to_vn_msg(eid, (idx + 1) as f64);
        }

        // Decoder 1:
        //            1  2  0  0  0  0  0  0
        //            3  4  0  0  0  0  0  0
        //            0  5  6  7  0  0  0  0
        // L_{i<-j} = 0  8  9  10 0  0  0  0
        //            0  0  0  11 12 13 0  0
        //            0  0  0  14 15 16 0  0
        //            0  0  0  0  0  17 18 19
        //            0  0  0  0  0  20 21 22

        // Decoder 2: (after transfer)
        //            12 13 0  0  0  0  0  0
        //            15 16 0  0  0  0  0  0
        //            0  17 18 19 0  0  0  0
        // L_{i<-j} = 0  20 21 22 0  0  0  0
        //            0  0  0  11 12 13 0  0
        //            0  0  0  14 15 16 0  0
        //            0  0  0  0  0  17 18 19
        //            0  0  0  0  0  20 21 22

        let decoder1_edges: Vec<EdgeId> = pcm.slice(&(4..), &(4..)).collect();
        let decoder2_edges: Vec<EdgeId> = pcm.slice(&(..4), &(..4)).collect();
        for (e1, e2) in
            decoder1_edges.into_iter().zip(decoder2_edges.into_iter())
        {
            let msg = decoder1.get_cn_to_vn_msg(e1);
            decoder2.set_cn_to_vn_msg(e2, msg);
        }

        let expected = vec![
            12.0, 13.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 11.0,
            12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
        ];

        let edge_ids: Vec<EdgeId> = pcm.slice(&(..), &(..)).collect();
        let got = edge_ids
            .iter()
            .map(|&e| decoder2.get_cn_to_vn_msg(e))
            .collect::<Vec<f64>>();

        assert_eq!(expected, got);

        // Decoder 2:
        //            1  2  0  0  0  0  0  0
        //            3  4  0  0  0  0  0  0
        //            0  5  6  7  0  0  0  0
        // L_{i<-j} = 0  8  9  10 0  0  0  0
        //            0  0  0  11 12 13 0  0
        //            0  0  0  14 15 16 0  0
        //            0  0  0  0  0  17 18 19
        //            0  0  0  0  0  20 21 22

        let decoder1_edges: Vec<EdgeId> = pcm.slice(&(0..), &(0..)).collect();
        let decoder2_edges: Vec<EdgeId> = pcm.slice(&(..8), &(..8)).collect();
        for (e1, e2) in
            decoder1_edges.into_iter().zip(decoder2_edges.into_iter())
        {
            let msg = decoder1.get_cn_to_vn_msg(e1);
            decoder2.set_cn_to_vn_msg(e2, msg);
        }

        let expected = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
        ];
        let edge_ids: Vec<EdgeId> = pcm.slice(&(..), &(..)).collect();
        let got: Vec<f64> = edge_ids
            .iter()
            .map(|&e| decoder2.get_cn_to_vn_msg(e))
            .collect();

        assert_eq!(expected, got);
    }
}
