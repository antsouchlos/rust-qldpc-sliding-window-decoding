use sprs::{CsMat, DenseVector};

use crate::decoders::Decoder;
use crate::decoders::bp::SimpleSyndromeBpDecoder;
use crate::decoders::bpgd::SyndromeBpGdDecoder;
use crate::decoders::core::{Edge, SyndromeBpDecoder, SyndromeBpStrategy};
use crate::windowing::{OverlapInfo, split_pcm, split_priors};
use crate::windowing::{get_overlap_info, get_window_borders};

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct Settings {
    pub pass_soft_info: bool,
    pub F: usize,
    pub W: usize,
}

// TODO: Write this trait in a way where it doesn't depend on the internal
// memory representation
pub trait SoftInitBpDecoder: SyndromeBpDecoder {
    fn get_cn_to_vn_tail(
        &self,
        start_row: usize,
        start_col: usize,
    ) -> Vec<Edge>;

    fn set_cn_to_vn_head(
        &mut self,
        msgs: &[Edge],
        end_row: usize,
        end_col: usize,
    );

    fn get_channel_llr_tail(&self, start_col: usize) -> Vec<f64>;

    fn set_channel_llr_head(&mut self, channel_llrs: &[f64], num_cols: usize);
}

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct WindowingSyndromeBpDecoder<InnerDecoder>
where
    InnerDecoder: SoftInitBpDecoder,
{
    pub settings: Settings,
    pub window_decoders: Vec<InnerDecoder>,
    pub window_borders: Vec<((usize, usize), (usize, usize))>,
    pub overlap_info: OverlapInfo,
    pub win_Hs: Vec<CsMat<u8>>,
}

#[allow(non_snake_case)]
impl<InnerDecoder> WindowingSyndromeBpDecoder<InnerDecoder>
where
    InnerDecoder: SoftInitBpDecoder,
{
    pub fn new(
        settings: Settings,
        inner_settings: <InnerDecoder as Decoder>::Settings,
        H: &CsMat<u8>,
        m: usize,
        num_rounds: usize,
        channel_llrs: &[f64],
    ) -> Self {
        let window_borders =
            get_window_borders(&H, m, num_rounds, settings.W, settings.F);

        let win_Hs = split_pcm(&H, &window_borders);
        let win_priors = split_priors(&channel_llrs, &window_borders);
        let overlap_info = get_overlap_info(&window_borders);

        let window_decoders = win_Hs
            .iter()
            .zip(win_priors)
            .map(|(H, channel_llrs)| {
                InnerDecoder::new(inner_settings.clone(), &H, &channel_llrs)
            })
            .collect();

        Self {
            settings,
            window_decoders,
            window_borders,
            overlap_info,
            win_Hs,
        }
    }

    pub fn reset(&mut self) {
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

impl<InnerDdecoder> WindowingSyndromeBpDecoder<InnerDdecoder>
where
    InnerDdecoder: SoftInitBpDecoder,
{
    // TODO: Clean this up
    #[allow(non_snake_case)]
    fn get_next_window_syndrome_diff(
        &self,
        e_hat: &[u8],
        win_idx: usize,
    ) -> Vec<u8> {
        if win_idx + 1 >= self.window_decoders.len() {
            return vec![];
        }

        let n_committed_cols = self.overlap_info.begin_positions[win_idx].1;
        let overlap_row_start = self.overlap_info.begin_positions[win_idx].0;
        let n_overlap_rows = self.overlap_info.end_positions[win_idx].0 + 1;

        let next_win = &self.window_borders[win_idx + 1];
        let n_rows_next = next_win.1.0 - next_win.0.0 + 1;

        let mut s_diff = vec![0u8; n_rows_next];

        let win_H = &self.win_Hs[win_idx];
        for local_row in overlap_row_start..overlap_row_start + n_overlap_rows {
            let row_vec = win_H.outer_view(local_row).unwrap();
            let mut val = 0u8;
            for (col, &h) in row_vec.iter() {
                if col < n_committed_cols && h != 0 {
                    val ^= e_hat[col];
                }
            }
            s_diff[local_row - overlap_row_start] = val;
        }

        s_diff
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

    fn get_soft_info_from_previous_window(&mut self, win_idx: usize) {
        assert!(win_idx >= 1);

        let prev_win_overlap_start =
            self.overlap_info.begin_positions[win_idx - 1];
        let curr_win_overlap_end = self.overlap_info.end_positions[win_idx - 1];

        let prev_cn_to_vn_msgs = self.window_decoders[win_idx - 1]
            .get_cn_to_vn_tail(
                prev_win_overlap_start.0,
                prev_win_overlap_start.1,
            );
        let prev_channel_llr_msgs = self.window_decoders[win_idx - 1]
            .get_channel_llr_tail(prev_win_overlap_start.1);

        self.window_decoders[win_idx].set_cn_to_vn_head(
            &prev_cn_to_vn_msgs,
            curr_win_overlap_end.0 + 1,
            curr_win_overlap_end.1 + 1,
        );
        self.window_decoders[win_idx].set_channel_llr_head(
            &prev_channel_llr_msgs,
            curr_win_overlap_end.1 + 1,
        );
    }
}

impl<InnerDecoder> Decoder for WindowingSyndromeBpDecoder<InnerDecoder>
where
    InnerDecoder: SoftInitBpDecoder,
{
    type Settings = Settings;

    fn decode(&mut self, s: &[u8]) -> Vec<u8> {
        let mut e_hat_total = Vec::<u8>::new();
        let mut s_diff = Vec::<u8>::zeros(self.window_borders[0].1.0 + 1);

        for win_idx in 0..self.window_decoders.len() {
            let mut s_win = self.cut_out_current_window_syndrome(&s, win_idx);
            vec_add_inplace(&mut s_win, &s_diff);

            if self.settings.pass_soft_info && win_idx >= 1 {
                self.get_soft_info_from_previous_window(win_idx);
            }

            let e_hat = self.window_decoders[win_idx].decode(&s_win);

            e_hat_total.extend(self.get_commited_e_hat(&e_hat, win_idx).iter());
            s_diff = self.get_next_window_syndrome_diff(&e_hat, win_idx);
        }

        e_hat_total
    }
}

impl<Core: SyndromeBpStrategy> SoftInitBpDecoder
    for SimpleSyndromeBpDecoder<Core>
{
    fn get_cn_to_vn_tail(
        &self,
        start_row: usize,
        start_col: usize,
    ) -> Vec<Edge> {
        self.core
            .get_state_ref()
            .edges
            .iter()
            .filter(|e| e.row >= start_row && e.col >= start_col)
            .map(|e| Edge {
                row: e.row - start_row,
                col: e.col - start_col,
                msg_vn_to_cn: 0.0,
                msg_cn_to_vn: e.msg_cn_to_vn,
            })
            .collect()
    }

    fn set_cn_to_vn_head(
        &mut self,
        msgs: &[Edge],
        num_rows: usize,
        num_cols: usize,
    ) {
        assert!(msgs.iter().all(|e| e.row < num_rows && e.col < num_cols));

        // Two-pointer merge: both edge lists are in (row, col) sorted CSR order.
        let mut msg_iter = msgs.iter().peekable();
        for edge in self.core.get_state().edges.iter_mut() {
            if edge.row >= num_rows || edge.col >= num_cols {
                continue;
            }
            // Advance msg_iter past entries that sort before this edge.
            loop {
                match msg_iter.peek() {
                    Some(m) if (m.row, m.col) < (edge.row, edge.col) => {
                        msg_iter.next();
                    }
                    _ => break,
                }
            }
            if let Some(m) = msg_iter.peek() {
                if m.row == edge.row && m.col == edge.col {
                    edge.msg_cn_to_vn = m.msg_cn_to_vn;
                }
            }
        }
    }

    fn get_channel_llr_tail(&self, start_col: usize) -> Vec<f64> {
        self.core.get_state_ref().channel_llrs[start_col..].to_vec()
    }

    fn set_channel_llr_head(&mut self, channel_llrs: &[f64], num_cols: usize) {
        assert!(num_cols <= self.core.get_state_ref().channel_llrs.len());

        for i in 0..num_cols {
            self.core.get_state().channel_llrs[i] = channel_llrs[i];
        }
    }
}

impl<Core: SyndromeBpStrategy> SoftInitBpDecoder for SyndromeBpGdDecoder<Core> {
    fn get_cn_to_vn_tail(
        &self,
        start_row: usize,
        start_col: usize,
    ) -> Vec<Edge> {
        self.core
            .get_state_ref()
            .edges
            .iter()
            .filter(|e| e.row >= start_row && e.col >= start_col)
            .map(|e| Edge {
                row: e.row - start_row,
                col: e.col - start_col,
                msg_vn_to_cn: 0.0,
                msg_cn_to_vn: e.msg_cn_to_vn,
            })
            .collect()
    }

    fn set_cn_to_vn_head(
        &mut self,
        msgs: &[Edge],
        num_rows: usize,
        num_cols: usize,
    ) {
        assert!(msgs.iter().all(|e| e.row < num_rows && e.col < num_cols));

        // Two-pointer merge: both edge lists are in (row, col) sorted CSR order.
        let mut msg_iter = msgs.iter().peekable();
        for edge in self.core.get_state().edges.iter_mut() {
            if edge.row >= num_rows || edge.col >= num_cols {
                continue;
            }
            // Advance msg_iter past entries that sort before this edge.
            loop {
                match msg_iter.peek() {
                    Some(m) if (m.row, m.col) < (edge.row, edge.col) => {
                        msg_iter.next();
                    }
                    _ => break,
                }
            }
            if let Some(m) = msg_iter.peek() {
                if m.row == edge.row && m.col == edge.col {
                    edge.msg_cn_to_vn = m.msg_cn_to_vn;
                }
            }
        }
    }

    fn get_channel_llr_tail(&self, start_col: usize) -> Vec<f64> {
        self.core.get_state_ref().channel_llrs[start_col..].to_vec()
    }

    fn set_channel_llr_head(&mut self, channel_llrs: &[f64], num_cols: usize) {
        assert!(num_cols <= self.core.get_state_ref().channel_llrs.len());

        for i in 0..num_cols {
            self.core.get_state().channel_llrs[i] = channel_llrs[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::decoders::{bp, core::min_sum::SyndromeMinSumCore};
    use sprs::TriMat;

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
    fn test_soft_init_get() {
        #[allow(non_snake_case)]
        let H = get_hamming_H();
        let channel_llrs: Vec<f64> = (0..H.cols()).map(|_| 0.0).collect();

        let mut decoder = SimpleSyndromeBpDecoder::<SyndromeMinSumCore>::new(
            bp::Settings { max_iter: 32 },
            &H,
            &channel_llrs,
        );

        for (idx, edge) in &mut decoder.core.0.edges.iter_mut().enumerate() {
            edge.msg_cn_to_vn = (idx + 1) as f64;
        }

        //     1 0 0 1 1 0 1
        // H = 0 1 0 0 1 1 1
        //     0 0 1 1 0 1 1
        //
        //            1 0 0 2  3 0  4
        // L_{i<-j} = 0 5 0 0  6 7  8
        //            0 0 9 10 0 11 12

        let expected1 = vec![6.0, 7.0, 8.0, 10.0, 11.0, 12.0];
        let got1: Vec<f64> = decoder
            .get_cn_to_vn_tail(1, 3)
            .iter()
            .map(|e| e.msg_cn_to_vn)
            .collect();

        let expected2 = vec![3.0, 4.0, 6.0, 7.0, 8.0, 11.0, 12.0];
        let got2: Vec<f64> = decoder
            .get_cn_to_vn_tail(0, 4)
            .iter()
            .map(|e| e.msg_cn_to_vn)
            .collect();

        let expected3 = vec![9.0, 10.0, 11.0, 12.0];
        let got3: Vec<f64> = decoder
            .get_cn_to_vn_tail(2, 0)
            .iter()
            .map(|e| e.msg_cn_to_vn)
            .collect();

        assert_eq!(expected1, got1);
        assert_eq!(expected2, got2);
        assert_eq!(expected3, got3);
    }

    #[test]
    fn test_window_result_manipulation() {
        #[allow(non_snake_case)]
        let H = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1],
            &[0, 0, 0, 0, 0, 1, 1, 1],
        ]);

        let channel_llrs = Vec::<f64>::zeros(H.cols());

        let decoder = WindowingSyndromeBpDecoder::<
            SimpleSyndromeBpDecoder<SyndromeMinSumCore>,
        >::new(
            Settings {
                pass_soft_info: false,
                F: 2,
                W: 3,
            },
            bp::Settings { max_iter: 32 },
            &H,
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

        let s: Vec<u8> = (0..H.rows()).map(|v| v as u8).collect();

        // Window 1

        let e_hat: Vec<u8> = (0..6).map(|v| v as u8).collect();
        let got = decoder.get_commited_e_hat(&e_hat, 0);
        let expected = vec![0, 1, 2, 3];
        assert_eq!(expected, got);

        let got = decoder.cut_out_current_window_syndrome(&s, 0);
        let expected = vec![0, 1, 2, 3, 4, 5];
        assert_eq!(expected, got);

        let e_hat = vec![1, 0, 0, 1, 0, 0, 0, 0];
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

        let e_hat = vec![1, 0, 0, 1, 0, 0, 0, 0];
        let got = decoder.get_next_window_syndrome_diff(&e_hat, 1);
        let expected = Vec::<u8>::new();
        assert_eq!(expected, got);
    }

    #[test]
    fn test_soft_info_passing() {
        #[allow(non_snake_case)]
        let H = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1],
            &[0, 0, 0, 0, 0, 1, 1, 1],
        ]);

        let channel_llrs = Vec::<f64>::zeros(H.cols());

        let mut decoder = WindowingSyndromeBpDecoder::<
            SimpleSyndromeBpDecoder<SyndromeMinSumCore>,
        >::new(
            Settings {
                pass_soft_info: false,
                F: 2,
                W: 3,
            },
            bp::Settings { max_iter: 32 },
            &H,
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

        for (idx, e) in &mut decoder.window_decoders[0]
            .core
            .0
            .edges
            .iter_mut()
            .enumerate()
        {
            e.msg_cn_to_vn = (idx + 1) as f64;
        }

        for (idx, e) in &mut decoder.window_decoders[1]
            .core
            .0
            .edges
            .iter_mut()
            .enumerate()
        {
            e.msg_cn_to_vn = (idx + 1) as f64;
        }

        decoder.get_soft_info_from_previous_window(1);

        let expected =
            vec![12.0, 13.0, 15.0, 16.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let got: Vec<f64> = decoder.window_decoders[1]
            .core
            .0
            .edges
            .iter()
            .map(|e| e.msg_cn_to_vn)
            .collect();
        assert_eq!(expected, got);
    }

    #[test]
    fn test_soft_init_set() {
        #[allow(non_snake_case)]
        let H = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1],
            &[0, 0, 0, 0, 0, 1, 1, 1],
        ]);

        let channel_llrs: Vec<f64> = (0..H.cols()).map(|_| 0.0).collect();

        let mut decoder1 = SimpleSyndromeBpDecoder::<SyndromeMinSumCore>::new(
            bp::Settings { max_iter: 32 },
            &H,
            &channel_llrs,
        );

        for (idx, edge) in &mut decoder1.core.0.edges.iter_mut().enumerate() {
            edge.msg_cn_to_vn = (idx + 1) as f64;
        }

        let mut decoder2 = SimpleSyndromeBpDecoder::<SyndromeMinSumCore>::new(
            bp::Settings { max_iter: 32 },
            &H,
            &channel_llrs,
        );

        for (idx, edge) in &mut decoder2.core.0.edges.iter_mut().enumerate() {
            edge.msg_cn_to_vn = (idx + 1) as f64;
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

        // Decoder 2:
        //            12 13 0  0  0  0  0  0
        //            15 16 0  0  0  0  0  0
        //            0  17 18 19 0  0  0  0
        // L_{i<-j} = 0  20 21 22 0  0  0  0
        //            0  0  0  11 12 13 0  0
        //            0  0  0  14 15 16 0  0
        //            0  0  0  0  0  17 18 19
        //            0  0  0  0  0  20 21 22

        let soft_info = decoder1.get_cn_to_vn_tail(4, 4);
        decoder2.set_cn_to_vn_head(&soft_info, 4, 4);

        let expected = vec![
            12.0, 13.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 11.0,
            12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
        ];
        let got: Vec<f64> = decoder2
            .core
            .0
            .edges
            .iter()
            .map(|e| e.msg_cn_to_vn)
            .collect();

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

        let soft_info = decoder1.get_cn_to_vn_tail(0, 0);
        decoder2.set_cn_to_vn_head(&soft_info, 8, 8);

        let expected = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
        ];
        let got: Vec<f64> = decoder2
            .core
            .0
            .edges
            .iter()
            .map(|e| e.msg_cn_to_vn)
            .collect();

        assert_eq!(expected, got);
    }
}
