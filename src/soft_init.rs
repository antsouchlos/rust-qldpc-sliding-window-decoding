use sprs::CsMat;

use crate::decoders::Decoder;
use crate::decoders::bp::SimpleSyndromeBpDecoder;
use crate::decoders::core::{Edge, SyndromeBpDecoder, SyndromeBpStrategy};
use crate::windowing::{OverlapInfo, split_pcm, split_priors};
use crate::windowing::{get_overlap_info, get_window_borders};

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct Settings {
    pub max_iter: usize,
    pub F: usize,
    pub W: usize,
}

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
}

#[derive(Clone)]
pub struct WindowingSyndromeBpDecoder<InnerDecoder>
where
    InnerDecoder: SoftInitBpDecoder,
{
    pub settings: Settings,
    pub window_decoders: Vec<InnerDecoder>,
    pub overlap_info: OverlapInfo,
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
        #[allow(non_snake_case)]
        let win_Hs = split_pcm(&H, &window_borders);
        let win_priors = split_priors(&channel_llrs, &window_borders);
        let overlap_info = get_overlap_info(&window_borders);

        #[allow(non_snake_case)]
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
            overlap_info,
        }
    }

    fn reset(&mut self) {
        for decoder in self.window_decoders.iter_mut() {
            decoder.reset();
        }
    }
}

// TODO: Implement
impl<InnerDecoder> Decoder for WindowingSyndromeBpDecoder<InnerDecoder>
where
    InnerDecoder: SoftInitBpDecoder,
{
    type Settings = Settings;

    // TODO: Update syndrome while decoding
    fn decode(&mut self, s: &[u8]) -> Vec<u8> {
        let mut e_hat_total = Vec::<u8>::new();

        for win_idx in 0..self.window_decoders.len() {
            let e_hat = self.window_decoders[win_idx].decode(s);
            e_hat_total.extend(
                e_hat[0..self.overlap_info.begin_positions[win_idx].1].iter(),
            );
        }

        e_hat_total
    }
}

// TODO: Write unit tests
// TODO: Implement this for BPGD as well
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
        end_row: usize,
        end_col: usize,
    ) {
        // Two-pointer merge: both edge lists are in (row, col) sorted CSR order.
        let mut msg_iter = msgs.iter().peekable();
        for edge in self.core.get_state().edges.iter_mut() {
            if edge.row >= end_row || edge.col >= end_col {
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
}

// // /// Compute (A @ v) mod 2 where A is a sparse GF(2) matrix in CSR format.
// // fn gf2_matvec(a: &CsMat<u8>, v: &[u8]) -> Vec<u8> {
// //     let mut result = vec![0u8; a.rows()];
// //     for (row_idx, row_vec) in a.outer_iterator().enumerate() {
// //         for (col_idx, &val) in row_vec.iter() {
// //             if val != 0 {
// //                 result[row_idx] ^= v[col_idx];
// //             }
// //         }
// //     }
// //     result
// // }
// //
// // impl<WindowDecoder> Decoder for SoftInitBp<WindowDecoder>
// // where
// //     WindowDecoder: SoftInit + Decoder,
// // {
// //     fn decode(&mut self, s: &[u8]) -> Vec<u8> {
// //         let m = self.settings.m;
// //         let f = self.settings.step;
// //         let w = self.settings.window;
// //
// //         // Compute number of sliding windows before the final window.
// //         // Mirrors the Python formula: num_cor_rounds = ceil((total_rounds - W) / F),
// //         // where total_rounds = s.len() / m (the -2 from the Python is already baked in).
// //         let total_rounds = s.len() / m;
// //         let diff = total_rounds.saturating_sub(w);
// //         let num_cor_rounds = if diff == 0 { 0 } else { (diff + f - 1) / f };
// //
// //         let num_observables = self.win_observable_sets[0].rows();
// //         let mut accumulated_correction = vec![0u8; num_observables];
// //         let mut syn_update = vec![0u8; m];
// //         let mut last_msgs: Option<Vec<Edge>> = None;
// //
// //         for win_idx in 0..num_cor_rounds {
// //             // Extract windowed syndrome and apply the syndrome carry from the previous window.
// //             let win_row_beg = f * win_idx * m;
// //             let mut win_syn = s[win_row_beg..win_row_beg + w * m].to_vec();
// //             for i in 0..m {
// //                 win_syn[i] ^= syn_update[i];
// //             }
// //
// //             // Soft initialization: inject cn_to_vn overlap tail from the previous window.
// //             if win_idx > 0 {
// //                 if let Some(ref msgs) = last_msgs {
// //                     let (end_row, end_col) =
// //                         self.overlap_info.end_positions[win_idx];
// //                     self.window_decoders[win_idx]
// //                         .set_cn_to_vn_head(msgs, end_row, end_col);
// //                 }
// //             }
// //
// //             let e_hat = self.window_decoders[win_idx].decode(&win_syn);
// //
// //             // Save the cn_to_vn overlap tail for the next window.
// //             let (start_row, start_col) =
// //                 self.overlap_info.start_positions[win_idx];
// //             last_msgs = Some(
// //                 self.window_decoders[win_idx]
// //                     .get_cn_to_vn_tail(start_row, start_col),
// //             );
// //
// //             // Syndrome carry: propagate the effect of committed errors to the next window's
// //             // first syndrome row.
// //             let k = self.win_observable_sets[win_idx].cols();
// //             syn_update = gf2_matvec(&self.win_updates[win_idx], &e_hat[..k]);
// //
// //             // Accumulate logical correction.
// //             let correction =
// //                 gf2_matvec(&self.win_observable_sets[win_idx], &e_hat[..k]);
// //             for i in 0..num_observables {
// //                 accumulated_correction[i] ^= correction[i];
// //             }
// //         }
// //
// //         // Final window: covers the remaining syndrome from F*num_cor_rounds*m onwards.
// //         let last_idx = num_cor_rounds;
// //         let last_beg = f * num_cor_rounds * m;
// //         let mut last_syn = s[last_beg..].to_vec();
// //         for i in 0..m {
// //             last_syn[i] ^= syn_update[i];
// //         }
// //
// //         if num_cor_rounds > 0 {
// //             if let Some(ref msgs) = last_msgs {
// //                 // The last window's overlap head is aligned with the last sliding window's
// //                 // tail, so we reuse end_positions[num_cor_rounds - 1].
// //                 let (end_row, end_col) =
// //                     self.overlap_info.end_positions[num_cor_rounds - 1];
// //                 self.window_decoders[last_idx]
// //                     .set_cn_to_vn_head(msgs, end_row, end_col);
// //             }
// //         }
// //
// //         let e_hat = self.window_decoders[last_idx].decode(&last_syn);
// //
// //         // The last window uses the full e_hat (no column slice).
// //         let correction =
// //             gf2_matvec(&self.win_observable_sets[last_idx], &e_hat);
// //         for i in 0..num_observables {
// //             accumulated_correction[i] ^= correction[i];
// //         }
// //
// //         accumulated_correction
// //     }
// // }
