use crate::decoders::Decoder;
use crate::decoders::core::Edge;

pub struct OverlapInfo {
    pub start_positions: Vec<(usize, usize)>,
    pub end_positions: Vec<(usize, usize)>,
}

#[derive(Clone)]
pub struct Settings {
    pub max_iter: usize,
}

pub trait SoftInit {
    fn set_vn_to_cn_messages(&mut self, msgs: &Vec<Edge>);
}

pub struct SoftInitBp<WindowDecoder: SoftInit> {
    settings: Settings,
    window_decoders: Vec<WindowDecoder>,
    overlap_info: OverlapInfo,
}

// TODO: Implement windowing in new() function

impl<WindowDecoder> Decoder for SoftInitBp<WindowDecoder>
where
    WindowDecoder: SoftInit,
    WindowDecoder: Decoder,
{
    fn decode(&mut self, s: &[u8]) -> Vec<u8> {
        //             m = hz.shape[0]
        //
        // # TODO: Why the -2? (maybe the final lossless measurement?)
        // num_trials = detector_results.shape[0]
        // num_rounds = detector_results.shape[1] // m - 2
        //
        // #
        // # Compute number of windows
        // #
        //
        // if 2 + num_rounds - W >= 0:
        //     # num_cor_rounds = num of windows before the last window
        //     num_cor_rounds = (2 + num_rounds - W) // F
        //
        //     # we can slide one more window if the remaining rounds > W
        //     if (2 + num_rounds - W) % F != 0:
        //         num_cor_rounds += 1
        // else:
        //     num_cor_rounds = 0
        //     warnings.warn(
        //         "Window size larger than the syndrome extraction rounds: Doing"
        //         " whole history correction"
        //     )
        //
        // #
        // # Prepare windows and decoders
        // #
        //
        // win_check_set, win_observable_set, win_priors_set, win_update = spacetime(
        //     circuit, hz, W, F, num_cor_rounds
        // )
        //
        // window_decoders: Sequence[BpDecoder] = []
        // for trial_idx in range(len(win_check_set) - 1):
        //     decoder1_params[error_rate_name1] = win_priors_set[trial_idx]
        //     decoder_i = decoder1(win_check_set[trial_idx], **decoder1_params)
        //     window_decoders.append(decoder_i)
        //
        // decoder2_params[error_rate_name2] = win_priors_set[len(win_check_set) - 1]
        // decoder_i = decoder2(win_check_set[len(win_check_set) - 1], **decoder2_params)
        // window_decoders.append(decoder_i)
        //
        // #
        // # Prepare overlap info
        // #
        //
        // col_start_indices = reconstruct_window_start_col_indices(win_observable_set)
        // overlap_begin_positions, overlap_end_positions = get_overlap_info(
        //     col_start_indices, W, F, m, win_check_set
        // )
        //
        // #
        // # Perform decoding
        // #
        //
        // resulting_logical_predictions = np.zeros((num_trials, lz.shape[0]), dtype=int)
        //
        // last_cn_to_vn_msgs = None
        //
        // iterator = tqdm(range(num_trials)) if tqdm_on else range(num_trials)
        // for trial_idx in iterator:
        //     accumulated_correction = np.zeros(win_observable_set[0].shape[0], dtype=int)
        //     syn_update = np.zeros(m, dtype=int)
        //
        //     for win_idx in range(num_cor_rounds):
        //         # Extract syndrome for window and update based on decoding of last window
        //         win_row_beg = F * win_idx * m
        //         win_row_end = (F * win_idx + W) * m
        //         diff_syndrome = detector_results[trial_idx, win_row_beg:win_row_end].copy()
        //         diff_syndrome[:m] = (diff_syndrome[:m] + syn_update) % 2
        //
        //         # Perform decoding
        //
        //         if win_idx > 0:
        //             cn_to_vn_msgs = window_decoders[win_idx].get_cn_to_vn_msgs()
        //             cn_to_vn_msgs[
        //                 : overlap_end_positions[win_idx][0],
        //                 : overlap_end_positions[win_idx][1],
        //             ] = last_cn_to_vn_msgs
        //             window_decoders[win_idx].set_cn_to_vn_msgs(cn_to_vn_msgs)
        //
        //         e_hat = getattr(window_decoders[win_idx], dec_func_name1)(diff_syndrome)
        //
        //         last_cn_to_vn_msgs = window_decoders[win_idx].get_cn_to_vn_msgs()[
        //             overlap_begin_positions[win_idx][0] :,
        //             overlap_begin_positions[win_idx][1] :,
        //         ]
        //
        //         syn_update = (
        //             win_update[win_idx] @ e_hat[: win_observable_set[win_idx].shape[1]] % 2
        //         )
        //
        //         correction = (
        //             win_observable_set[win_idx]
        //             @ e_hat[: win_observable_set[win_idx].shape[1]]
        //             % 2
        //         )
        //         accumulated_correction = (accumulated_correction + correction) % 2
        //
        //     # Last round
        //
        //     # In the last round we just correct the whole window
        //     # syndrome of last round
        //
        //     last_win_row_beg = (F * num_cor_rounds) * m
        //     diff_syndrome = detector_results[trial_idx, last_win_row_beg:].copy()
        //
        //     diff_syndrome[:m] = (diff_syndrome[:m] + syn_update) % 2
        //
        //     # Observable flips based on correction
        //
        //     cn_to_vn_msgs = window_decoders[num_cor_rounds].get_cn_to_vn_msgs()
        //     cn_to_vn_msgs[
        //         : overlap_end_positions[num_cor_rounds - 1][0],
        //         : overlap_end_positions[num_cor_rounds - 1][1],
        //     ] = last_cn_to_vn_msgs
        //     window_decoders[num_cor_rounds].set_cn_to_vn_msgs(cn_to_vn_msgs)
        //
        //     e_hat = getattr(window_decoders[num_cor_rounds], dec_func_name2)(diff_syndrome)
        //
        //     correction = win_observable_set[num_cor_rounds] @ e_hat % 2
        //     accumulated_correction = (accumulated_correction + correction) % 2
        //
        //     resulting_logical_predictions[trial_idx, :] = accumulated_correction
        //  return resulting_logical_predictions

        todo!()
    }
}
