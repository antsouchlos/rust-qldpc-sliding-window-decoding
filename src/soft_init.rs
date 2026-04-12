use sprs::CsMat;

use crate::decoders::Decoder;
use crate::decoders::bp::SyndromeBpDecoder;
use crate::decoders::core::{Edge, SyndromeBpStrategy};

#[derive(PartialEq, Debug)]
pub struct OverlapInfo {
    pub begin_positions: Vec<(usize, usize)>,
    pub end_positions: Vec<(usize, usize)>,
}

#[derive(Clone)]
pub struct Settings {
    pub max_iter: usize,
    pub m: usize,
    pub step: usize,
    pub window: usize,
}

pub trait SoftInit {
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

pub struct SoftInitBp<WindowDecoder: SoftInit> {
    pub settings: Settings,
    pub window_decoders: Vec<WindowDecoder>,
    pub overlap_info: OverlapInfo,
}

// TODO: Write unit tests
impl<Core: SyndromeBpStrategy> SoftInit for SyndromeBpDecoder<Core> {
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

/// Two more measurement rounds than the number of syndrome extraction
/// rounds exist, because of the initial and final measurement rounds in
/// the circuit.
#[allow(non_snake_case)]
fn get_num_windows(num_rounds: usize, W: usize, F: usize) -> usize {
    let num_rounds = num_rounds as isize;
    #[allow(non_snake_case)]
    let W = W as isize;
    #[allow(non_snake_case)]
    let F = F as isize;

    if 2 + num_rounds - W >= 0 {
        let mut result = 1 + (2 + num_rounds - W) / F;

        if (2 + num_rounds - W) % F != 0 {
            result += 1
        }

        result as usize
    } else {
        1
    }
}

// TODO: Doc
#[allow(non_snake_case)]
fn get_window_borders(
    H: &CsMat<u8>,
    m: usize,
    num_rounds: usize,
    W: usize,
    F: usize,
) -> Vec<((usize, usize), (usize, usize))> {
    let num_windows = get_num_windows(num_rounds, W, F);

    let i_max = |j: usize| {
        let H_csr = H.to_csr();
        let row = H_csr.outer_view(j).unwrap();
        let indices = row.indices();

        indices.last().unwrap().clone()
    };

    let mut window_borders =
        Vec::<((usize, usize), (usize, usize))>::with_capacity(
            num_windows as usize - 1,
        );

    let mut col_begin = 0;

    for k in 0..(num_windows - 1) {
        let row_begin = (k * F * m) as usize;
        let row_end = ((k * F + W) * m - 1) as usize;

        let col_end = (row_begin..row_end + 1)
            .map(|j| i_max(j as usize))
            .max()
            .unwrap();

        let row_end_corr = ((k + 1) * F * m - 1) as usize;
        let col_end_corr = (row_begin..row_end_corr + 1)
            .map(|j| i_max(j as usize))
            .max()
            .unwrap();

        window_borders.push(((row_begin, col_begin), (row_end, col_end)));

        col_begin = col_end_corr + 1;
    }

    window_borders.push((
        ((num_windows - 1) * F * m, col_begin),
        (H.rows() - 1, H.cols() - 1),
    ));

    window_borders
}

fn get_overlap_info(
    window_borders: &Vec<((usize, usize), (usize, usize))>,
) -> OverlapInfo {
    let mut begin_positions =
        Vec::<(usize, usize)>::with_capacity(window_borders.len());
    let mut end_positions =
        Vec::<(usize, usize)>::with_capacity(window_borders.len());

    for idx in 0..window_borders.len() - 1 {
        let row_begin = window_borders[idx + 1].0.0 as isize
            - window_borders[idx].0.0 as isize;
        let col_begin = window_borders[idx + 1].0.1 as isize
            - window_borders[idx].0.1 as isize;

        begin_positions.push((row_begin as usize, col_begin as usize));
    }

    for idx in 1..window_borders.len() {
        let row_end = window_borders[idx - 1].1.0 as isize
            - window_borders[idx].0.0 as isize;
        let col_end = window_borders[idx - 1].1.1 as isize
            - window_borders[idx].0.1 as isize;

        end_positions.push((row_end as usize, col_end as usize));
    }

    OverlapInfo {
        begin_positions,
        end_positions,
    }
}

#[allow(non_snake_case)]
fn split_pcm(
    H: &CsMat<u8>,
    window_borders: &Vec<((usize, usize), (usize, usize))>,
) -> Vec<CsMat<u8>> {
    let mut result = Vec::<CsMat<u8>>::with_capacity(window_borders.len());

    for &((row_begin, col_begin), (row_end, col_end)) in window_borders {
        let mut triplets = Vec::<(usize, usize, u8)>::new();

        for row in row_begin..row_end + 1 {
            #[allow(non_snake_case)]
            let H_csr = H.to_csr();

            let row_vec = H_csr.outer_view(row).unwrap();
            for (col, &val) in row_vec.iter() {
                if col >= col_begin && col <= col_end {
                    triplets.push((row - row_begin, col - col_begin, val));
                }
            }
        }

        let nrows = row_end - row_begin + 1;
        let ncols = col_end - col_begin + 1;
        let mut tri = sprs::TriMat::new((nrows, ncols));
        for (i, j, v) in triplets {
            tri.add_triplet(i, j, v);
        }
        result.push(tri.to_csr());
    }

    result
}

fn split_priors(
    priors: &[f64],
    window_borders: &Vec<((usize, usize), (usize, usize))>,
) -> Vec<Vec<f64>> {
    let mut result = Vec::<Vec<f64>>::with_capacity(window_borders.len());

    for &((_, col_begin), (_, col_end)) in window_borders {
        result.push(priors[col_begin..col_end + 1].to_vec());
    }

    result
}

#[cfg(test)]
mod tests {
    use pyo3::{
        Python,
        types::{PyAnyMethods, PyDict, PyDictMethods},
    };
    use sprs::TriMat;

    use super::*;

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

    /// Two more measurement rounds than the number of syndrome extraction
    /// rounds exist, because of the initial and final measurement rounds in
    /// the circuit.
    #[test]
    fn test_get_num_windows() {
        assert_eq!(get_num_windows(4 - 2, 3, 1), 2);
        assert_eq!(get_num_windows(16 - 2, 5, 3), 5);
    }

    #[test]
    fn test_get_window_borders() {
        #[allow(non_snake_case)]
        let H_1 = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1],
            &[0, 0, 0, 0, 0, 1, 1, 1],
        ]);

        #[allow(non_snake_case)]
        let H_2 = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0, 0],
            &[1, 1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 0, 1, 1, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0, 0],
            &[0, 0, 0, 0, 1, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]);

        let window_borders_1 = get_window_borders(&H_1, 2, 4 - 2, 3, 1);
        let expected_1 = vec![((0, 0), (5, 5)), ((2, 2), (7, 7))];

        assert_eq!(window_borders_1, expected_1);

        let window_borders_2 = get_window_borders(&H_2, 2, 5 - 2, 3, 1);
        let expected_2 =
            vec![((0, 0), (5, 5)), ((2, 3), (7, 7)), ((4, 4), (8, 8))];

        assert_eq!(window_borders_2, expected_2);
    }

    #[test]
    fn test_get_overlap_info() {
        #[allow(non_snake_case)]
        let H_1 = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1],
            &[0, 0, 0, 0, 0, 1, 1, 1],
        ]);

        #[allow(non_snake_case)]
        let H_2 = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0, 0],
            &[1, 1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 0, 1, 1, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0, 0],
            &[0, 0, 0, 0, 1, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]);

        let window_borders_1 = get_window_borders(&H_1, 2, 4 - 2, 3, 1);
        let overlap_info_1 = get_overlap_info(&window_borders_1);
        let expected_1 = OverlapInfo {
            begin_positions: vec![(2, 2)],
            end_positions: vec![(3, 3)],
        };

        assert_eq!(overlap_info_1, expected_1);

        let window_borders_2 = get_window_borders(&H_2, 2, 5 - 2, 3, 1);
        let overlap_info_2 = get_overlap_info(&window_borders_2);
        let expected_2 = OverlapInfo {
            begin_positions: vec![(2, 3), (2, 1)],
            end_positions: vec![(3, 2), (3, 3)],
        };

        assert_eq!(overlap_info_2, expected_2);
    }

    #[test]
    fn test_split_pcm() {
        #[allow(non_snake_case)]
        let H_1 = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1],
            &[0, 0, 0, 0, 0, 1, 1, 1],
        ]);

        #[allow(non_snake_case)]
        let H_2 = csr_from_dense(&[
            &[1, 1, 0, 0, 0, 0, 0, 0, 0],
            &[1, 1, 1, 0, 0, 0, 0, 0, 0],
            &[0, 0, 1, 1, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0, 0],
            &[0, 0, 0, 1, 1, 1, 0, 0, 0],
            &[0, 0, 0, 0, 1, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]);

        let window_borders_1 = get_window_borders(&H_1, 2, 4 - 2, 3, 1);
        #[allow(non_snake_case)]
        let H_win1 = split_pcm(&H_1, &window_borders_1);
        let expected_win1 = vec![
            csr_from_dense(&[
                &[1, 1, 0, 0, 0, 0],
                &[1, 1, 0, 0, 0, 0],
                &[0, 1, 1, 1, 0, 0],
                &[0, 1, 1, 1, 0, 0],
                &[0, 0, 0, 1, 1, 1],
                &[0, 0, 0, 1, 1, 1],
            ]),
            csr_from_dense(&[
                &[1, 1, 0, 0, 0, 0],
                &[1, 1, 0, 0, 0, 0],
                &[0, 1, 1, 1, 0, 0],
                &[0, 1, 1, 1, 0, 0],
                &[0, 0, 0, 1, 1, 1],
                &[0, 0, 0, 1, 1, 1],
            ]),
        ];
        assert_eq!(H_win1, expected_win1);

        let window_borders_2 = get_window_borders(&H_2, 2, 5 - 2, 3, 1);
        #[allow(non_snake_case)]
        let H_win2 = split_pcm(&H_2, &window_borders_2);
        let expected_win2 = vec![
            csr_from_dense(&[
                &[1, 1, 0, 0, 0, 0],
                &[1, 1, 1, 0, 0, 0],
                &[0, 0, 1, 1, 0, 0],
                &[0, 1, 1, 1, 0, 0],
                &[0, 0, 0, 1, 1, 1],
                &[0, 0, 0, 1, 1, 1],
            ]),
            csr_from_dense(&[
                &[1, 0, 0, 0, 0],
                &[1, 0, 0, 0, 0],
                &[1, 1, 1, 0, 0],
                &[1, 1, 1, 0, 0],
                &[0, 1, 1, 1, 1],
                &[0, 0, 1, 1, 1],
            ]),
            csr_from_dense(&[
                &[1, 1, 0, 0, 0],
                &[1, 1, 0, 0, 0],
                &[1, 1, 1, 1, 0],
                &[0, 1, 1, 1, 0],
                &[0, 0, 0, 1, 1],
            ]),
        ];
        assert_eq!(H_win2, expected_win2);
    }

    #[test]
    fn test_split_pcm_vs_quits() {
        unsafe {
            pyo3::ffi::Py_InitializeEx(0);
        }

        #[allow(non_snake_case)]
        let mut win_Hs = Vec::<Vec<Vec<u8>>>::new();
        #[allow(non_snake_case)]
        let mut win_priors = Vec::<Vec<f64>>::new();
        #[allow(non_snake_case)]
        let mut circuit_H = Vec::<Vec<u8>>::new();
        #[allow(non_snake_case)]
        let mut circuit_priors = Vec::<f64>::new();

        let mut m: usize = 0;
        let mut num_rounds: usize = 0;
        #[allow(non_snake_case)]
        let mut W: usize = 0;
        #[allow(non_snake_case)]
        let mut F: usize = 0;

        Python::try_attach(|py| {
                let locals = PyDict::new(py);
                py.run(
                    c"
from quits.decoder import spacetime, detector_error_model_to_matrix
from quits.qldpc_code import BbCode
from quits import ErrorModel, CircuitBuildOptions

num_rounds = 12
p = 0.001

code = BbCode(l=6, m=6, A_x_pows=[3], A_y_pows=[1, 2], B_x_pows=[1, 2], B_y_pows=[3])

circuit = code.build_circuit(
    error_model=ErrorModel(p, p, p, p),
    num_rounds=num_rounds,
    basis='Z',
    circuit_build_options=CircuitBuildOptions(),
    seed=1,
)

W = 5
F = 3

if 2 + num_rounds - W >= 0:
    # num_cor_rounds = num of windows before the last window
    num_cor_rounds = (2 + num_rounds - W) // F

    # we can slide one more window if the remaining rounds > W
    if (2 + num_rounds - W) % F != 0:
        num_cor_rounds += 1
else:
    num_cor_rounds = 0
    warnings.warn(
        'Window size larger than the syndrome extraction rounds: Doing'
        ' whole history correction'
    )

win_Hs, _, win_priors, _ = spacetime(
    circuit, code.hz, W, F, num_cor_rounds
)

model = circuit.detector_error_model()
circuit_H, _, circuit_priors = detector_error_model_to_matrix(model)

m,_ = code.hz.shape
win_Hs_dense = [mat.toarray().astype('uint8').tolist() for mat in win_Hs]
circuit_H_dense = circuit_H.toarray().astype('uint8').tolist()
                    ",
                    None,
                    Some(&locals),
                )
                .unwrap();

                win_Hs = locals
                   .get_item("win_Hs_dense")
                   .unwrap()
                   .unwrap()
                   .extract()
                   .unwrap();

                win_priors = locals
                   .get_item("win_priors")
                   .unwrap()
                   .unwrap()
                   .extract()
                   .unwrap();

                circuit_H = locals
                   .get_item("circuit_H_dense")
                   .unwrap()
                   .unwrap()
                   .extract()
                   .unwrap();

                circuit_priors = locals
                   .get_item("circuit_priors")
                   .unwrap()
                   .unwrap()
                   .extract()
                   .unwrap();

                m = locals
                   .get_item("m")
                   .unwrap()
                   .unwrap()
                   .extract()
                   .unwrap();

                W = locals
                   .get_item("W")
                   .unwrap()
                   .unwrap()
                   .extract()
                   .unwrap();

                F = locals
                   .get_item("F")
                   .unwrap()
                   .unwrap()
                   .extract()
                   .unwrap();

                num_rounds = locals
                   .get_item("num_rounds")
                   .unwrap()
                   .unwrap()
                   .extract()
                   .unwrap();
            })
            .expect("Python interpreter not initialized");

        #[allow(non_snake_case)]
        let win_Hs: Vec<CsMat<u8>> = win_Hs
            .iter()
            .map(|v| {
                csr_from_dense(
                    &v.iter().map(|row| row.as_slice()).collect::<Vec<&[u8]>>(),
                )
            })
            .collect();

        #[allow(non_snake_case)]
        let circuit_H = csr_from_dense(
            &circuit_H
                .iter()
                .map(|row| row.as_slice())
                .collect::<Vec<&[u8]>>(),
        );

        let window_borders =
            get_window_borders(&circuit_H, m, num_rounds, W, F);
        #[allow(non_snake_case)]
        let split_Hs = split_pcm(&circuit_H, &window_borders);
        let split_priors = split_priors(&circuit_priors, &window_borders);

        assert_eq!(win_Hs, split_Hs);
        assert_eq!(win_priors, split_priors);
    }
}

// /// Compute (A @ v) mod 2 where A is a sparse GF(2) matrix in CSR format.
// fn gf2_matvec(a: &CsMat<u8>, v: &[u8]) -> Vec<u8> {
//     let mut result = vec![0u8; a.rows()];
//     for (row_idx, row_vec) in a.outer_iterator().enumerate() {
//         for (col_idx, &val) in row_vec.iter() {
//             if val != 0 {
//                 result[row_idx] ^= v[col_idx];
//             }
//         }
//     }
//     result
// }
//
// impl<WindowDecoder> Decoder for SoftInitBp<WindowDecoder>
// where
//     WindowDecoder: SoftInit + Decoder,
// {
//     fn decode(&mut self, s: &[u8]) -> Vec<u8> {
//         let m = self.settings.m;
//         let f = self.settings.step;
//         let w = self.settings.window;
//
//         // Compute number of sliding windows before the final window.
//         // Mirrors the Python formula: num_cor_rounds = ceil((total_rounds - W) / F),
//         // where total_rounds = s.len() / m (the -2 from the Python is already baked in).
//         let total_rounds = s.len() / m;
//         let diff = total_rounds.saturating_sub(w);
//         let num_cor_rounds = if diff == 0 { 0 } else { (diff + f - 1) / f };
//
//         let num_observables = self.win_observable_sets[0].rows();
//         let mut accumulated_correction = vec![0u8; num_observables];
//         let mut syn_update = vec![0u8; m];
//         let mut last_msgs: Option<Vec<Edge>> = None;
//
//         for win_idx in 0..num_cor_rounds {
//             // Extract windowed syndrome and apply the syndrome carry from the previous window.
//             let win_row_beg = f * win_idx * m;
//             let mut win_syn = s[win_row_beg..win_row_beg + w * m].to_vec();
//             for i in 0..m {
//                 win_syn[i] ^= syn_update[i];
//             }
//
//             // Soft initialization: inject cn_to_vn overlap tail from the previous window.
//             if win_idx > 0 {
//                 if let Some(ref msgs) = last_msgs {
//                     let (end_row, end_col) =
//                         self.overlap_info.end_positions[win_idx];
//                     self.window_decoders[win_idx]
//                         .set_cn_to_vn_head(msgs, end_row, end_col);
//                 }
//             }
//
//             let e_hat = self.window_decoders[win_idx].decode(&win_syn);
//
//             // Save the cn_to_vn overlap tail for the next window.
//             let (start_row, start_col) =
//                 self.overlap_info.start_positions[win_idx];
//             last_msgs = Some(
//                 self.window_decoders[win_idx]
//                     .get_cn_to_vn_tail(start_row, start_col),
//             );
//
//             // Syndrome carry: propagate the effect of committed errors to the next window's
//             // first syndrome row.
//             let k = self.win_observable_sets[win_idx].cols();
//             syn_update = gf2_matvec(&self.win_updates[win_idx], &e_hat[..k]);
//
//             // Accumulate logical correction.
//             let correction =
//                 gf2_matvec(&self.win_observable_sets[win_idx], &e_hat[..k]);
//             for i in 0..num_observables {
//                 accumulated_correction[i] ^= correction[i];
//             }
//         }
//
//         // Final window: covers the remaining syndrome from F*num_cor_rounds*m onwards.
//         let last_idx = num_cor_rounds;
//         let last_beg = f * num_cor_rounds * m;
//         let mut last_syn = s[last_beg..].to_vec();
//         for i in 0..m {
//             last_syn[i] ^= syn_update[i];
//         }
//
//         if num_cor_rounds > 0 {
//             if let Some(ref msgs) = last_msgs {
//                 // The last window's overlap head is aligned with the last sliding window's
//                 // tail, so we reuse end_positions[num_cor_rounds - 1].
//                 let (end_row, end_col) =
//                     self.overlap_info.end_positions[num_cor_rounds - 1];
//                 self.window_decoders[last_idx]
//                     .set_cn_to_vn_head(msgs, end_row, end_col);
//             }
//         }
//
//         let e_hat = self.window_decoders[last_idx].decode(&last_syn);
//
//         // The last window uses the full e_hat (no column slice).
//         let correction =
//             gf2_matvec(&self.win_observable_sets[last_idx], &e_hat);
//         for i in 0..num_observables {
//             accumulated_correction[i] ^= correction[i];
//         }
//
//         accumulated_correction
//     }
// }
