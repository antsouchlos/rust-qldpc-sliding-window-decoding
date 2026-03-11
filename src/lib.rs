use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use sprs::CsMat;

// ---------------------------------------------------------------------------
// Core BP Decoder Implementation
// ---------------------------------------------------------------------------

pub enum BpMethod {
    ProductSum,
    MinSum { alpha: f64 },
}

pub trait SoftInfoInnerDecoder {
    fn decode(&mut self, syndrome: &Array1<u8>) -> Array1<u8>;
    fn set_cn_to_vn_msgs(&mut self, msgs: CsMat<f64>);
    fn get_cn_to_vn_msgs(&self) -> CsMat<f64>;
}

#[allow(non_snake_case)]
struct BpDecoder {
    H: CsMat<u8>,
    cn_to_vn_msgs: CsMat<f64>,
    vn_to_cn_msgs: CsMat<f64>,
    channel_llrs: Array1<f64>,
    total_llrs: Array1<f64>,
    cn_to_vn_msgs_initialized_manually: bool,
    max_iter: usize,
    bp_method: BpMethod,
}

impl BpDecoder {
    #[allow(non_snake_case)]
    fn new(
        H: CsMat<u8>,
        channel_probs: &Vec<f64>,
        max_iter: usize,
        bp_method: BpMethod,
    ) -> BpDecoder {
        #[allow(non_snake_case)]
        let H = if H.is_csc() { H } else { H.to_csc() };

        let channel_llrs: Array1<f64> = Array1::from_vec(
            channel_probs
                .iter()
                .map(|val| ((1.0 - val) / val).ln())
                .collect(),
        );

        let cn_to_vn_msgs: CsMat<f64> = H.map(|_| 0.0);
        let vn_to_cn_msgs: CsMat<f64> = H.map(|_| 0.0);
        let total_llrs = Array1::<f64>::zeros(H.cols());

        BpDecoder {
            H,
            cn_to_vn_msgs,
            vn_to_cn_msgs,
            channel_llrs,
            total_llrs,
            cn_to_vn_msgs_initialized_manually: false,
            max_iter,
            bp_method,
        }
    }

    fn vn_update(&mut self) {
        let indptr_view = self.H.indptr();
        let indptr = indptr_view.raw_storage();

        let cn_to_vn = self.cn_to_vn_msgs.data();
        let vn_to_cn = self.vn_to_cn_msgs.data_mut();

        for i in 0..self.H.cols() {
            let col_i_indices = indptr[i]..indptr[i + 1];

            let total: f64 =
                self.channel_llrs[i] + cn_to_vn[col_i_indices.clone()].iter().sum::<f64>();

            for idx in col_i_indices {
                vn_to_cn[idx] = total - cn_to_vn[idx];
            }
        }
    }

    fn cn_update_spa(&mut self, syndrome: &Array1<u8>) {
        let indptr_view = self.H.indptr();
        let indptr = indptr_view.raw_storage();

        let indices = self.H.indices();
        let vn_to_cn = self.vn_to_cn_msgs.data();
        let cn_to_vn = self.cn_to_vn_msgs.data_mut();

        let mut totals: Vec<f64> = vec![1.0; self.H.rows()];
        for i in 0..self.H.cols() {
            for idx in indptr[i]..indptr[i + 1] {
                totals[indices[idx]] *= (vn_to_cn[idx] / 2.0).tanh();
            }
        }

        for i in 0..self.H.cols() {
            for idx in indptr[i]..indptr[i + 1] {
                let j = indices[idx];
                let mut product_excluding = totals[j] / (vn_to_cn[idx] / 2.0).tanh();

                if product_excluding > 1.0 - 1e-7 {
                    product_excluding = 1.0 - 1e-7;
                } else if product_excluding < -1.0 + 1e-7 {
                    product_excluding = -1.0 + 1e-7;
                }

                cn_to_vn[idx] = 2.0 * (1.0 - 2.0 * syndrome[j] as f64) * product_excluding.atanh();
            }
        }
    }

    fn cn_update_min_sum(&mut self, syndrome: &Array1<u8>, alpha: f64) {
        let indptr_view = self.H.indptr();
        let indptr = indptr_view.raw_storage();
        let indices = self.H.indices();
        let vn_to_cn = self.vn_to_cn_msgs.data();
        let cn_to_vn = self.cn_to_vn_msgs.data_mut();

        let mut min1: Vec<f64> = vec![f64::INFINITY; self.H.rows()];
        let mut min2: Vec<f64> = vec![f64::INFINITY; self.H.rows()];
        let mut sign: Vec<f64> = vec![1.0; self.H.rows()];

        for col in 0..self.H.cols() {
            for idx in indptr[col]..indptr[col + 1] {
                let row = indices[idx];
                let val = vn_to_cn[idx];
                let abs_val = val.abs();

                if val < 0.0 {
                    sign[row] = -sign[row];
                }

                if abs_val < min1[row] {
                    min2[row] = min1[row];
                    min1[row] = abs_val;
                } else if abs_val < min2[row] {
                    min2[row] = abs_val;
                }
            }
        }

        for col in 0..self.H.cols() {
            for idx in indptr[col]..indptr[col + 1] {
                let row = indices[idx];
                let val = vn_to_cn[idx];
                let abs_val = val.abs();

                let exclude_sign = if val < 0.0 { -sign[row] } else { sign[row] };
                let exclude_min = if abs_val == min1[row] {
                    min2[row]
                } else {
                    min1[row]
                };

                let syndrome_sign = 1.0 - 2.0 * syndrome[row] as f64;

                cn_to_vn[idx] = alpha * exclude_sign * exclude_min * syndrome_sign;
            }
        }
    }

    fn total_llrs(&mut self) {
        let indptr_view = self.H.indptr();
        let indptr = indptr_view.raw_storage();

        let cn_to_vn = self.cn_to_vn_msgs.data();

        for i in 0..self.H.cols() {
            let col_i_indices = indptr[i]..indptr[i + 1];

            let total: f64 =
                self.channel_llrs[i] + cn_to_vn[col_i_indices.clone()].iter().sum::<f64>();

            self.total_llrs[i] = total;
        }
    }
}

impl SoftInfoInnerDecoder for BpDecoder {
    fn decode(&mut self, syndrome: &Array1<u8>) -> Array1<u8> {
        if !self.cn_to_vn_msgs_initialized_manually {
            self.cn_to_vn_msgs = self.cn_to_vn_msgs.map(|_| 0.0);
        }
        self.cn_to_vn_msgs_initialized_manually = false;

        let mut e_hat: Array1<u8> =
            Array1::from_vec(self.channel_llrs.iter().map(|&v| (v < 0.0) as u8).collect());

        let alpha = match self.bp_method {
            BpMethod::ProductSum => None,
            BpMethod::MinSum { alpha } => Some(alpha),
        };

        for _ in 0..self.max_iter {
            self.vn_update();
            match alpha {
                None => self.cn_update_spa(syndrome),
                Some(alpha) => self.cn_update_min_sum(syndrome, alpha),
            }
            self.total_llrs();

            e_hat = self.total_llrs.mapv(|v| (v < 0.0) as u8);
            let computed_syndrome: Array1<u8> = (&self.H * &e_hat).mapv(|v: u8| v % 2);
            if computed_syndrome == *syndrome {
                break;
            }
        }

        return e_hat;
    }

    fn set_cn_to_vn_msgs(&mut self, msgs: CsMat<f64>) {
        let msgs = if msgs.is_csc() { msgs } else { msgs.to_csc() };
        self.cn_to_vn_msgs = msgs;
        self.cn_to_vn_msgs_initialized_manually = true;
    }

    fn get_cn_to_vn_msgs(&self) -> CsMat<f64> {
        return self.cn_to_vn_msgs.clone();
    }
}

// ---------------------------------------------------------------------------
// PyO3 Python Bindings
// ---------------------------------------------------------------------------

/// Helper: extract CSC components from a scipy.sparse matrix (calls .tocsc()
/// on the Python side) and build a sprs::CsMat<u8>.
#[allow(non_snake_case)]
fn scipy_sparse_to_csmat_u8(h: &Bound<'_, PyAny>) -> PyResult<CsMat<u8>> {
    let h_csc = h.call_method0("tocsc")?;

    let shape: (usize, usize) = h_csc.getattr("shape")?.extract()?;
    let indices: Vec<usize> = h_csc
        .getattr("indices")?
        .call_method0("tolist")?
        .extract()?;
    let indptr: Vec<usize> = h_csc.getattr("indptr")?.call_method0("tolist")?.extract()?;
    let data_py: Vec<i64> = h_csc.getattr("data")?.call_method0("tolist")?.extract()?;

    let data: Vec<u8> = data_py.iter().map(|&v| v as u8).collect();

    CsMat::new_csc(shape, indptr, indices, data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid sparse matrix: {e}")))
}

/// Helper: extract CSC components from a scipy.sparse matrix and build a
/// sprs::CsMat<f64>.
fn scipy_sparse_to_csmat_f64(m: &Bound<'_, PyAny>) -> PyResult<CsMat<f64>> {
    let m_csc = m.call_method0("tocsc")?;

    let shape: (usize, usize) = m_csc.getattr("shape")?.extract()?;
    let indices: Vec<usize> = m_csc
        .getattr("indices")?
        .call_method0("tolist")?
        .extract()?;
    let indptr: Vec<usize> = m_csc.getattr("indptr")?.call_method0("tolist")?.extract()?;
    let data: Vec<f64> = m_csc.getattr("data")?.call_method0("tolist")?.extract()?;

    CsMat::new_csc(shape, indptr, indices, data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid sparse matrix: {e}")))
}

/// Helper: convert a sprs::CsMat<f64> into a scipy.sparse.csc_matrix.
fn csmat_f64_to_scipy<'py>(py: Python<'py>, m: &CsMat<f64>) -> PyResult<PyObject> {
    let scipy_sparse = py.import("scipy.sparse")?;

    let data = m.data().to_vec().into_pyarray(py);
    let indices: Vec<i64> = m.indices().iter().map(|&i| i as i64).collect();
    let indices = indices.into_pyarray(py);
    let indptr: Vec<i64> = m.indptr().raw_storage().iter().map(|&i| i as i64).collect();
    let indptr = indptr.into_pyarray(py);

    let shape = PyTuple::new(py, &[m.rows(), m.cols()])?;
    let args_inner = PyTuple::new(py, &[data.as_any(), indices.as_any(), indptr.as_any()])?;
    let args = PyTuple::new(py, &[args_inner.as_any(), shape.as_any()])?;

    let csc = scipy_sparse.call_method1("csc_matrix", args)?;
    Ok(csc.into())
}

/// Python-facing BP decoder.
///
/// Example usage from Python:
/// ```python
/// import numpy as np
/// from scipy.sparse import csc_matrix
/// from bp_decoder import PyBpDecoder
///
/// H = csc_matrix(np.array([
///     [1, 0, 0, 1, 1, 0, 1],
///     [0, 1, 0, 0, 1, 1, 1],
///     [0, 0, 1, 1, 0, 1, 1],
/// ], dtype=np.uint8))
///
/// decoder = PyBpDecoder(
///     h=H,
///     channel_probs=[0.1] * 7,
///     max_iter=30,
///     bp_method="product_sum",
/// )
///
/// syndrome = np.array([1, 0, 1], dtype=np.uint8)
/// error_estimate = decoder.decode(syndrome)
/// ```
#[pyclass]
#[pyo3(name = "BpDecoder")]
struct PyBpDecoder {
    inner: BpDecoder,
}

#[pymethods]
impl PyBpDecoder {
    /// Create a new BpDecoder.
    ///
    /// Parameters
    /// ----------
    /// h : scipy.sparse matrix
    ///     The parity-check matrix H. Will be converted to CSC internally.
    /// channel_probs : list[float]
    ///     Per-bit channel error probabilities (length must equal H.cols()).
    /// max_iter : int
    ///     Maximum number of BP iterations.
    /// bp_method : str, optional
    ///     Either "product_sum" (default) or "min_sum".
    /// alpha : float, optional
    ///     Scaling factor for min-sum (only used when bp_method="min_sum").
    ///     Defaults to 1.0.
    #[new]
    #[pyo3(signature = (h, channel_probs, max_iter, bp_method="product_sum", alpha=1.0))]
    fn new(
        h: &Bound<'_, PyAny>,
        channel_probs: Vec<f64>,
        max_iter: usize,
        bp_method: &str,
        alpha: f64,
    ) -> PyResult<Self> {
        let h_mat = scipy_sparse_to_csmat_u8(h)?;

        let method = match bp_method {
            "min_sum" => BpMethod::MinSum { alpha },
            "product_sum" => BpMethod::ProductSum,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown bp_method '{other}'. Expected 'product_sum' or 'min_sum'."
                )));
            }
        };

        Ok(PyBpDecoder {
            inner: BpDecoder::new(h_mat, &channel_probs, max_iter, method),
        })
    }

    /// Run BP decoding for the given syndrome.
    ///
    /// Parameters
    /// ----------
    /// syndrome : numpy.ndarray[uint8]
    ///     The syndrome vector (length must equal H.rows()).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray[uint8]
    ///     The estimated error vector.
    fn decode<'py>(
        &mut self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let syndrome = syndrome.as_array().to_owned();
        let result = self.inner.decode(&syndrome);
        Ok(result.into_pyarray(py))
    }

    /// Set the check-node to variable-node messages manually.
    ///
    /// This allows warm-starting BP from externally provided messages.
    /// The next call to `decode` will use these messages instead of
    /// resetting to zero.
    ///
    /// Parameters
    /// ----------
    /// msgs : scipy.sparse matrix (float64)
    ///     Must have the same sparsity pattern as H.
    fn set_cn_to_vn_msgs(&mut self, msgs: &Bound<'_, PyAny>) -> PyResult<()> {
        let m = scipy_sparse_to_csmat_f64(msgs)?;
        self.inner.set_cn_to_vn_msgs(m);
        Ok(())
    }

    /// Get the current check-node to variable-node messages.
    ///
    /// Returns
    /// -------
    /// scipy.sparse.csc_matrix
    ///     The CN-to-VN message matrix.
    fn get_cn_to_vn_msgs(&self, py: Python) -> PyResult<PyObject> {
        let msgs = self.inner.get_cn_to_vn_msgs();
        csmat_f64_to_scipy(py, &msgs)
    }
}

/// Python module definition.
#[pymodule]
fn bp_decoder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBpDecoder>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests (pure Rust, unchanged)
// ---------------------------------------------------------------------------

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use std::f32::consts::E;

    use ndarray::ArrayView2;
    use ndarray::arr2;

    use super::*;

    fn assert_sparse_approx_eq(a: &CsMat<f64>, b: &CsMat<f64>, tol: f64) {
        assert_eq!(a.indptr().raw_storage(), b.indptr().raw_storage());
        assert_eq!(a.indices(), b.indices());
        for (x, y) in a.data().iter().zip(b.data().iter()) {
            assert!((x - y).abs() < tol, "{x} != {y} (tol={tol})");
        }
    }

    fn dense_to_sparse_u8(dense: ArrayView2<i8>) -> CsMat<u8> {
        CsMat::<i8>::csc_from_dense(dense, 0).map(|&v| v as u8)
    }

    fn dense_to_sparse_f64(dense: ArrayView2<f64>) -> CsMat<f64> {
        CsMat::csc_from_dense(dense, 1e-5)
    }

    fn hamming_h() -> CsMat<u8> {
        dense_to_sparse_u8(
            arr2(&[
                [1, 0, 0, 1, 1, 0, 1],
                [0, 1, 0, 0, 1, 1, 1],
                [0, 0, 1, 1, 0, 1, 1],
            ])
            .view(),
        )
    }

    #[test]
    fn test_vn_update() {
        let mut decoder = BpDecoder::new(
            hamming_h(),
            &vec![1.0 / (1.0 + E as f64); 7],
            30,
            BpMethod::ProductSum,
        );

        decoder.cn_to_vn_msgs = decoder.cn_to_vn_msgs.map(|_| 1.0);
        decoder.vn_update();

        let expected = dense_to_sparse_f64(
            arr2(&[
                [1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 3.0],
                [0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 3.0],
                [0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 3.0],
            ])
            .view(),
        );

        assert_sparse_approx_eq(&decoder.vn_to_cn_msgs, &expected, 1e-5);
    }

    #[test]
    fn test_spa_cn_update() {
        #[allow(non_snake_case)]
        let H = hamming_h().map(|&v| v as u8);
        let mut decoder = BpDecoder::new(H, &vec![0.5; 7], 30, BpMethod::ProductSum);

        decoder.vn_to_cn_msgs = decoder.vn_to_cn_msgs.map(|_| 1.0);
        decoder.cn_to_vn_msgs = decoder.cn_to_vn_msgs.map(|_| 0.0);

        let syndrome = Array1::from_vec(vec![0u8, 1, 0]);
        decoder.cn_update_spa(&syndrome);

        let expected = dense_to_sparse_f64(
            arr2(&[
                [0.198017, 0.0, 0.0, 0.198017, 0.198017, 0.0, 0.198017],
                [0.0, -0.198017, 0.0, 0.0, -0.198017, -0.198017, -0.198017],
                [0.0, 0.0, 0.198017, 0.198017, 0.0, 0.198017, 0.198017],
            ])
            .view(),
        );

        assert_sparse_approx_eq(&decoder.cn_to_vn_msgs, &expected, 1e-5);
    }

    #[test]
    fn test_min_sum_cn_update() {
        #[allow(non_snake_case)]
        let H = hamming_h().map(|&v| v as u8);
        let mut decoder = BpDecoder::new(H, &vec![0.5; 7], 30, BpMethod::MinSum { alpha: 1.0 });

        decoder.vn_to_cn_msgs = decoder.vn_to_cn_msgs.map(|_| 1.0);
        decoder.cn_to_vn_msgs = decoder.cn_to_vn_msgs.map(|_| 0.0);

        let syndrome = Array1::from_vec(vec![0u8, 1, 0]);
        decoder.cn_update_min_sum(&syndrome, 1.0);

        let expected = dense_to_sparse_f64(
            arr2(&[
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            ])
            .view(),
        );

        assert_sparse_approx_eq(&decoder.cn_to_vn_msgs, &expected, 1e-5);
    }

    fn assert_vec_approx_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < tol, "{x} != {y} (tol={tol})");
        }
    }

    #[test]
    fn test_total_llrs() {
        #[allow(non_snake_case)]
        let H = hamming_h().map(|&v| v as u8);
        let channel_probs = vec![1.0 / (1.0 + E as f64); 7];
        let mut decoder = BpDecoder::new(H, &channel_probs, 30, BpMethod::ProductSum);

        decoder.cn_to_vn_msgs = decoder.cn_to_vn_msgs.map(|_| 1.0);
        decoder.total_llrs();

        let expected = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        assert_vec_approx_eq(decoder.total_llrs.as_slice().unwrap(), &expected, 1e5);
    }
}
