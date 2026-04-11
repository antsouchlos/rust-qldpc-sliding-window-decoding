use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sprs::CsMat;

use crate::Decoder;
use crate::bp::{BpMethod, Settings, SyndromeBpDecoder};

#[pyclass(name = "SyndromeBpDecoder")]
pub struct PyBpDecoder {
    decoder: SyndromeBpDecoder,
}

#[pymethods]
impl PyBpDecoder {
    #[new]
    #[allow(non_snake_case)]
    pub fn new(
        H_csr_indptr: Vec<i32>,
        H_csr_indices: Vec<i32>,
        H_csr_data: Vec<u8>,
        H_shape: (usize, usize),
        channel_llrs: Vec<f64>,
        max_iter: usize,
        method: &str,
    ) -> PyResult<Self> {
        if H_csr_indptr.len() != H_shape.0 + 1 {
            return Err(PyValueError::new_err(
                "indptr.len() must equal nrows + 1",
            ));
        }
        if H_csr_indices.len() != H_csr_data.len() {
            return Err(PyValueError::new_err(
                "indices and data must have the same length",
            ));
        }
        if channel_llrs.len() != H_shape.1 {
            return Err(PyValueError::new_err(
                "channel_llrs.len() must equal ncols",
            ));
        }

        let bp_method = match method {
            "spa" => BpMethod::Spa,
            "min_sum" => BpMethod::MinSum,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown method {other:?}; expected \"spa\" or \"min_sum\""
                )));
            }
        };

        // TODO: Do the indptr and indices arrays have to be i32 arrays to begin with?
        let indptr_u: Vec<usize> =
            H_csr_indptr.iter().map(|&v| v as usize).collect();
        let indices_u: Vec<usize> =
            H_csr_indices.iter().map(|&v| v as usize).collect();
        let h_csr = CsMat::new(H_shape, indptr_u, indices_u, H_csr_data);

        Ok(Self {
            decoder: SyndromeBpDecoder::new(
                Settings {
                    max_iter: max_iter,
                    bp_method: bp_method,
                },
                &h_csr,
                &channel_llrs,
            ),
        })
    }

    pub fn decode<'py>(
        &mut self,
        py: Python<'py>,
        syndrome: Vec<u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let result = self.decoder.decode(&syndrome);
        Ok(PyArray1::from_vec(py, result))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBpDecoder>()?;
    Ok(())
}
