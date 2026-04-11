use numpy::{PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sprs::CsMat;

use crate::bp::{BpMethod, Settings};
use crate::bp_core::{SyndromeBpDecoderCore, compute_syndrome};

#[pyclass(name = "SyndromeBpDecoder")]
pub struct PyBpDecoder {
    settings: Settings,
    core: SyndromeBpDecoderCore,
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
            core: SyndromeBpDecoderCore::new(&h_csr, &channel_llrs),
            settings: Settings {
                max_iter,
                bp_method,
            },
        })
    }

    pub fn decode<'py>(
        &mut self,
        py: Python<'py>,
        syndrome: Vec<u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        // let s = syndrome.as_slice()?;
        let result = self.bp_decode(&syndrome);

        Ok(PyArray1::from_vec(py, result))
    }
}

impl PyBpDecoder {
    fn bp_decode(&mut self, s: &[u8]) -> Vec<u8> {
        let mut e_hat: Vec<u8> = self
            .core
            .channel_llrs
            .iter()
            .map(|&v| if v < 0.0 { 1 } else { 0 })
            .collect();

        for _ in 0..self.settings.max_iter {
            self.core.vn_update();
            match self.settings.bp_method {
                BpMethod::Spa => self.core.cn_update_spa(s),
                BpMethod::MinSum => self.core.cn_update_min_sum(s),
            }
            self.core.total_llrs();

            e_hat = self
                .core
                .total_llrs
                .iter()
                .map(|&v| if v < 0.0 { 1 } else { 0 })
                .collect();

            if compute_syndrome(&self.core.H_csc, &e_hat) == s {
                break;
            }
        }
        e_hat
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBpDecoder>()?;
    Ok(())
}
