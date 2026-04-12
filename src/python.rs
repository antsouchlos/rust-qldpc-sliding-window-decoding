use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use sprs::CsMat;

use crate::decoders::Decoder;
use crate::decoders::bp::{Settings, SyndromeBpDecoder};
use crate::decoders::core::{
    SyndromeBpStrategy, min_sum::SyndromeMinSumCore, spa::SyndromeSpaCore,
};
use crate::soft_init::{self, SoftInitBp};

#[pyclass(name = "SyndromeMinSumDecoder")]
pub struct PySyndromeMinSumDecoder {
    decoder: SyndromeBpDecoder<SyndromeMinSumCore>,
}

#[pymethods]
impl PySyndromeMinSumDecoder {
    #[new]
    #[allow(non_snake_case)]
    pub fn new(
        H_csr_indptr: Vec<usize>,
        H_csr_indices: Vec<usize>,
        H_csr_data: Vec<u8>,
        H_shape: (usize, usize),
        channel_llrs: Vec<f64>,
        max_iter: usize,
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

        let h_csr =
            CsMat::new(H_shape, H_csr_indptr, H_csr_indices, H_csr_data);

        Ok(Self {
            decoder: SyndromeBpDecoder::new(
                Settings { max_iter: max_iter },
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

    pub fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        syndromes: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let syndromes_array = syndromes.as_array();
        let num_syndromes = syndromes_array.shape()[0];

        let syndromes_vec: Vec<Vec<u8>> = (0..num_syndromes)
            .map(|i| syndromes_array.row(i).to_vec())
            .collect();

        let mut template = self.decoder.clone();
        for edge in &mut template.core.get_state().edges {
            edge.msg_vn_to_cn = 0.0;
            edge.msg_cn_to_vn = 0.0;
        }

        let results: Vec<Vec<u8>> = py.detach(|| {
            syndromes_vec
                .par_iter()
                .map(|s| template.clone().decode(s))
                .collect()
        });

        PyArray2::from_vec2(py, &results)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "SyndromeSpaDecoder")]
pub struct PySyndromeSpaDecoder {
    decoder: SyndromeBpDecoder<SyndromeSpaCore>,
}

#[pymethods]
impl PySyndromeSpaDecoder {
    #[new]
    #[allow(non_snake_case)]
    pub fn new(
        H_csr_indptr: Vec<usize>,
        H_csr_indices: Vec<usize>,
        H_csr_data: Vec<u8>,
        H_shape: (usize, usize),
        channel_llrs: Vec<f64>,
        max_iter: usize,
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

        let h_csr =
            CsMat::new(H_shape, H_csr_indptr, H_csr_indices, H_csr_data);

        Ok(Self {
            decoder: SyndromeBpDecoder::new(
                Settings { max_iter: max_iter },
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

    pub fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        syndromes: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let syndromes_array = syndromes.as_array();
        let num_syndromes = syndromes_array.shape()[0];

        let syndromes_vec: Vec<Vec<u8>> = (0..num_syndromes)
            .map(|i| syndromes_array.row(i).to_vec())
            .collect();

        let mut template = self.decoder.clone();
        template.reset();

        let results: Vec<Vec<u8>> = py.detach(|| {
            syndromes_vec
                .par_iter()
                .map(|s| template.clone().decode(s))
                .collect()
        });

        PyArray2::from_vec2(py, &results)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "WindowedSyndromeSpaDecoder")]
pub struct PyWindowedSyndromeSpaDecoder {
    decoder: SoftInitBp<SyndromeBpDecoder<SyndromeSpaCore>>,
}

#[pymethods]
impl PyWindowedSyndromeSpaDecoder {
    #[new]
    #[allow(non_snake_case)]
    pub fn new(
        H_csr_indptr: Vec<usize>,
        H_csr_indices: Vec<usize>,
        H_csr_data: Vec<u8>,
        H_shape: (usize, usize),
        m: usize,
        num_rounds: usize,
        channel_llrs: Vec<f64>,
        W: usize,
        F: usize,
        max_iter: usize,
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

        let h_csr =
            CsMat::new(H_shape, H_csr_indptr, H_csr_indices, H_csr_data);

        Ok(Self {
            decoder: SoftInitBp::new(
                soft_init::Settings {
                    max_iter: max_iter,
                    W,
                    F,
                },
                &h_csr,
                m,
                num_rounds,
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

    pub fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        syndromes: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let syndromes_array = syndromes.as_array();
        let num_syndromes = syndromes_array.shape()[0];

        let syndromes_vec: Vec<Vec<u8>> = (0..num_syndromes)
            .map(|i| syndromes_array.row(i).to_vec())
            .collect();

        let mut template = self.decoder.clone();
        template.reset();

        let results: Vec<Vec<u8>> = py.detach(|| {
            syndromes_vec
                .par_iter()
                .map(|s| template.clone().decode(s))
                .collect()
        });

        PyArray2::from_vec2(py, &results)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "WindowedSyndromeMinSumDecoder")]
pub struct PyWindowedSyndromeMinSumDecoder {
    decoder: SoftInitBp<SyndromeBpDecoder<SyndromeMinSumCore>>,
}

#[pymethods]
impl PyWindowedSyndromeMinSumDecoder {
    #[new]
    #[allow(non_snake_case)]
    pub fn new(
        H_csr_indptr: Vec<usize>,
        H_csr_indices: Vec<usize>,
        H_csr_data: Vec<u8>,
        H_shape: (usize, usize),
        m: usize,
        num_rounds: usize,
        channel_llrs: Vec<f64>,
        W: usize,
        F: usize,
        max_iter: usize,
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

        let h_csr =
            CsMat::new(H_shape, H_csr_indptr, H_csr_indices, H_csr_data);

        Ok(Self {
            decoder: SoftInitBp::new(
                soft_init::Settings {
                    max_iter: max_iter,
                    W,
                    F,
                },
                &h_csr,
                m,
                num_rounds,
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

    pub fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        syndromes: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let syndromes_array = syndromes.as_array();
        let num_syndromes = syndromes_array.shape()[0];

        let syndromes_vec: Vec<Vec<u8>> = (0..num_syndromes)
            .map(|i| syndromes_array.row(i).to_vec())
            .collect();

        let mut template = self.decoder.clone();
        template.reset();

        let results: Vec<Vec<u8>> = py.detach(|| {
            syndromes_vec
                .par_iter()
                .map(|s| template.clone().decode(s))
                .collect()
        });

        PyArray2::from_vec2(py, &results)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySyndromeMinSumDecoder>()?;
    m.add_class::<PySyndromeSpaDecoder>()?;
    m.add_class::<PyWindowedSyndromeSpaDecoder>()?;
    m.add_class::<PyWindowedSyndromeMinSumDecoder>()?;
    Ok(())
}
