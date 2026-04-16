use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use sprs::CsMat;

use crate::decoders::Decoder;
use crate::decoders::bp::{VanillaBpDecoder, VanillaBpSettings};
use crate::decoders::engine::ParityCheckMatrix;
use crate::decoders::engine::min_sum::MinSumComputeEngine;
use crate::decoders::engine::spa::SpaComputeEngine;
use crate::decoders::meta::sliding_window::{
    SlidingWindowDecoder, SlidingWindowSettings,
};

fn extract_parity_check_matrix(h: &Bound<'_, PyAny>) -> PyResult<CsMat<u8>> {
    let h_csr_indptr: PyReadonlyArray1<usize> =
        h.getattr("indptr")?.extract()?;
    let h_csr_indices: PyReadonlyArray1<usize> =
        h.getattr("indices")?.extract()?;
    let h_csr_data: PyReadonlyArray1<u8> = h.getattr("data")?.extract()?;
    let h_shape: (usize, usize) = h.getattr("shape")?.extract()?;

    let h_csr_indptr = h_csr_indptr.as_slice()?;
    let h_csr_indices = h_csr_indices.as_slice()?;
    let h_csr_data = h_csr_data.as_slice()?;

    if h_csr_indptr.len() != h_shape.0 + 1 {
        return Err(PyValueError::new_err("indptr.len() must equal nrows + 1"));
    }
    if h_csr_indices.len() != h_csr_data.len() {
        return Err(PyValueError::new_err(
            "indices and data must have the same length",
        ));
    }

    let h_csr = CsMat::new(
        h_shape,
        h_csr_indptr.to_vec(),
        h_csr_indices.to_vec(),
        h_csr_data.to_vec(),
    );

    Ok(h_csr)
}

#[pyclass(name = "VanillaMinSumDecoder")]
pub struct PyVanillaMinSumDecoder {
    decoder: VanillaBpDecoder<MinSumComputeEngine>,
}

#[pymethods]
impl PyVanillaMinSumDecoder {
    #[new]
    #[allow(non_snake_case)]
    pub fn new(
        H: &Bound<'_, PyAny>,
        priors: Vec<f64>,
        max_iter: usize,
    ) -> PyResult<Self> {
        let h = extract_parity_check_matrix(H)?;

        if priors.len() != h.cols() {
            return Err(PyValueError::new_err(
                "channel_llrs.len() must equal ncols",
            ));
        }

        let channel_llrs = priors
            .into_iter()
            .map(|p| (1.0 - p).ln() - p.ln())
            .collect::<Vec<_>>();

        Ok(Self {
            decoder: VanillaBpDecoder::new(
                VanillaBpSettings { max_iter: max_iter },
                &ParityCheckMatrix::new(&h),
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
        Ok(PyArray1::from_vec(py, result.to_vec()))
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
                .map(|s| template.clone().decode(s).to_vec())
                .collect()
        });

        PyArray2::from_vec2(py, &results)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "VanillaSpaDecoder")]
pub struct PyVanillaSpaDecoder {
    decoder: VanillaBpDecoder<SpaComputeEngine>,
}

#[pymethods]
impl PyVanillaSpaDecoder {
    #[new]
    #[allow(non_snake_case)]
    pub fn new(
        H: &Bound<'_, PyAny>,
        priors: Vec<f64>,
        max_iter: usize,
    ) -> PyResult<Self> {
        let h = extract_parity_check_matrix(H)?;

        if priors.len() != h.cols() {
            return Err(PyValueError::new_err(
                "channel_llrs.len() must equal ncols",
            ));
        }

        let channel_llrs = priors
            .into_iter()
            .map(|p| (1.0 - p).ln() - p.ln())
            .collect::<Vec<_>>();

        if channel_llrs.len() != h.cols() {
            return Err(PyValueError::new_err(
                "channel_llrs.len() must equal ncols",
            ));
        }

        Ok(Self {
            decoder: VanillaBpDecoder::new(
                VanillaBpSettings { max_iter: max_iter },
                &ParityCheckMatrix::new(&h),
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
        Ok(PyArray1::from_vec(py, result.to_vec()))
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
                .map(|s| template.clone().decode(s).to_vec())
                .collect()
        });

        PyArray2::from_vec2(py, &results)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "SlidingWindowMinSumDecoder")]
pub struct PySlidingWindowMinSumDecoder {
    decoder: SlidingWindowDecoder<VanillaBpDecoder<MinSumComputeEngine>>,
}

#[pymethods]
impl PySlidingWindowMinSumDecoder {
    #[new]
    #[allow(non_snake_case)]
    pub fn new(
        H: &Bound<'_, PyAny>,
        m: usize,
        num_rounds: usize,
        W: usize,
        F: usize,
        priors: Vec<f64>,
        max_iter: usize,
        warm_start: bool,
    ) -> PyResult<Self> {
        let h = extract_parity_check_matrix(H)?;

        if priors.len() != h.cols() {
            return Err(PyValueError::new_err(
                "channel_llrs.len() must equal ncols",
            ));
        }

        let channel_llrs = priors
            .into_iter()
            .map(|p| (1.0 - p).ln() - p.ln())
            .collect::<Vec<_>>();

        if channel_llrs.len() != h.cols() {
            return Err(PyValueError::new_err(
                "channel_llrs.len() must equal ncols",
            ));
        }

        Ok(Self {
            decoder: SlidingWindowDecoder::new(
                SlidingWindowSettings { warm_start, W, F },
                VanillaBpSettings { max_iter },
                &h,
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
        Ok(PyArray1::from_vec(py, result.to_vec()))
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
                .map(|s| template.clone().decode(s).to_vec())
                .collect()
        });

        PyArray2::from_vec2(py, &results)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "SlidingWindowSpaDecoder")]
pub struct PySlidingWindowSpaDecoder {
    decoder: SlidingWindowDecoder<VanillaBpDecoder<SpaComputeEngine>>,
}

#[pymethods]
impl PySlidingWindowSpaDecoder {
    #[new]
    #[allow(non_snake_case)]
    pub fn new(
        H: &Bound<'_, PyAny>,
        m: usize,
        num_rounds: usize,
        W: usize,
        F: usize,
        priors: Vec<f64>,
        max_iter: usize,
        warm_start: bool,
    ) -> PyResult<Self> {
        let h = extract_parity_check_matrix(H)?;

        if priors.len() != h.cols() {
            return Err(PyValueError::new_err(
                "channel_llrs.len() must equal ncols",
            ));
        }

        let channel_llrs = priors
            .into_iter()
            .map(|p| (1.0 - p).ln() - p.ln())
            .collect::<Vec<_>>();

        if channel_llrs.len() != h.cols() {
            return Err(PyValueError::new_err(
                "channel_llrs.len() must equal ncols",
            ));
        }

        Ok(Self {
            decoder: SlidingWindowDecoder::new(
                SlidingWindowSettings { warm_start, W, F },
                VanillaBpSettings { max_iter },
                &h,
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
        Ok(PyArray1::from_vec(py, result.to_vec()))
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
                .map(|s| template.clone().decode(s).to_vec())
                .collect()
        });

        PyArray2::from_vec2(py, &results)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVanillaMinSumDecoder>()?;
    m.add_class::<PyVanillaSpaDecoder>()?;
    m.add_class::<PySlidingWindowMinSumDecoder>()?;
    m.add_class::<PySlidingWindowSpaDecoder>()?;
    Ok(())
}
