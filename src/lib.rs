use pyo3::prelude::*;

pub mod bp;
pub mod bp_core;
pub mod bpgd;

mod python;

pub trait Decoder {
    fn decode(&mut self, s: &[u8]) -> Vec<u8>;
}

pub trait SoftInitDecoder {
    fn init_soft_info(s: &[f64]);
    fn decode(s: &[f64]) -> Vec<f64>;
}

#[pymodule]
fn rust_qldpc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
