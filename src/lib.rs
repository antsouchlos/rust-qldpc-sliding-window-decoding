use pyo3::prelude::*;

pub mod decoders;
pub mod windowing;

mod python;

#[pymodule]
fn rust_qldpc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
