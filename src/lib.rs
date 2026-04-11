use pyo3::prelude::*;

pub mod decoders;
pub mod soft_init;

mod python;

#[pymodule]
fn rust_qldpc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
