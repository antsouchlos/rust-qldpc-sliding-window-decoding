use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

pub mod decoders;
pub mod windowing;

mod python;

define_stub_info_gatherer!(stub_info);

#[pymodule]
fn rust_qldpc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
