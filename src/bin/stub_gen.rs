use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = rust_qldpc::stub_info()?;
    stub.generate()?;
    Ok(())
}
