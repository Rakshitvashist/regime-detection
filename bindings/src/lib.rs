use pyo3::prelude::*;

#[pyfunction]
fn ping() -> PyResult<String> {
    Ok("Rust engine connected 🚀".to_string())
}

#[pymodule]
fn regime_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    Ok(())
}
