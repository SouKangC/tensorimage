use pyo3::prelude::*;

mod load;

#[pymodule]
fn _tensorimage(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load::load, m)?)?;
    Ok(())
}
