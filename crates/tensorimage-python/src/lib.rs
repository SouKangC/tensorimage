use pyo3::prelude::*;

mod load;

#[pymodule]
fn _tensorimage(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load::load, m)?)?;
    m.add_function(wrap_pyfunction!(load::load_batch, m)?)?;
    m.add_function(wrap_pyfunction!(load::_resize_array, m)?)?;
    Ok(())
}
