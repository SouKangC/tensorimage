use pyo3::prelude::*;

mod load;

#[pymodule]
fn _tensorimage(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load::load, m)?)?;
    m.add_function(wrap_pyfunction!(load::load_batch, m)?)?;
    m.add_function(wrap_pyfunction!(load::_resize_array, m)?)?;
    m.add_function(wrap_pyfunction!(load::_to_tensor_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(load::_load_pipeline, m)?)?;
    Ok(())
}
