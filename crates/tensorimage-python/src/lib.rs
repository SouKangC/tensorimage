use pyo3::prelude::*;

mod hash;
mod load;

#[pymodule]
fn _tensorimage(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load::load, m)?)?;
    m.add_function(wrap_pyfunction!(load::load_batch, m)?)?;
    m.add_function(wrap_pyfunction!(load::load_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(load::load_batch_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(load::_resize_array, m)?)?;
    m.add_function(wrap_pyfunction!(load::_to_tensor_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(load::_load_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(hash::compute_phash, m)?)?;
    m.add_function(wrap_pyfunction!(hash::phash_array, m)?)?;
    m.add_function(wrap_pyfunction!(hash::phash_batch, m)?)?;
    m.add_function(wrap_pyfunction!(hash::hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(hash::deduplicate, m)?)?;
    m.add_function(wrap_pyfunction!(hash::image_info, m)?)?;
    m.add_function(wrap_pyfunction!(hash::image_info_batch, m)?)?;
    Ok(())
}
