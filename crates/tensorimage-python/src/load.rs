use std::path::Path;

use numpy::{PyArray1, PyArrayMethods};
use numpy::ndarray::Ix3;
use pyo3::prelude::*;

use tensorimage_core::decode;
use tensorimage_core::error::TensorImageError;
use tensorimage_core::resize::{Algorithm, resize_shortest_edge};

#[pyfunction]
#[pyo3(signature = (path, size=None, algorithm=None))]
pub fn load<'py>(
    py: Python<'py>,
    path: &str,
    size: Option<u32>,
    algorithm: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray<u8, Ix3>>> {
    let algo = match algorithm {
        Some(name) => Algorithm::from_str(name).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        })?,
        None => Algorithm::Lanczos3,
    };

    let path_owned = path.to_string();
    let result = py.detach(|| {
        let decoded = decode::decode_file(Path::new(&path_owned), size)?;
        match size {
            Some(s) => resize_shortest_edge(decoded, s, algo),
            None => Ok(decoded),
        }
    });

    let image = result.map_err(|e: TensorImageError| {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    })?;

    let h = image.height as usize;
    let w = image.width as usize;
    let c = image.channels as usize;

    let flat = PyArray1::from_vec(py, image.data);
    let reshaped = flat
        .reshape_with_order([h, w, c], numpy::npyffi::NPY_ORDER::NPY_CORDER)
        .map_err(|e: PyErr| e)?;

    Ok(reshaped)
}
