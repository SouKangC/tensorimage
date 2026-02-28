use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

use tensorimage_core::augment;
use tensorimage_core::error::TensorImageError;

fn to_pyerr(e: TensorImageError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Apply Gaussian blur to a numpy uint8 HWC array.
#[pyfunction]
#[pyo3(signature = (array, kernel_size, sigma))]
pub fn _gaussian_blur<'py>(
    py: Python<'py>,
    array: PyReadonlyArray3<'_, u8>,
    kernel_size: u32,
    sigma: f64,
) -> PyResult<Py<PyAny>> {
    let shape = array.shape();
    let h = shape[0] as u32;
    let w = shape[1] as u32;
    let c = shape[2] as u32;

    let slice = array
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;

    let data = slice.to_vec();
    let result = py.detach(|| augment::gaussian_blur(&data, w, h, c, kernel_size, sigma));
    let output = result.map_err(to_pyerr)?;

    let flat = PyArray1::from_vec(py, output);
    let reshaped = flat
        .reshape_with_order(
            [h as usize, w as usize, c as usize],
            numpy::npyffi::NPY_ORDER::NPY_CORDER,
        )?;
    Ok(reshaped.into_any().unbind())
}

/// Apply affine transformation to a numpy uint8 HWC array.
#[pyfunction]
#[pyo3(signature = (array, matrix, out_h, out_w, fill))]
pub fn _affine_transform<'py>(
    py: Python<'py>,
    array: PyReadonlyArray3<'_, u8>,
    matrix: Vec<f64>,
    out_h: u32,
    out_w: u32,
    fill: Vec<u8>,
) -> PyResult<Py<PyAny>> {
    let shape = array.shape();
    let h = shape[0] as u32;
    let w = shape[1] as u32;
    let c = shape[2] as u32;

    if matrix.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matrix must have exactly 6 elements [a, b, tx, c, d, ty]",
        ));
    }

    let mat: [f64; 6] = [matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5]];

    let slice = array
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;

    let data = slice.to_vec();
    let result = py.detach(|| {
        augment::affine_transform(&data, w, h, c, &mat, out_w, out_h, &fill)
    });
    let output = result.map_err(to_pyerr)?;

    let flat = PyArray1::from_vec(py, output);
    let reshaped = flat
        .reshape_with_order(
            [out_h as usize, out_w as usize, c as usize],
            numpy::npyffi::NPY_ORDER::NPY_CORDER,
        )?;
    Ok(reshaped.into_any().unbind())
}

/// Apply perspective transformation to a numpy uint8 HWC array.
#[pyfunction]
#[pyo3(signature = (array, coeffs, out_h, out_w, fill))]
pub fn _perspective_transform<'py>(
    py: Python<'py>,
    array: PyReadonlyArray3<'_, u8>,
    coeffs: Vec<f64>,
    out_h: u32,
    out_w: u32,
    fill: Vec<u8>,
) -> PyResult<Py<PyAny>> {
    let shape = array.shape();
    let h = shape[0] as u32;
    let w = shape[1] as u32;
    let c = shape[2] as u32;

    if coeffs.len() != 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coeffs must have exactly 8 elements",
        ));
    }

    let cf: [f64; 8] = [
        coeffs[0], coeffs[1], coeffs[2], coeffs[3],
        coeffs[4], coeffs[5], coeffs[6], coeffs[7],
    ];

    let slice = array
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;

    let data = slice.to_vec();
    let result = py.detach(|| {
        augment::perspective_transform(&data, w, h, c, &cf, out_w, out_h, &fill)
    });
    let output = result.map_err(to_pyerr)?;

    let flat = PyArray1::from_vec(py, output);
    let reshaped = flat
        .reshape_with_order(
            [out_h as usize, out_w as usize, c as usize],
            numpy::npyffi::NPY_ORDER::NPY_CORDER,
        )?;
    Ok(reshaped.into_any().unbind())
}
