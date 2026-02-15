use std::path::{Path, PathBuf};

use numpy::ndarray::Ix3;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

use tensorimage_core::crop::CropMode;
use tensorimage_core::error::TensorImageError;
use tensorimage_core::normalize::{NormalizeParams, normalize_hwc_to_chw_from_slice};
use tensorimage_core::pipeline::{PipelineConfig, PipelineOutput, execute_pipeline};
use tensorimage_core::resize::{Algorithm, resize_exact_borrowed};

fn parse_crop(crop: &str, size: Option<u32>) -> Result<(CropMode, u32, u32), PyErr> {
    let mode = CropMode::from_str(crop)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let s = size.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "crop requires size parameter to determine crop dimensions",
        )
    })?;
    Ok((mode, s, s))
}

fn parse_normalize(normalize: &str) -> Result<NormalizeParams, PyErr> {
    NormalizeParams::from_preset(normalize)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

fn build_config(
    size: Option<u32>,
    algorithm: Option<&str>,
    crop: Option<&str>,
    normalize: Option<&str>,
) -> Result<PipelineConfig, PyErr> {
    let algo = match algorithm {
        Some(name) => Algorithm::from_str(name)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        None => Algorithm::Lanczos3,
    };

    let crop_config = match crop {
        Some(c) => Some(parse_crop(c, size)?),
        None => None,
    };

    let norm_config = match normalize {
        Some(n) => Some(parse_normalize(n)?),
        None => None,
    };

    Ok(PipelineConfig {
        size,
        algorithm: algo,
        crop: crop_config,
        normalize: norm_config,
    })
}

fn to_pyerr(e: TensorImageError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

#[pyfunction]
#[pyo3(signature = (path, size=None, algorithm=None, crop=None, normalize=None))]
pub fn load<'py>(
    py: Python<'py>,
    path: &str,
    size: Option<u32>,
    algorithm: Option<&str>,
    crop: Option<&str>,
    normalize: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let config = build_config(size, algorithm, crop, normalize)?;
    let path_owned = path.to_string();

    let result = py.detach(|| execute_pipeline(Path::new(&path_owned), &config));
    let output = result.map_err(to_pyerr)?;

    match output {
        PipelineOutput::U8Hwc(image) => {
            let h = image.height as usize;
            let w = image.width as usize;
            let c = image.channels as usize;
            let flat = PyArray1::from_vec(py, image.data);
            let reshaped = flat
                .reshape_with_order([h, w, c], numpy::npyffi::NPY_ORDER::NPY_CORDER)?;
            Ok(reshaped.into_any().unbind())
        }
        PipelineOutput::F32Chw { data, height, width } => {
            let h = height as usize;
            let w = width as usize;
            let flat = PyArray1::from_vec(py, data);
            let reshaped = flat
                .reshape_with_order([3, h, w], numpy::npyffi::NPY_ORDER::NPY_CORDER)?;
            Ok(reshaped.into_any().unbind())
        }
    }
}

#[pyfunction]
#[pyo3(signature = (paths, size=None, algorithm=None, crop=None, normalize=None, workers=None))]
pub fn load_batch<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    size: Option<u32>,
    algorithm: Option<&str>,
    crop: Option<&str>,
    normalize: Option<&str>,
    workers: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let config = build_config(size, algorithm, crop, normalize)?;
    let num_workers = workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();

    // Fast path: when crop + normalize are both set, we know the output dimensions
    // upfront and can write directly into a single contiguous buffer.
    if let (Some((_, cw, ch)), Some(_)) = (&config.crop, &config.normalize) {
        let (crop_h, crop_w) = (*ch, *cw);
        let n = paths.len();
        let h = crop_h as usize;
        let w = crop_w as usize;

        let batch_data = py.detach(|| {
            tensorimage_core::batch::load_batch_contiguous(
                &path_bufs, &config, num_workers, crop_h, crop_w,
            )
        });
        let data = batch_data.map_err(to_pyerr)?;

        let flat = PyArray1::from_vec(py, data);
        let reshaped = flat.reshape_with_order(
            [n, 3, h, w],
            numpy::npyffi::NPY_ORDER::NPY_CORDER,
        )?;
        return Ok(reshaped.into_any().unbind());
    }

    let results = py.detach(|| {
        tensorimage_core::batch::load_batch(&path_bufs, &config, num_workers)
    });
    let outputs = results.map_err(to_pyerr)?;

    // Check if all outputs are F32Chw with the same dimensions → stack into [N,3,H,W]
    if let Some(PipelineOutput::F32Chw { height, width, .. }) = outputs.first() {
        let h = *height;
        let w = *width;
        let all_same = outputs.iter().all(|o| matches!(o, PipelineOutput::F32Chw { height: oh, width: ow, .. } if *oh == h && *ow == w));

        if all_same {
            return build_f32_batch(py, outputs, h, w);
        }
    }

    // Otherwise return a list of individual arrays
    build_output_list(py, outputs)
}

fn build_f32_batch<'py>(
    py: Python<'py>,
    outputs: Vec<PipelineOutput>,
    height: u32,
    width: u32,
) -> PyResult<Py<PyAny>> {
    let n = outputs.len();
    let h = height as usize;
    let w = width as usize;
    let plane_size = 3 * h * w;
    let mut all_data = Vec::with_capacity(n * plane_size);

    for output in outputs {
        if let PipelineOutput::F32Chw { data, .. } = output {
            all_data.extend_from_slice(&data);
        }
    }

    let flat = PyArray1::from_vec(py, all_data);
    let reshaped = flat.reshape_with_order(
        [n, 3, h, w],
        numpy::npyffi::NPY_ORDER::NPY_CORDER,
    )?;
    Ok(reshaped.into_any().unbind())
}

fn build_output_list<'py>(
    py: Python<'py>,
    outputs: Vec<PipelineOutput>,
) -> PyResult<Py<PyAny>> {
    let list = pyo3::types::PyList::empty(py);
    for output in outputs {
        match output {
            PipelineOutput::U8Hwc(image) => {
                let h = image.height as usize;
                let w = image.width as usize;
                let c = image.channels as usize;
                let flat = PyArray1::from_vec(py, image.data);
                let reshaped: Bound<'_, numpy::PyArray<u8, Ix3>> = flat
                    .reshape_with_order([h, w, c], numpy::npyffi::NPY_ORDER::NPY_CORDER)?;
                list.append(reshaped)?;
            }
            PipelineOutput::F32Chw { data, height, width } => {
                let h = height as usize;
                let w = width as usize;
                let flat = PyArray1::from_vec(py, data);
                let reshaped = flat
                    .reshape_with_order([3, h, w], numpy::npyffi::NPY_ORDER::NPY_CORDER)?;
                list.append(reshaped)?;
            }
        }
    }
    Ok(list.into_any().unbind())
}

/// Fused ToTensor + Normalize: u8 HWC → f32 CHW in a single pass.
/// Uses slice-based normalize to avoid copying the numpy array data.
#[pyfunction]
#[pyo3(signature = (array, mean, std))]
pub fn _to_tensor_normalize<'py>(
    py: Python<'py>,
    array: PyReadonlyArray3<'py, u8>,
    mean: [f32; 3],
    std: [f32; 3],
) -> PyResult<Py<PyAny>> {
    let shape = array.shape();
    let h = shape[0] as u32;
    let w = shape[1] as u32;

    let params = NormalizeParams::custom(mean, std).map_err(to_pyerr)?;

    let slice = array
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;

    // Use slice directly — no .to_vec() copy needed for read-only normalize
    let result = normalize_hwc_to_chw_from_slice(slice, w, h, &params);

    let flat = PyArray1::from_vec(py, result);
    let reshaped =
        flat.reshape_with_order([3, h as usize, w as usize], numpy::npyffi::NPY_ORDER::NPY_CORDER)?;
    Ok(reshaped.into_any().unbind())
}

/// Full Rust pipeline: decode → IDCT-scaled resize → center crop → fused normalize+transpose.
/// Used by Compose fast-path for Resize(int) → CenterCrop → ToTensor → Normalize patterns.
#[pyfunction]
#[pyo3(signature = (path, size, crop_size, mean, std))]
pub fn _load_pipeline<'py>(
    py: Python<'py>,
    path: &str,
    size: u32,
    crop_size: u32,
    mean: [f32; 3],
    std: [f32; 3],
) -> PyResult<Py<PyAny>> {
    let params = NormalizeParams::custom(mean, std).map_err(to_pyerr)?;
    let config = PipelineConfig {
        size: Some(size),
        algorithm: Algorithm::Lanczos3,
        crop: Some((CropMode::Center, crop_size, crop_size)),
        normalize: Some(params),
    };
    let path_owned = path.to_string();

    let result = py.detach(|| execute_pipeline(Path::new(&path_owned), &config));
    let output = result.map_err(to_pyerr)?;

    match output {
        PipelineOutput::F32Chw { data, height, width } => {
            let h = height as usize;
            let w = width as usize;
            let flat = PyArray1::from_vec(py, data);
            let reshaped =
                flat.reshape_with_order([3, h, w], numpy::npyffi::NPY_ORDER::NPY_CORDER)?;
            Ok(reshaped.into_any().unbind())
        }
        PipelineOutput::U8Hwc(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "_load_pipeline always normalizes; unexpected U8Hwc output",
        )),
    }
}

/// Resize a numpy u8 HWC array using SIMD-accelerated resize.
/// Used by the transforms module for Resize transform.
#[pyfunction]
#[pyo3(signature = (array, target_h, target_w, algorithm))]
pub fn _resize_array<'py>(
    py: Python<'py>,
    array: PyReadonlyArray3<'py, u8>,
    target_h: u32,
    target_w: u32,
    algorithm: &str,
) -> PyResult<Py<PyAny>> {
    let shape = array.shape();
    let h = shape[0] as u32;
    let w = shape[1] as u32;

    let algo = Algorithm::from_str(algorithm)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Copy numpy data into a mutable buffer for from_slice_u8
    let mut data = array.as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?
        .to_vec();

    let result = py.detach(|| resize_exact_borrowed(&mut data, w, h, target_w, target_h, algo));
    let resized = result.map_err(to_pyerr)?;

    let rh = resized.height as usize;
    let rw = resized.width as usize;
    let flat = PyArray1::from_vec(py, resized.data);
    let reshaped = flat
        .reshape_with_order([rh, rw, 3], numpy::npyffi::NPY_ORDER::NPY_CORDER)?;
    Ok(reshaped.into_any().unbind())
}
