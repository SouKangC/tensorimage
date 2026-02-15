use std::path::PathBuf;

use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use tensorimage_core::dedup;
use tensorimage_core::error::TensorImageError;
use tensorimage_core::jpeg_info;
use tensorimage_core::phash::{self, HashAlgorithm};

fn to_pyerr(e: TensorImageError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

fn default_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Compute a perceptual hash of an image file.
///
/// Args:
///     path: Path to an image file.
///     algorithm: Hash algorithm — "dhash" (default, fast) or "phash" (robust).
///
/// Returns:
///     64-bit hash as an integer.
#[pyfunction]
#[pyo3(signature = (path, algorithm="dhash"))]
pub fn compute_phash(py: Python<'_>, path: &str, algorithm: &str) -> PyResult<u64> {
    let algo = HashAlgorithm::from_str(algorithm).map_err(to_pyerr)?;
    let path_owned = path.to_string();

    let result = py.detach(|| phash::hash_file(std::path::Path::new(&path_owned), algo));
    result.map_err(to_pyerr)
}

/// Compute a perceptual hash from a numpy RGB array (H, W, 3).
///
/// Args:
///     array: numpy uint8 array with shape (H, W, 3).
///     algorithm: Hash algorithm — "dhash" or "phash".
///
/// Returns:
///     64-bit hash as an integer.
#[pyfunction]
#[pyo3(signature = (array, algorithm="dhash"))]
pub fn phash_array(
    _py: Python<'_>,
    array: PyReadonlyArray3<'_, u8>,
    algorithm: &str,
) -> PyResult<u64> {
    let algo = HashAlgorithm::from_str(algorithm).map_err(to_pyerr)?;
    let shape = array.shape();
    let h = shape[0] as u32;
    let w = shape[1] as u32;

    let slice = array
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;

    let result = match algo {
        HashAlgorithm::DHash => phash::dhash(slice, w, h),
        HashAlgorithm::PHash => phash::phash(slice, w, h),
    };
    result.map_err(to_pyerr)
}

/// Compute perceptual hashes for a batch of image files in parallel.
///
/// Args:
///     paths: List of image file paths.
///     algorithm: Hash algorithm — "dhash" or "phash".
///     workers: Number of worker threads (default: CPU count).
///
/// Returns:
///     List of 64-bit hashes.
#[pyfunction]
#[pyo3(signature = (paths, algorithm="dhash", workers=None))]
pub fn phash_batch(
    py: Python<'_>,
    paths: Vec<String>,
    algorithm: &str,
    workers: Option<usize>,
) -> PyResult<Vec<u64>> {
    let algo = HashAlgorithm::from_str(algorithm).map_err(to_pyerr)?;
    let num_workers = workers.unwrap_or_else(default_workers);
    let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();

    let result = py.detach(|| {
        // Use dedup's parallel hash path (reuses shared rayon pool)
        let dedup_result =
            dedup::deduplicate_paths(&path_bufs, algo, u32::MAX, num_workers);
        dedup_result.map(|r| r.hashes)
    });
    result.map_err(to_pyerr)
}

/// Compute the Hamming distance between two 64-bit hashes.
#[pyfunction]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    phash::hamming_distance(a, b)
}

/// Deduplicate images by perceptual hash.
///
/// Args:
///     paths: List of image file paths.
///     algorithm: Hash algorithm — "dhash" or "phash".
///     threshold: Max Hamming distance to consider as duplicate.
///         Default: 0 for dhash (exact), 10 for phash.
///     workers: Number of worker threads.
///
/// Returns:
///     Dict with keys: "keep_indices", "duplicate_groups", "hashes".
#[pyfunction]
#[pyo3(signature = (paths, algorithm="dhash", threshold=None, workers=None))]
pub fn deduplicate<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    algorithm: &str,
    threshold: Option<u32>,
    workers: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let algo = HashAlgorithm::from_str(algorithm).map_err(to_pyerr)?;
    let thresh = threshold.unwrap_or(match algo {
        HashAlgorithm::DHash => 0,
        HashAlgorithm::PHash => 10,
    });
    let num_workers = workers.unwrap_or_else(default_workers);
    let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();

    let result =
        py.detach(|| dedup::deduplicate_paths(&path_bufs, algo, thresh, num_workers));
    let dedup_result = result.map_err(to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("keep_indices", &dedup_result.keep_indices)?;
    dict.set_item("duplicate_groups", &dedup_result.duplicate_groups)?;
    dict.set_item("hashes", &dedup_result.hashes)?;
    Ok(dict.into_any().unbind())
}

/// Read image dimensions for a batch of files (header-only, no decode).
///
/// Args:
///     paths: List of image file paths.
///     workers: Number of worker threads.
///
/// Returns:
///     List of (width, height) tuples. Raises on error.
#[pyfunction]
#[pyo3(signature = (paths, workers=None))]
pub fn image_info_batch(
    py: Python<'_>,
    paths: Vec<String>,
    workers: Option<usize>,
) -> PyResult<Vec<(u32, u32)>> {
    let num_workers = workers.unwrap_or_else(default_workers);
    let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();

    let results =
        py.detach(|| jpeg_info::image_info_batch(&path_bufs, num_workers));

    results
        .into_iter()
        .map(|r| {
            let info = r.map_err(to_pyerr)?;
            Ok((info.width, info.height))
        })
        .collect()
}
