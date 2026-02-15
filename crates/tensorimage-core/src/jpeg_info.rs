use std::path::Path;
use std::path::PathBuf;

use rayon::prelude::*;

use crate::error::{Result, TensorImageError};
use crate::pool::get_or_create_pool;

/// Image dimensions (width × height).
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
}

/// Read image dimensions from file without full decode.
///
/// - JPEG: uses turbojpeg `read_header()` (fast, header-only)
/// - PNG: parses the IHDR chunk (first 33 bytes)
/// - Other: falls back to `image` crate header reader
pub fn image_info(path: &Path) -> Result<ImageInfo> {
    let bytes = std::fs::read(path)?;

    // JPEG: 0xFF 0xD8
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        let mut decompressor =
            turbojpeg::Decompressor::new().map_err(|e| TensorImageError::Decode(e.to_string()))?;
        let header = decompressor
            .read_header(&bytes)
            .map_err(|e| TensorImageError::Decode(e.to_string()))?;
        return Ok(ImageInfo {
            width: header.width as u32,
            height: header.height as u32,
        });
    }

    // PNG: check signature + IHDR
    if bytes.len() >= 24
        && bytes[0..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    {
        // IHDR starts at offset 16 (8 signature + 4 length + 4 "IHDR")
        let width = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let height = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        return Ok(ImageInfo { width, height });
    }

    // Fallback: use image crate reader (reads header only)
    let reader = image::ImageReader::open(path)
        .map_err(|e| TensorImageError::Decode(e.to_string()))?;
    let dims = reader
        .into_dimensions()
        .map_err(|e| TensorImageError::Decode(e.to_string()))?;
    Ok(ImageInfo {
        width: dims.0,
        height: dims.1,
    })
}

/// Read image dimensions for a batch of files in parallel.
pub fn image_info_batch(
    paths: &[PathBuf],
    num_workers: usize,
) -> Vec<Result<ImageInfo>> {
    let pool = get_or_create_pool(num_workers);

    pool.install(|| {
        paths
            .par_iter()
            .map(|p| image_info(p))
            .collect()
    })
}
