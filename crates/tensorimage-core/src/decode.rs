use std::path::Path;

use crate::error::{Result, TensorImageError};

pub struct DecodedImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

/// Detect format from file extension or magic bytes.
fn is_jpeg(path: &Path, bytes: &[u8]) -> bool {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg")
    } else {
        bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8
    }
}

/// Decode a JPEG using turbojpeg (libjpeg-turbo).
/// If `target_shortest_edge` is provided, uses IDCT scaling to decode at a
/// smaller resolution (the largest scale factor whose output shortest edge >= target).
fn decode_jpeg(bytes: &[u8], target_shortest_edge: Option<u32>) -> Result<DecodedImage> {
    let mut decompressor =
        turbojpeg::Decompressor::new().map_err(|e| TensorImageError::Decode(e.to_string()))?;

    let header = decompressor
        .read_header(bytes)
        .map_err(|e| TensorImageError::Decode(e.to_string()))?;

    // Apply IDCT scaling if a target size is given
    if let Some(target) = target_shortest_edge {
        let factors = turbojpeg::Decompressor::supported_scaling_factors();

        // Find the smallest scaling factor whose scaled shortest edge >= target.
        let mut best = turbojpeg::ScalingFactor::ONE;
        for f in &factors {
            let scaled = header.scaled(*f);
            let shortest = scaled.width.min(scaled.height);
            if shortest >= target as usize {
                best = *f;
            }
        }
        decompressor
            .set_scaling_factor(best)
            .map_err(|e| TensorImageError::Decode(e.to_string()))?;
    }

    // Get output dimensions (after scaling)
    let scaled = header.scaled(decompressor.scaling_factor());
    let width = scaled.width;
    let height = scaled.height;

    let mut image = turbojpeg::Image {
        pixels: vec![0u8; 3 * width * height],
        width,
        pitch: 3 * width,
        height,
        format: turbojpeg::PixelFormat::RGB,
    };

    decompressor
        .decompress(bytes, image.as_deref_mut())
        .map_err(|e| TensorImageError::Decode(e.to_string()))?;

    Ok(DecodedImage {
        data: image.pixels,
        width: width as u32,
        height: height as u32,
        channels: 3,
    })
}

/// Decode a non-JPEG image using the `image` crate (handles PNG, etc.).
fn decode_other(bytes: &[u8]) -> Result<DecodedImage> {
    let img =
        image::load_from_memory(bytes).map_err(|e| TensorImageError::Decode(e.to_string()))?;
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    Ok(DecodedImage {
        data: rgb.into_raw(),
        width,
        height,
        channels: 3,
    })
}

/// Decode an image file. For JPEG, uses turbojpeg with optional IDCT scaling.
pub fn decode_file(path: &Path, target_shortest_edge: Option<u32>) -> Result<DecodedImage> {
    let bytes = std::fs::read(path)?;
    if is_jpeg(path, &bytes) {
        decode_jpeg(&bytes, target_shortest_edge)
    } else {
        decode_other(&bytes)
    }
}

/// Decode from raw bytes (format auto-detected).
pub fn decode_bytes(bytes: &[u8], target_shortest_edge: Option<u32>) -> Result<DecodedImage> {
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        decode_jpeg(bytes, target_shortest_edge)
    } else {
        decode_other(bytes)
    }
}
