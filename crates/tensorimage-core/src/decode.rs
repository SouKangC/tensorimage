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
        is_jpeg_bytes(bytes)
    }
}

/// Detect JPEG from magic bytes only.
fn is_jpeg_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8
}

/// Detect WebP from magic bytes: RIFF....WEBP
fn is_webp(bytes: &[u8]) -> bool {
    bytes.len() >= 12
        && &bytes[0..4] == b"RIFF"
        && &bytes[8..12] == b"WEBP"
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

/// Decode a WebP image using the `webp` crate (libwebp C bindings).
fn decode_webp(bytes: &[u8]) -> Result<DecodedImage> {
    let decoder = webp::Decoder::new(bytes);
    let webp_image = decoder
        .decode()
        .ok_or_else(|| TensorImageError::Decode("Failed to decode WebP image".into()))?;

    // Convert to DynamicImage then to RGB8 — handles both RGB and RGBA WebP
    let rgb = webp_image.to_image().into_rgb8();
    let (w, h) = rgb.dimensions();
    Ok(DecodedImage {
        data: rgb.into_raw(),
        width: w,
        height: h,
        channels: 3,
    })
}

/// Decode a non-JPEG, non-WebP image using the `image` crate (handles PNG, AVIF, etc.).
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

/// Decode an image file. Routes: JPEG → WebP → fallback.
/// Applies EXIF orientation correction for JPEG files.
pub fn decode_file(path: &Path, target_shortest_edge: Option<u32>) -> Result<DecodedImage> {
    let bytes = std::fs::read(path)?;

    // Only read EXIF from JPEG — WebP/AVIF decoders handle orientation internally
    let orientation = if is_jpeg(path, &bytes) || is_jpeg_bytes(&bytes) {
        crate::exif::read_exif_orientation(&bytes)
    } else {
        None
    };

    let image = if is_jpeg(path, &bytes) {
        decode_jpeg(&bytes, target_shortest_edge)?
    } else if is_webp(&bytes) {
        decode_webp(&bytes)?
    } else {
        decode_other(&bytes)?
    };

    Ok(match orientation {
        Some(o) if o != 1 => crate::exif::apply_orientation(image, o),
        _ => image,
    })
}

/// Decode from raw bytes (format auto-detected via magic bytes).
/// Applies EXIF orientation correction for JPEG data.
pub fn decode_bytes(bytes: &[u8], target_shortest_edge: Option<u32>) -> Result<DecodedImage> {
    // Only read EXIF from JPEG
    let orientation = if is_jpeg_bytes(bytes) {
        crate::exif::read_exif_orientation(bytes)
    } else {
        None
    };

    let image = if is_jpeg_bytes(bytes) {
        decode_jpeg(bytes, target_shortest_edge)?
    } else if is_webp(bytes) {
        decode_webp(bytes)?
    } else {
        decode_other(bytes)?
    };

    Ok(match orientation {
        Some(o) if o != 1 => crate::exif::apply_orientation(image, o),
        _ => image,
    })
}
