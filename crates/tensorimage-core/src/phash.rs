use std::path::Path;

use crate::decode::decode_file;
use crate::error::{Result, TensorImageError};

/// Hash algorithm for perceptual hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    /// Difference hash — fastest, compares adjacent pixels. Produces 64-bit hash.
    DHash,
    /// Perceptual hash — more robust, uses DCT. Produces 64-bit hash.
    PHash,
}

impl HashAlgorithm {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "dhash" => Ok(HashAlgorithm::DHash),
            "phash" => Ok(HashAlgorithm::PHash),
            _ => Err(TensorImageError::Hash(format!(
                "Unknown hash algorithm: '{}'. Supported: dhash, phash",
                s
            ))),
        }
    }
}

/// Compute the Hamming distance between two 64-bit hashes.
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Compute a difference hash (dHash) from RGB pixel data.
///
/// Grayscale → resize to 9×8 → compare adjacent pixels → 64 bits.
pub fn dhash(data: &[u8], width: u32, height: u32) -> Result<u64> {
    let gray = rgb_to_grayscale(data, width, height)?;
    let resized = resize_grayscale(&gray, width as usize, height as usize, 9, 8);
    // Compare adjacent horizontal pixels: 8 columns × 8 rows = 64 bits
    let mut hash: u64 = 0;
    for row in 0..8 {
        for col in 0..8 {
            hash <<= 1;
            if resized[row * 9 + col] < resized[row * 9 + col + 1] {
                hash |= 1;
            }
        }
    }
    Ok(hash)
}

/// Compute a dHash from an image file. Uses IDCT-scaled decode for speed.
pub fn dhash_file(path: &Path) -> Result<u64> {
    let img = decode_file(path, Some(32))?;
    dhash(&img.data, img.width, img.height)
}

/// Compute a perceptual hash (pHash) from RGB pixel data.
///
/// Grayscale → resize to 32×32 → 8×8 DCT of top-left → median threshold → 64 bits.
pub fn phash(data: &[u8], width: u32, height: u32) -> Result<u64> {
    let gray = rgb_to_grayscale(data, width, height)?;
    let resized = resize_grayscale(&gray, width as usize, height as usize, 32, 32);

    // Convert to f32 for DCT
    let floats: Vec<f32> = resized.iter().map(|&b| b as f32).collect();

    // Compute 2D DCT on 32×32 block
    let dct = dct_2d(&floats, 32);

    // Extract top-left 8×8 (low-frequency components), excluding DC (0,0)
    let mut low_freq = Vec::with_capacity(64);
    for row in 0..8 {
        for col in 0..8 {
            low_freq.push(dct[row * 32 + col]);
        }
    }

    // Compute median of 8×8 block (excluding DC term at [0])
    let mut sorted: Vec<f32> = low_freq[1..].to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    // Threshold against median → 64-bit hash
    let mut hash: u64 = 0;
    for &val in &low_freq {
        hash <<= 1;
        if val > median {
            hash |= 1;
        }
    }
    Ok(hash)
}

/// Compute a pHash from an image file. Uses IDCT-scaled decode for speed.
pub fn phash_file(path: &Path) -> Result<u64> {
    let img = decode_file(path, Some(32))?;
    phash(&img.data, img.width, img.height)
}

/// Compute a hash from a file using the specified algorithm.
pub fn hash_file(path: &Path, algorithm: HashAlgorithm) -> Result<u64> {
    match algorithm {
        HashAlgorithm::DHash => dhash_file(path),
        HashAlgorithm::PHash => phash_file(path),
    }
}

/// Convert RGB pixel data to grayscale using BT.601 weights.
fn rgb_to_grayscale(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize) * 3;
    if data.len() < expected {
        return Err(TensorImageError::Hash(format!(
            "Expected {} bytes for {}x{} RGB, got {}",
            expected,
            width,
            height,
            data.len()
        )));
    }

    let n_pixels = (width as usize) * (height as usize);
    let mut gray = Vec::with_capacity(n_pixels);
    for i in 0..n_pixels {
        let r = data[i * 3] as f32;
        let g = data[i * 3 + 1] as f32;
        let b = data[i * 3 + 2] as f32;
        // BT.601 luma weights
        gray.push((0.299 * r + 0.587 * g + 0.114 * b) as u8);
    }
    Ok(gray)
}

/// Simple area-average downscale for grayscale images.
/// Designed for tiny targets (9×8 or 32×32) — no dependency on fast_image_resize.
fn resize_grayscale(
    data: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<u8> {
    let mut result = Vec::with_capacity(dst_width * dst_height);

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            // Map destination pixel to source region
            let src_x0 = (dst_x * src_width) / dst_width;
            let src_x1 = ((dst_x + 1) * src_width) / dst_width;
            let src_y0 = (dst_y * src_height) / dst_height;
            let src_y1 = ((dst_y + 1) * src_height) / dst_height;

            // Average all source pixels in the region
            let mut sum: u32 = 0;
            let mut count: u32 = 0;
            for sy in src_y0..src_y1 {
                for sx in src_x0..src_x1 {
                    sum += data[sy * src_width + sx] as u32;
                    count += 1;
                }
            }
            result.push(if count > 0 { (sum / count) as u8 } else { 0 });
        }
    }

    result
}

/// Naive 2D DCT (Type II) on an n×n block.
/// For n=32, this is 32×32 = 1024 values, ~1M multiply-adds — fast enough for hashing.
fn dct_2d(block: &[f32], n: usize) -> Vec<f32> {
    // First pass: DCT on each row
    let mut row_dct = vec![0.0f32; n * n];
    for y in 0..n {
        for u in 0..n {
            let mut sum = 0.0f32;
            for x in 0..n {
                let cos_val =
                    (std::f32::consts::PI * (2.0 * x as f32 + 1.0) * u as f32 / (2.0 * n as f32))
                        .cos();
                sum += block[y * n + x] * cos_val;
            }
            row_dct[y * n + u] = sum;
        }
    }

    // Second pass: DCT on each column
    let mut result = vec![0.0f32; n * n];
    for x in 0..n {
        for v in 0..n {
            let mut sum = 0.0f32;
            for y in 0..n {
                let cos_val =
                    (std::f32::consts::PI * (2.0 * y as f32 + 1.0) * v as f32 / (2.0 * n as f32))
                        .cos();
                sum += row_dct[y * n + x] * cos_val;
            }
            result[v * n + x] = sum;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_identical() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(u64::MAX, u64::MAX), 0);
    }

    #[test]
    fn test_hamming_opposite() {
        assert_eq!(hamming_distance(0, u64::MAX), 64);
    }

    #[test]
    fn test_hamming_single_bit() {
        assert_eq!(hamming_distance(0, 1), 1);
        assert_eq!(hamming_distance(0b1010, 0b1011), 1);
    }

    #[test]
    fn test_grayscale_conversion() {
        // Pure red pixel
        let data = vec![255, 0, 0];
        let gray = rgb_to_grayscale(&data, 1, 1).unwrap();
        assert_eq!(gray[0], 76); // 0.299 * 255 ≈ 76
    }

    #[test]
    fn test_resize_grayscale() {
        // 4×4 → 2×2: should average each 2×2 block
        let data = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let result = resize_grayscale(&data, 4, 4, 2, 2);
        assert_eq!(result.len(), 4);
        // Top-left 2×2: (10+20+50+60)/4 = 35
        assert_eq!(result[0], 35);
    }
}
