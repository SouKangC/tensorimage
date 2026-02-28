use crate::error::{Result, TensorImageError};

/// Generate a 1D Gaussian kernel of the given size and sigma.
fn gaussian_kernel_1d(kernel_size: u32, sigma: f64) -> Vec<f64> {
    let center = (kernel_size / 2) as f64;
    let mut kernel: Vec<f64> = (0..kernel_size)
        .map(|i| {
            let x = i as f64 - center;
            (-0.5 * (x / sigma).powi(2)).exp()
        })
        .collect();
    let sum: f64 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= sum;
    }
    kernel
}

/// Reflect-index helper: maps an out-of-bounds coordinate to its reflected
/// position, matching torchvision's reflection padding behaviour.
#[inline]
fn reflect_index(i: isize, size: usize) -> usize {
    if i < 0 {
        (-i) as usize % size.max(1)
    } else if (i as usize) >= size {
        let rem = (i as usize) % (2 * size - 2).max(1);
        if rem < size {
            rem
        } else {
            2 * (size - 1) - rem
        }
    } else {
        i as usize
    }
}

/// Separable Gaussian blur on a u8 HWC image.
///
/// Two 1D passes (horizontal then vertical) — O(n*k) per axis.
/// Edge handling: reflection padding (matches torchvision).
pub fn gaussian_blur(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
    sigma: f64,
) -> Result<Vec<u8>> {
    if kernel_size == 0 || kernel_size % 2 == 0 {
        return Err(TensorImageError::Augment(
            "kernel_size must be a positive odd number".into(),
        ));
    }
    if sigma <= 0.0 {
        return Err(TensorImageError::Augment(
            "sigma must be positive".into(),
        ));
    }

    let w = width as usize;
    let h = height as usize;
    let c = channels as usize;
    let ks = kernel_size as usize;
    let half = ks / 2;
    let kernel = gaussian_kernel_1d(kernel_size, sigma);

    // Horizontal pass: data (u8) → temp (f32)
    let mut temp = vec![0.0f32; h * w * c];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let mut sum = 0.0f64;
                for ki in 0..ks {
                    let sx = reflect_index(x as isize + ki as isize - half as isize, w);
                    sum += data[y * w * c + sx * c + ch] as f64 * kernel[ki];
                }
                temp[y * w * c + x * c + ch] = sum as f32;
            }
        }
    }

    // Vertical pass: temp (f32) → output (u8)
    let mut output = vec![0u8; h * w * c];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let mut sum = 0.0f64;
                for ki in 0..ks {
                    let sy = reflect_index(y as isize + ki as isize - half as isize, h);
                    sum += temp[sy * w * c + x * c + ch] as f64 * kernel[ki];
                }
                output[y * w * c + x * c + ch] = sum.round().max(0.0).min(255.0) as u8;
            }
        }
    }

    Ok(output)
}

/// Bilinear interpolation helper for a single pixel.
#[inline]
fn bilinear_sample(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    sx: f64,
    sy: f64,
    fill: &[u8],
) -> Vec<u8> {
    let x0 = sx.floor() as isize;
    let y0 = sy.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = sx - x0 as f64;
    let fy = sy - y0 as f64;

    let mut result = vec![0u8; channels];

    let in_bounds = |x: isize, y: isize| -> bool {
        x >= 0 && x < width as isize && y >= 0 && y < height as isize
    };

    let pixel = |x: isize, y: isize, ch: usize| -> f64 {
        if in_bounds(x, y) {
            data[(y as usize) * width * channels + (x as usize) * channels + ch] as f64
        } else {
            fill[ch % fill.len()] as f64
        }
    };

    for ch in 0..channels {
        let v00 = pixel(x0, y0, ch);
        let v10 = pixel(x1, y0, ch);
        let v01 = pixel(x0, y1, ch);
        let v11 = pixel(x1, y1, ch);

        let v = v00 * (1.0 - fx) * (1.0 - fy)
            + v10 * fx * (1.0 - fy)
            + v01 * (1.0 - fx) * fy
            + v11 * fx * fy;

        result[ch] = v.round().max(0.0).min(255.0) as u8;
    }

    result
}

/// Apply a 2x3 affine transformation with bilinear interpolation.
///
/// matrix = [a, b, tx, c, d, ty] representing the FORWARD transform.
/// Uses inverse mapping: for each output pixel, compute source coordinates.
pub fn affine_transform(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    matrix: &[f64; 6],
    out_width: u32,
    out_height: u32,
    fill: &[u8],
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let c = channels as usize;
    let ow = out_width as usize;
    let oh = out_height as usize;

    // Compute inverse of the 2x2 part: [[a, b], [c, d]]
    let a = matrix[0];
    let b = matrix[1];
    let tx = matrix[2];
    let cc = matrix[3];
    let d = matrix[4];
    let ty = matrix[5];

    let det = a * d - b * cc;
    if det.abs() < 1e-10 {
        return Err(TensorImageError::Augment(
            "Affine matrix is singular".into(),
        ));
    }

    let inv_det = 1.0 / det;
    // Inverse matrix: [[d, -b], [-c, a]] / det
    let ia = d * inv_det;
    let ib = -b * inv_det;
    let ic = -cc * inv_det;
    let id = a * inv_det;
    // Inverse translation
    let itx = -(ia * tx + ib * ty);
    let ity = -(ic * tx + id * ty);

    let mut output = vec![0u8; oh * ow * c];

    for oy in 0..oh {
        for ox in 0..ow {
            let sx = ia * ox as f64 + ib * oy as f64 + itx;
            let sy = ic * ox as f64 + id * oy as f64 + ity;

            let pixel = bilinear_sample(data, w, h, c, sx, sy, fill);
            let idx = (oy * ow + ox) * c;
            output[idx..idx + c].copy_from_slice(&pixel);
        }
    }

    Ok(output)
}

/// Apply a perspective transformation.
///
/// coeffs = [a, b, c, d, e, f, g, h] where:
///   sx = (a*ox + b*oy + c) / (g*ox + h*oy + 1)
///   sy = (d*ox + e*oy + f) / (g*ox + h*oy + 1)
///
/// These coefficients map OUTPUT to INPUT (inverse mapping).
pub fn perspective_transform(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    coeffs: &[f64; 8],
    out_width: u32,
    out_height: u32,
    fill: &[u8],
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let c = channels as usize;
    let ow = out_width as usize;
    let oh = out_height as usize;

    let mut output = vec![0u8; oh * ow * c];

    for oy in 0..oh {
        for ox in 0..ow {
            let ox_f = ox as f64;
            let oy_f = oy as f64;
            let denom = coeffs[6] * ox_f + coeffs[7] * oy_f + 1.0;

            if denom.abs() < 1e-10 {
                // Degenerate — fill
                let idx = (oy * ow + ox) * c;
                for ch in 0..c {
                    output[idx + ch] = fill[ch % fill.len()];
                }
                continue;
            }

            let sx = (coeffs[0] * ox_f + coeffs[1] * oy_f + coeffs[2]) / denom;
            let sy = (coeffs[3] * ox_f + coeffs[4] * oy_f + coeffs[5]) / denom;

            let pixel = bilinear_sample(data, w, h, c, sx, sy, fill);
            let idx = (oy * ow + ox) * c;
            output[idx..idx + c].copy_from_slice(&pixel);
        }
    }

    Ok(output)
}
