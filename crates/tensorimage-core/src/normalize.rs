use crate::decode::DecodedImage;
use crate::error::{Result, TensorImageError};

#[derive(Debug, Clone)]
pub struct NormalizeParams {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl NormalizeParams {
    pub fn from_preset(name: &str) -> Result<Self> {
        match name.to_lowercase().as_str() {
            "imagenet" => Ok(NormalizeParams {
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            }),
            "clip" => Ok(NormalizeParams {
                mean: [0.48145466, 0.4578275, 0.40821073],
                std: [0.26862954, 0.26130258, 0.27577711],
            }),
            "[-1,1]" => Ok(NormalizeParams {
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
            }),
            _ => Err(TensorImageError::Normalize(format!(
                "Unknown normalize preset '{}'. Valid options: imagenet, clip, [-1,1]",
                name
            ))),
        }
    }

    pub fn custom(mean: [f32; 3], std: [f32; 3]) -> Result<Self> {
        for i in 0..3 {
            if std[i] == 0.0 {
                return Err(TensorImageError::Normalize(
                    "Standard deviation must not be zero".to_string(),
                ));
            }
        }
        Ok(NormalizeParams { mean, std })
    }
}

/// Fused single-pass: normalizes u8 HWC pixels to f32 and transposes to CHW layout.
/// Computes `(pixel / 255.0 - mean) / std` per channel while writing in CHW order.
/// Pre-computes `scale = 1 / (255 * std)` and `bias = -mean / std` per channel.
pub fn normalize_hwc_to_chw(image: &DecodedImage, params: &NormalizeParams) -> Vec<f32> {
    normalize_hwc_to_chw_from_slice(&image.data, image.width, image.height, params)
}

/// Slice-based variant: normalizes u8 HWC pixels to f32 CHW without requiring a DecodedImage.
pub fn normalize_hwc_to_chw_from_slice(
    pixels: &[u8],
    width: u32,
    height: u32,
    params: &NormalizeParams,
) -> Vec<f32> {
    let h = height as usize;
    let w = width as usize;
    let total = h * w;

    let mut output = vec![0.0f32; 3 * total];
    normalize_hwc_to_chw_into(pixels, width, height, params, &mut output);
    output
}

/// Write-into variant: normalizes u8 HWC pixels to f32 CHW directly into a provided buffer.
/// `output` must have length >= 3 * width * height.
pub fn normalize_hwc_to_chw_into(
    pixels: &[u8],
    width: u32,
    height: u32,
    params: &NormalizeParams,
    output: &mut [f32],
) {
    let h = height as usize;
    let w = width as usize;
    let total = h * w;

    // Pre-compute fused scale and bias: output = pixel * scale + bias
    let mut scale = [0.0f32; 3];
    let mut bias = [0.0f32; 3];
    for c in 0..3 {
        scale[c] = 1.0 / (255.0 * params.std[c]);
        bias[c] = -params.mean[c] / params.std[c];
    }

    // Split output into 3 channel planes using split_at_mut for aliasing guarantees
    let (plane_r, rest) = output[..3 * total].split_at_mut(total);
    let (plane_g, plane_b) = rest.split_at_mut(total);

    for i in 0..total {
        let base = i * 3;
        plane_r[i] = pixels[base] as f32 * scale[0] + bias[0];
        plane_g[i] = pixels[base + 1] as f32 * scale[1] + bias[1];
        plane_b[i] = pixels[base + 2] as f32 * scale[2] + bias[2];
    }
}
