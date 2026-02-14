use std::path::Path;

use crate::crop::{CropMode, center_crop};
use crate::decode::{DecodedImage, decode_file};
use crate::error::Result;
use crate::normalize::{NormalizeParams, normalize_hwc_to_chw};
use crate::resize::{Algorithm, resize_shortest_edge};

pub enum PipelineOutput {
    /// No normalize → u8 [H, W, 3]
    U8Hwc(DecodedImage),
    /// Normalize → f32 [3, H, W]
    F32Chw {
        data: Vec<f32>,
        height: u32,
        width: u32,
    },
}

pub struct PipelineConfig {
    pub size: Option<u32>,
    pub algorithm: Algorithm,
    pub crop: Option<(CropMode, u32, u32)>,
    pub normalize: Option<NormalizeParams>,
}

/// Execute the full processing pipeline: decode → resize → crop → normalize+transpose.
/// Each step is skipped if its config field is `None`.
pub fn execute_pipeline(path: &Path, config: &PipelineConfig) -> Result<PipelineOutput> {
    // 1. Decode (with IDCT hint for shortest-edge target)
    let mut image = decode_file(path, config.size)?;

    // 2. Resize shortest edge
    if let Some(size) = config.size {
        image = resize_shortest_edge(image, size, config.algorithm)?;
    }

    // 3. Center crop
    if let Some((CropMode::Center, cw, ch)) = &config.crop {
        image = center_crop(image, *cw, *ch)?;
    }

    // 4. Normalize + HWC→CHW transpose (fused)
    match &config.normalize {
        Some(params) => {
            let height = image.height;
            let width = image.width;
            let data = normalize_hwc_to_chw(&image, params);
            Ok(PipelineOutput::F32Chw {
                data,
                height,
                width,
            })
        }
        None => Ok(PipelineOutput::U8Hwc(image)),
    }
}
