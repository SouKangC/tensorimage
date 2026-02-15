use std::path::Path;

use crate::crop::{CropMode, center_crop};
use crate::decode::{DecodedImage, decode_bytes, decode_file};
use crate::error::{Result, TensorImageError};
use crate::normalize::{NormalizeParams, normalize_hwc_to_chw, normalize_hwc_to_chw_into};
use crate::resize::{Algorithm, resize_crop_fused, resize_shortest_edge};

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

/// Shared resize → crop logic for a decoded image.
fn pipeline_resize_crop(mut image: DecodedImage, config: &PipelineConfig) -> Result<DecodedImage> {
    match (config.size, &config.crop) {
        (Some(size), Some((CropMode::Center, cw, ch))) => {
            image = resize_crop_fused(image, size, *cw, *ch, config.algorithm)?;
        }
        (Some(size), None) => {
            image = resize_shortest_edge(image, size, config.algorithm)?;
        }
        (None, Some((CropMode::Center, cw, ch))) => {
            image = center_crop(image, *cw, *ch)?;
        }
        (None, None) => {}
    }
    Ok(image)
}

/// Decode → resize → crop (from file path).
fn pipeline_decode_resize_crop(path: &Path, config: &PipelineConfig) -> Result<DecodedImage> {
    let image = decode_file(path, config.size)?;
    pipeline_resize_crop(image, config)
}

/// Decode → resize → crop (from raw bytes).
fn pipeline_decode_resize_crop_bytes(bytes: &[u8], config: &PipelineConfig) -> Result<DecodedImage> {
    let image = decode_bytes(bytes, config.size)?;
    pipeline_resize_crop(image, config)
}

/// Execute the full processing pipeline: decode → resize → crop → normalize+transpose.
/// Each step is skipped if its config field is `None`.
pub fn execute_pipeline(path: &Path, config: &PipelineConfig) -> Result<PipelineOutput> {
    let image = pipeline_decode_resize_crop(path, config)?;

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

/// Execute the pipeline and write normalized f32 CHW output directly into a provided buffer.
/// Requires `config.normalize` to be set. `output` must have length >= 3 * crop_h * crop_w.
pub fn execute_pipeline_into(path: &Path, config: &PipelineConfig, output: &mut [f32]) -> Result<()> {
    let image = pipeline_decode_resize_crop(path, config)?;

    match &config.normalize {
        Some(params) => {
            normalize_hwc_to_chw_into(&image.data, image.width, image.height, params, output);
            Ok(())
        }
        None => Err(TensorImageError::InvalidParam(
            "execute_pipeline_into requires normalize to be set".into(),
        )),
    }
}

/// Execute the full processing pipeline from raw bytes.
pub fn execute_pipeline_bytes(bytes: &[u8], config: &PipelineConfig) -> Result<PipelineOutput> {
    let image = pipeline_decode_resize_crop_bytes(bytes, config)?;

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

/// Execute the bytes pipeline and write normalized f32 CHW output directly into a provided buffer.
pub fn execute_pipeline_bytes_into(bytes: &[u8], config: &PipelineConfig, output: &mut [f32]) -> Result<()> {
    let image = pipeline_decode_resize_crop_bytes(bytes, config)?;

    match &config.normalize {
        Some(params) => {
            normalize_hwc_to_chw_into(&image.data, image.width, image.height, params, output);
            Ok(())
        }
        None => Err(TensorImageError::InvalidParam(
            "execute_pipeline_bytes_into requires normalize to be set".into(),
        )),
    }
}
