use crate::decode::DecodedImage;
use crate::error::{Result, TensorImageError};

#[derive(Debug, Clone, Copy)]
pub enum CropMode {
    Center,
}

impl CropMode {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "center" => Ok(CropMode::Center),
            _ => Err(TensorImageError::InvalidParam(format!(
                "Unknown crop mode '{}'. Valid options: center",
                s
            ))),
        }
    }
}

/// Extract a center crop of `crop_width x crop_height` from the image.
pub fn center_crop(
    image: DecodedImage,
    crop_width: u32,
    crop_height: u32,
) -> Result<DecodedImage> {
    if crop_width > image.width || crop_height > image.height {
        return Err(TensorImageError::Crop(format!(
            "Crop size {}x{} exceeds image size {}x{}",
            crop_width, crop_height, image.width, image.height
        )));
    }

    let x_offset = ((image.width - crop_width) / 2) as usize;
    let y_offset = ((image.height - crop_height) / 2) as usize;
    let channels = image.channels as usize;
    let src_stride = image.width as usize * channels;
    let dst_stride = crop_width as usize * channels;
    let dst_len = crop_height as usize * dst_stride;

    let mut data = vec![0u8; dst_len];
    for row in 0..crop_height as usize {
        let src_start = (y_offset + row) * src_stride + x_offset * channels;
        let dst_start = row * dst_stride;
        data[dst_start..dst_start + dst_stride]
            .copy_from_slice(&image.data[src_start..src_start + dst_stride]);
    }

    Ok(DecodedImage {
        data,
        width: crop_width,
        height: crop_height,
        channels: image.channels,
    })
}
