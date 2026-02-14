use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use crate::decode::DecodedImage;
use crate::error::{Result, TensorImageError};

#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    Nearest,
    Bilinear,
    CatmullRom,
    Mitchell,
    Lanczos3,
}

impl Algorithm {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "nearest" => Ok(Algorithm::Nearest),
            "bilinear" => Ok(Algorithm::Bilinear),
            "catmullrom" | "catmull-rom" | "catmull_rom" => Ok(Algorithm::CatmullRom),
            "mitchell" => Ok(Algorithm::Mitchell),
            "lanczos3" | "lanczos" => Ok(Algorithm::Lanczos3),
            _ => Err(TensorImageError::InvalidParam(format!(
                "Unknown resize algorithm '{}'. Valid options: nearest, bilinear, catmullrom, mitchell, lanczos3",
                s
            ))),
        }
    }

    fn to_resize_alg(self) -> ResizeAlg {
        match self {
            Algorithm::Nearest => ResizeAlg::Nearest,
            Algorithm::Bilinear => ResizeAlg::Convolution(FilterType::Bilinear),
            Algorithm::CatmullRom => ResizeAlg::Convolution(FilterType::CatmullRom),
            Algorithm::Mitchell => ResizeAlg::Convolution(FilterType::Mitchell),
            Algorithm::Lanczos3 => ResizeAlg::Convolution(FilterType::Lanczos3),
        }
    }
}

/// Resize so that the shortest edge equals `target_size`, preserving aspect ratio.
/// Matches torchvision `Resize(size)` semantics.
pub fn resize_shortest_edge(
    image: DecodedImage,
    target_size: u32,
    algorithm: Algorithm,
) -> Result<DecodedImage> {
    let (w, h) = (image.width, image.height);
    let (new_w, new_h) = if w < h {
        let new_w = target_size;
        let new_h = (h as f64 * target_size as f64 / w as f64).round() as u32;
        (new_w, new_h)
    } else {
        let new_h = target_size;
        let new_w = (w as f64 * target_size as f64 / h as f64).round() as u32;
        (new_w, new_h)
    };
    resize_exact(image, new_w, new_h, algorithm)
}

/// Resize to exact dimensions.
pub fn resize_exact(
    image: DecodedImage,
    new_width: u32,
    new_height: u32,
    algorithm: Algorithm,
) -> Result<DecodedImage> {
    let src = Image::from_vec_u8(image.width, image.height, image.data, PixelType::U8x3)
        .map_err(|e| TensorImageError::Resize(e.to_string()))?;

    let mut dst = Image::new(new_width, new_height, PixelType::U8x3);

    let mut resizer = Resizer::new();
    let options = ResizeOptions::new().resize_alg(algorithm.to_resize_alg());

    resizer
        .resize(&src, &mut dst, Some(&options))
        .map_err(|e| TensorImageError::Resize(e.to_string()))?;

    Ok(DecodedImage {
        data: dst.into_vec(),
        width: new_width,
        height: new_height,
        channels: 3,
    })
}
