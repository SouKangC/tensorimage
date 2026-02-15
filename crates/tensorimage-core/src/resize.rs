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

/// Compute shortest-edge resize dimensions (same logic as resize_shortest_edge).
fn compute_shortest_edge_dims(w: u32, h: u32, target_size: u32) -> (u32, u32) {
    if w < h {
        let new_w = target_size;
        let new_h = (h as f64 * target_size as f64 / w as f64).round() as u32;
        (new_w, new_h)
    } else {
        let new_h = target_size;
        let new_w = (w as f64 * target_size as f64 / h as f64).round() as u32;
        (new_w, new_h)
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
    let (new_w, new_h) = compute_shortest_edge_dims(w, h, target_size);
    resize_exact(image, new_w, new_h, algorithm)
}

/// Resize to exact dimensions.
pub fn resize_exact(
    image: DecodedImage,
    new_width: u32,
    new_height: u32,
    algorithm: Algorithm,
) -> Result<DecodedImage> {
    // Skip resize if already at target dimensions
    if new_width == image.width && new_height == image.height {
        return Ok(image);
    }

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

/// Resize to exact dimensions from a mutable slice (avoids DecodedImage wrapper).
pub fn resize_exact_borrowed(
    data: &mut [u8],
    width: u32,
    height: u32,
    new_width: u32,
    new_height: u32,
    algorithm: Algorithm,
) -> Result<DecodedImage> {
    if new_width == width && new_height == height {
        return Ok(DecodedImage {
            data: data.to_vec(),
            width,
            height,
            channels: 3,
        });
    }

    let src = Image::from_slice_u8(width, height, data, PixelType::U8x3)
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

/// Fused resize + center crop: resizes directly from source to crop dimensions,
/// using source-space cropping to skip intermediate allocation and redundant pixels.
pub fn resize_crop_fused(
    image: DecodedImage,
    target_shortest_edge: u32,
    crop_w: u32,
    crop_h: u32,
    algorithm: Algorithm,
) -> Result<DecodedImage> {
    let (w, h) = (image.width, image.height);

    // Compute intermediate shortest-edge dimensions (same as resize_shortest_edge)
    let (inter_w, inter_h) = compute_shortest_edge_dims(w, h, target_shortest_edge);

    // If the crop is larger than the intermediate, fall back to sequential resize + crop
    if crop_w > inter_w || crop_h > inter_h {
        let resized = resize_shortest_edge(image, target_shortest_edge, algorithm)?;
        return crate::crop::center_crop(resized, crop_w, crop_h);
    }

    // If the intermediate already equals the crop, just resize directly
    if inter_w == crop_w && inter_h == crop_h {
        return resize_exact(image, crop_w, crop_h, algorithm);
    }

    // Compute center-crop offsets in intermediate (resized) coordinates
    let crop_x = (inter_w - crop_w) as f64 / 2.0;
    let crop_y = (inter_h - crop_h) as f64 / 2.0;

    // Map the crop region back to source image coordinates
    let scale_x = w as f64 / inter_w as f64;
    let scale_y = h as f64 / inter_h as f64;
    let src_left = crop_x * scale_x;
    let src_top = crop_y * scale_y;
    let src_width = crop_w as f64 * scale_x;
    let src_height = crop_h as f64 * scale_y;

    let src = Image::from_vec_u8(w, h, image.data, PixelType::U8x3)
        .map_err(|e| TensorImageError::Resize(e.to_string()))?;

    let mut dst = Image::new(crop_w, crop_h, PixelType::U8x3);

    let mut resizer = Resizer::new();
    let options = ResizeOptions::new()
        .resize_alg(algorithm.to_resize_alg())
        .crop(src_left, src_top, src_width, src_height);

    resizer
        .resize(&src, &mut dst, Some(&options))
        .map_err(|e| TensorImageError::Resize(e.to_string()))?;

    Ok(DecodedImage {
        data: dst.into_vec(),
        width: crop_w,
        height: crop_h,
        channels: 3,
    })
}
