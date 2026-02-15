use crate::decode::DecodedImage;

/// Read EXIF orientation tag from JPEG bytes. Returns None for non-JPEG or missing tag.
pub fn read_exif_orientation(bytes: &[u8]) -> Option<u16> {
    let exif_reader = exif::Reader::new();
    let mut cursor = std::io::Cursor::new(bytes);
    let exif_data = exif_reader.read_from_container(&mut cursor).ok()?;
    let orientation = exif_data.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
    orientation.value.get_uint(0).map(|v| v as u16)
}

/// Apply EXIF orientation transform to a decoded image.
/// Only orientations 2-8 need transforms; orientation 1 is identity.
pub fn apply_orientation(image: DecodedImage, orientation: u16) -> DecodedImage {
    match orientation {
        2 => flip_horizontal(image),
        3 => rotate_180(image),
        4 => flip_vertical(image),
        5 => flip_horizontal(rotate_90_cw(image)),
        6 => rotate_90_cw(image),
        7 => flip_vertical(rotate_90_cw(image)),
        8 => rotate_270_cw(image),
        _ => image,
    }
}

fn flip_horizontal(mut image: DecodedImage) -> DecodedImage {
    let w = image.width as usize;
    let c = image.channels as usize;
    let stride = w * c;
    for row in image.data.chunks_exact_mut(stride) {
        for x in 0..w / 2 {
            let left = x * c;
            let right = (w - 1 - x) * c;
            for ch in 0..c {
                row.swap(left + ch, right + ch);
            }
        }
    }
    image
}

fn flip_vertical(mut image: DecodedImage) -> DecodedImage {
    let w = image.width as usize;
    let h = image.height as usize;
    let c = image.channels as usize;
    let stride = w * c;
    let mut buf = vec![0u8; stride];
    for y in 0..h / 2 {
        let top = y * stride;
        let bot = (h - 1 - y) * stride;
        buf.copy_from_slice(&image.data[top..top + stride]);
        image.data.copy_within(bot..bot + stride, top);
        image.data[bot..bot + stride].copy_from_slice(&buf);
    }
    image
}

fn rotate_180(image: DecodedImage) -> DecodedImage {
    flip_vertical(flip_horizontal(image))
}

/// Rotate 90 degrees clockwise. Output dimensions are swapped (W×H → H×W).
fn rotate_90_cw(image: DecodedImage) -> DecodedImage {
    let w = image.width as usize;
    let h = image.height as usize;
    let c = image.channels as usize;
    let new_w = h;
    let new_h = w;
    let mut out = vec![0u8; new_w * new_h * c];

    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * c;
            // (x, y) → (h-1-y, x) in the rotated image
            let dst_x = h - 1 - y;
            let dst_y = x;
            let dst = (dst_y * new_w + dst_x) * c;
            out[dst..dst + c].copy_from_slice(&image.data[src..src + c]);
        }
    }

    DecodedImage {
        data: out,
        width: new_w as u32,
        height: new_h as u32,
        channels: image.channels,
    }
}

/// Rotate 270 degrees clockwise (= 90 degrees counter-clockwise).
fn rotate_270_cw(image: DecodedImage) -> DecodedImage {
    let w = image.width as usize;
    let h = image.height as usize;
    let c = image.channels as usize;
    let new_w = h;
    let new_h = w;
    let mut out = vec![0u8; new_w * new_h * c];

    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * c;
            // (x, y) → (y, w-1-x) in the rotated image
            let dst_x = y;
            let dst_y = w - 1 - x;
            let dst = (dst_y * new_w + dst_x) * c;
            out[dst..dst + c].copy_from_slice(&image.data[src..src + c]);
        }
    }

    DecodedImage {
        data: out,
        width: new_w as u32,
        height: new_h as u32,
        channels: image.channels,
    }
}
