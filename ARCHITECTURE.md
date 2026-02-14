# tensorimage — Architecture

## Design Principles

1. **Zero-copy end-to-end**: Pixels decoded in Rust flow to Python tensors without a single memcpy
2. **Fused operations**: Never allocate intermediate buffers between resize/crop/normalize
3. **GIL-free parallelism**: All heavy work happens in Rust with the GIL released
4. **Exact compatibility option**: Can match PIL/torchvision output bit-for-bit when needed
5. **Minimal dependencies**: The Rust binary should be self-contained (no system OpenCV, no libjpeg install)

## Module Design

### `decode.rs` — Image Decoding

```rust
pub enum ImageFormat {
    Jpeg,
    Png,
    WebP,
    Avif,
    Tiff,
    Auto, // detect from magic bytes
}

pub struct DecodedImage {
    pub data: Vec<u8>,        // RGB pixels, tightly packed
    pub width: u32,
    pub height: u32,
    pub channels: u8,         // 3 (RGB) or 4 (RGBA)
}

pub fn decode(bytes: &[u8], format: ImageFormat) -> Result<DecodedImage>;
pub fn decode_file(path: &Path) -> Result<DecodedImage>;
```

Use `image` crate for initial implementation. Later, consider:
- `turbojpeg` bindings for 2-3x faster JPEG decode (uses libjpeg-turbo SIMD)
- Custom WebP decoder if `image`'s is slow
- NVJPEG for GPU path (feature-gated behind `cuda` feature flag)

### `resize.rs` — SIMD Resize

```rust
pub enum ResizeAlgorithm {
    Nearest,
    Bilinear,
    CatmullRom,
    Mitchell,
    Lanczos3,    // default, highest quality
}

pub enum CropMode {
    Center,
    TopLeft,
    Random(u64),  // seed
    None,
}

/// Resize + optional crop in a single pass
pub fn resize_crop(
    image: &DecodedImage,
    target_width: u32,
    target_height: u32,
    algorithm: ResizeAlgorithm,
    crop: CropMode,
) -> Result<DecodedImage>;
```

Use `fast_image_resize` crate — it uses AVX2/SSE4.1/NEON SIMD intrinsics. Key: resize to the correct dimension first, then crop, to avoid resizing pixels that will be cropped.

### `pipeline.rs` — Fused Transform Pipeline

```rust
pub struct Pipeline {
    steps: Vec<Transform>,
}

pub enum Transform {
    Resize { width: u32, height: u32, algorithm: ResizeAlgorithm },
    CenterCrop { width: u32, height: u32 },
    RandomCrop { width: u32, height: u32, seed: u64 },
    Normalize { mean: [f32; 3], std: [f32; 3] },
    ToFloat,        // u8 → f32, divide by 255
    RandomFlipH { p: f32, seed: u64 },
    RandomFlipV { p: f32, seed: u64 },
    ColorJitter { brightness: f32, contrast: f32, saturation: f32, hue: f32 },
}

impl Pipeline {
    pub fn new() -> Self;
    pub fn add(mut self, transform: Transform) -> Self;

    /// Execute pipeline, returning f32 tensor [C, H, W]
    pub fn execute(&self, image: DecodedImage) -> Result<Vec<f32>>;

    /// Execute on batch in parallel
    pub fn execute_batch(&self, images: Vec<DecodedImage>) -> Result<Vec<Vec<f32>>>;
}
```

**Optimization**: The pipeline optimizer can fuse adjacent operations:
- `Resize(512) + CenterCrop(512)` → single resize to exact target
- `ToFloat + Normalize` → single pass: `(pixel / 255.0 - mean) / std`
- `RandomFlipH + RandomFlipV` → single pass with conditional axis reversal

### `batch.rs` — Parallel Batch Loading

```rust
use rayon::prelude::*;

pub fn load_batch(
    paths: &[PathBuf],
    pipeline: &Pipeline,
    num_workers: usize,  // rayon thread pool size
) -> Result<Vec<Vec<f32>>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build()?;

    pool.install(|| {
        paths.par_iter()
            .map(|path| {
                let image = decode_file(path)?;
                pipeline.execute(image)
            })
            .collect()
    })
}
```

### PyO3 Bindings (`tensorimage-python/src/lib.rs`)

```rust
use pyo3::prelude::*;
use numpy::{PyArray3, PyArray4};

#[pyfunction]
fn load(
    py: Python<'_>,
    path: &str,
    size: Option<u32>,
    crop: Option<&str>,
    normalize: Option<&str>,
) -> PyResult<Py<PyArray3<f32>>> {
    // Build pipeline from args
    // Execute (GIL released during Rust work)
    // Return as numpy array (zero-copy)
    py.allow_threads(|| {
        // All heavy work here — GIL is released
    })
}

#[pyfunction]
fn load_batch(
    py: Python<'_>,
    paths: Vec<String>,
    size: Option<u32>,
    crop: Option<&str>,
    normalize: Option<&str>,
    workers: Option<usize>,
) -> PyResult<Py<PyArray4<f32>>> {
    // Returns [N, C, H, W] tensor
}

#[pymodule]
fn _tensorimage(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_batch, m)?)?;
    Ok(())
}
```

## Memory Layout

Output tensor layout: **CHW** (channels first) to match PyTorch convention.

```
Input: JPEG file (compressed)
    ↓ decode
RGB pixels: [H, W, 3] u8 (HWC, row-major)
    ↓ resize + crop (in-place on same buffer when possible)
RGB pixels: [H', W', 3] u8
    ↓ to_float + normalize (single pass)
Float pixels: [3, H', W'] f32 (CHW, for PyTorch)
    ↓ PyO3 numpy bridge
numpy.ndarray: shape (3, H', W'), dtype float32
```

The HWC→CHW transpose happens during the normalize pass to avoid an extra copy.

## Error Handling Strategy

- Corrupt images: Return `Err` with descriptive message, skip in batch mode (configurable)
- Unsupported format: Return `Err`, suggest converting
- OOM on large batch: Chunk into sub-batches, process sequentially
- GPU unavailable: Graceful fallback to CPU path with warning

## Testing Strategy

1. **Correctness**: Compare output against PIL/torchvision for a test suite of 100 images. Max absolute difference per pixel < 1/255 for default mode, == 0 for compatibility mode.
2. **Performance**: `criterion` benchmarks for Rust, `pytest-benchmark` for Python. Track regressions in CI.
3. **Edge cases**: Grayscale images, RGBA with transparency, 1x1 images, 16-bit PNG, animated GIF (first frame), EXIF rotation.
4. **Platform**: Test on Linux (x86_64), macOS (ARM64), Windows (x86_64).

## Dependencies to Vendor vs. Link

| Dependency | Strategy | Reason |
|-----------|---------|--------|
| libjpeg-turbo | Bundle via `image` crate | Avoid system dependency |
| libpng | Bundle via `image` crate | Avoid system dependency |
| libwebp | Bundle via `image` crate | Avoid system dependency |
| NVJPEG | Dynamic link (optional) | Only for CUDA users, too large to bundle |
| rayon | Cargo dependency | Pure Rust, no system deps |
| fast_image_resize | Cargo dependency | Pure Rust + SIMD intrinsics |

Goal: `pip install tensorimage` should work with zero system dependencies on all platforms.
