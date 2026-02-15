# tensorimage

Fast image loading for Python. Built in Rust.

A drop-in replacement for `PIL.Image.open()` that decodes, resizes, crops, and normalizes images **5x+ faster** using libjpeg-turbo SIMD decoding, IDCT downscaling, and hardware-accelerated resize. Returns numpy arrays with zero-copy. Includes fused pipeline and parallel batch loading via rayon.

## Quickstart

```bash
pip install tensorimage
```

```python
import tensorimage as ti

# Load an image as a numpy array (H, W, 3) uint8
img = ti.load("photo.jpg")

# Load and resize — shortest edge becomes 512, aspect ratio preserved
img = ti.load("photo.jpg", size=512)

# Full ML pipeline — resize, center crop, normalize → f32 (3, 224, 224) ready for PyTorch
tensor = ti.load("photo.jpg", size=224, crop="center", normalize="imagenet")

# Batch loading — parallel via rayon, returns stacked (N, 3, 224, 224) ndarray
batch = ti.load_batch(paths, size=224, crop="center", normalize="imagenet")
```

That's it. No `Image.open()`, no `.convert("RGB")`, no manual normalize+transpose. One call, one tensor.

### Drop-in torchvision.transforms replacement

```python
# from torchvision import transforms
from tensorimage import transforms  # same API, faster

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = Image.open("photo.jpg")
tensor = transform(img)  # (3, 224, 224) float32
```

Supports: `Compose`, `Resize`, `CenterCrop`, `RandomCrop`, `ToTensor`, `Normalize`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `ColorJitter`. Resize uses the Rust SIMD backend. torch is optional — `ToTensor` returns `torch.Tensor` if available, numpy otherwise. `Compose` auto-detects common patterns and fuses operations in Rust for extra speed.

## Why tensorimage?

Loading and resizing images in Python is slow. A typical PIL pipeline does this:

```python
from PIL import Image
import numpy as np

img = Image.open("photo.jpg")          # decode JPEG
img = img.resize((512, 512), Image.LANCZOS)  # resize in Python
arr = np.array(img)                    # copy pixels to numpy
```

Each step allocates memory and copies data. tensorimage replaces all of it with a single Rust call that:

1. **Decodes JPEG via libjpeg-turbo** — the same C library PIL uses, but called directly with no Python overhead
2. **Scales during decode** — IDCT downscaling produces a smaller image *during* decompression, skipping millions of pixels entirely
3. **Resizes with SIMD** — NEON (Apple Silicon) / AVX2 (x86) accelerated Lanczos via `fast_image_resize`
4. **Returns a numpy array with zero copy** — Rust hands ownership of the pixel buffer directly to numpy

### Benchmark

4000x2000 JPEG → 512px shortest edge, 100 iterations, Apple M4:

| Task | tensorimage | PIL + numpy | Speedup |
|---|---|---|---|
| Resize only | **7.9 ms** | 41.0 ms | **5.2x** |
| Full pipeline (resize + crop + normalize) | **8.3 ms** | 42.6 ms | **5.2x** |
| Batch (8 images, 4 workers) | **17.3 ms** | 345.1 ms | **20x** |

Transforms pipeline (Resize(256) → CenterCrop(224) → ToTensor → Normalize), 4000x2000 JPEG, 100 iterations, Apple M4:

| Pipeline | tensorimage.transforms | torchvision.transforms | Speedup |
|---|---|---|---|
| From numpy (fused ToTensor+Normalize) | **3.2 ms** | 10.4 ms | **3.2x** |
| End-to-end file → tensor (fast-path) | **4.1 ms** | 18.1 ms | **4.4x** |

The fast-path detects `Resize → CenterCrop → ToTensor → Normalize` and routes the entire pipeline through Rust, including JPEG decode with IDCT scaling. This is especially powerful for dataloaders where the input is a file path.

## API

### `ti.load(path, size=None, algorithm=None, crop=None, normalize=None)`

Load an image file and return a numpy array.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Path to image file (JPEG or PNG) |
| `size` | `int` or `None` | Target shortest edge size. Preserves aspect ratio. `None` = no resize. |
| `algorithm` | `str` or `None` | Resize algorithm. Default: `"lanczos3"` |
| `crop` | `str` or `None` | Crop mode. `"center"` = center crop to `size x size`. Requires `size`. |
| `normalize` | `str` or `None` | Normalize preset. Converts to `float32` CHW layout. |

**Returns:**
- Without `normalize`: `uint8` array with shape `(H, W, 3)` (HWC layout)
- With `normalize`: `float32` array with shape `(3, H, W)` (CHW layout)

**Normalize presets:** `"imagenet"` (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), `"clip"`, `"[-1,1]"`

**Supported algorithms:** `nearest`, `bilinear`, `catmullrom`, `mitchell`, `lanczos3`

**Supported formats:** JPEG, PNG. RGBA and grayscale images are automatically converted to RGB.

**Raises:** `ValueError` on file not found, corrupt image, invalid algorithm name, or invalid crop/normalize option.

### `ti.load_batch(paths, size=None, algorithm=None, crop=None, normalize=None, workers=None)`

Load multiple images in parallel using a rayon thread pool.

| Parameter | Type | Description |
|---|---|---|
| `paths` | `list[str]` | List of image file paths |
| `size` | `int` or `None` | Target shortest edge size |
| `algorithm` | `str` or `None` | Resize algorithm. Default: `"lanczos3"` |
| `crop` | `str` or `None` | Crop mode. `"center"` = center crop to `size x size`. |
| `normalize` | `str` or `None` | Normalize preset |
| `workers` | `int` or `None` | Number of worker threads. Default: number of CPU cores. |

**Returns:**
- With `normalize` + `crop` (all images same size): contiguous `float32` array with shape `(N, 3, H, W)`
- Otherwise: Python list of individual arrays

### Examples

```python
import tensorimage as ti

# Basic load
img = ti.load("photo.jpg")
print(img.shape)  # (1080, 1920, 3)
print(img.dtype)  # uint8

# Resize for model input — matches torchvision Resize(512) semantics
img = ti.load("photo.jpg", size=512)
print(img.shape)  # (512, 910, 3)  — shortest edge = 512

# Full ML preprocessing pipeline — one call, ready for PyTorch
tensor = ti.load("photo.jpg", size=224, crop="center", normalize="imagenet")
print(tensor.shape)  # (3, 224, 224)
print(tensor.dtype)  # float32

# Batch loading — parallel decode+resize+crop+normalize
paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
batch = ti.load_batch(paths, size=224, crop="center", normalize="imagenet")
print(batch.shape)  # (4, 3, 224, 224)

# Fast nearest-neighbor for previews
thumb = ti.load("photo.jpg", size=128, algorithm="nearest")
```

## Building from source

Requires Rust toolchain, CMake, and NASM (for libjpeg-turbo):

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install build deps (macOS)
brew install cmake nasm

# Build and install in development mode
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release
```

## How it works

```
JPEG file on disk
  |
  |  std::fs::read (single syscall)
  v
Raw bytes in memory
  |
  |  libjpeg-turbo IDCT scaling (decode at 1/4 resolution if target is small)
  v
RGB pixels at reduced resolution
  |
  |  fast_image_resize SIMD Lanczos (NEON / AVX2)
  v
RGB pixels at exact target size
  |
  |  center_crop (single allocation, row-by-row copy)
  v
RGB pixels at crop size
  |
  |  fused normalize + HWC→CHW transpose (single pass, pre-computed scale/bias)
  v
float32 pixels in CHW layout
  |
  |  PyArray::from_vec (zero-copy ownership transfer to numpy)
  v
numpy.ndarray (3, H, W) float32   — or (H, W, 3) uint8 without normalize
```

The GIL is released during all Rust work. Batch loading runs images in parallel via a rayon thread pool.

## Project structure

```
tensorimage/
├── crates/
│   ├── tensorimage-core/       # Pure Rust — decode, resize, crop, normalize, pipeline, batch
│   │   └── src/
│   │       ├── decode.rs       # JPEG (turbojpeg) + PNG (image crate) decoding
│   │       ├── resize.rs       # SIMD resize via fast_image_resize
│   │       ├── crop.rs         # Center crop
│   │       ├── normalize.rs    # Fused normalize + HWC→CHW transpose
│   │       ├── pipeline.rs     # Chained decode→resize→crop→normalize
│   │       ├── batch.rs        # Parallel batch loading via rayon
│   │       └── error.rs        # Error types
│   └── tensorimage-python/     # PyO3 bindings
│       └── src/
│           ├── lib.rs          # Python module definition
│           └── load.rs         # load(), load_batch(), _resize_array, _to_tensor_normalize, _load_pipeline
├── python/tensorimage/         # Python package
│   ├── __init__.py             # Re-exports load(), load_batch() from Rust
│   └── transforms.py           # Drop-in torchvision.transforms replacement (Phase 3)
├── tests/
│   ├── test_decode.py          # 45 tests: decode, resize, crop, normalize, pipeline, batch
│   └── test_transforms.py      # 67 tests: transforms + fused optimizations (Phase 3)
└── benches/
    ├── compare.py              # Benchmark vs PIL (resize, pipeline, batch)
    └── compare_transforms.py   # Benchmark transforms vs torchvision.transforms
```

## Roadmap

### Phase 1: Core decode + resize ✅

`ti.load("image.jpg", size=512)` returns a numpy array, 5x+ faster than PIL.

- [x] JPEG decode via libjpeg-turbo with IDCT scaling
- [x] PNG decode via image crate
- [x] SIMD resize (Lanczos3, bilinear, nearest, CatmullRom, Mitchell)
- [x] Shortest-edge resize matching torchvision `Resize(size)` semantics
- [x] RGBA/grayscale auto-conversion to RGB
- [x] Zero-copy numpy output
- [x] GIL released during all Rust work
- [x] 20 correctness tests including pixel-level comparison vs PIL

### Phase 2: Fused pipeline + batch loading ✅

`ti.load("img.jpg", size=224, crop="center", normalize="imagenet")` returns a `float32 [3, 224, 224]` tensor ready for PyTorch.

- [x] Center crop (`crop="center"`)
- [x] Fused normalize + HWC→CHW transpose in a single pass (no intermediate allocations)
- [x] Normalize presets: `"imagenet"`, `"clip"`, `"[-1,1]"`
- [x] Chained pipeline: decode → resize → crop → normalize
- [x] Parallel batch loading via rayon: `ti.load_batch(paths, workers=8)`
- [x] Batches with uniform size stack into contiguous `[N, 3, H, W]` ndarray
- [x] Pixel-exact match vs manual numpy normalization (atol=1e-5)
- [x] Full backward compatibility with Phase 1 API
- [x] 45 tests total (20 Phase 1 + 25 Phase 2)

### Phase 3: torchvision.transforms compatibility + smart Compose ✅

Drop-in replacement — change one import line.

```python
# from torchvision import transforms
from tensorimage import transforms  # same API, faster
```

- [x] `Compose`, `Resize`, `CenterCrop`, `RandomCrop`, `ToTensor`, `Normalize`
- [x] `RandomHorizontalFlip`, `RandomVerticalFlip`, `ColorJitter`
- [x] Resize uses Rust SIMD backend via `_resize_array` binding
- [x] torch optional — `ToTensor` returns numpy if torch unavailable
- [x] Pixel-exact match for crop/flip/normalize; ≤3 pixel values for resize
- [x] Full pipeline matches torchvision within atol=0.02
- [x] Fused `ToTensor + Normalize` — single Rust pass, no intermediate float allocation (3.2x)
- [x] Full fast-path: `Resize → CenterCrop → ToTensor → Normalize` from file routes through Rust pipeline with IDCT scaling (4.4x end-to-end)
- [x] 67 tests including fused optimization validation and conditional torchvision comparison

### Phase 4: PyTorch tensor output + GPU path

`ti.load("img.jpg", device="cuda")` — decode to GPU tensor directly.

- DLPack export for zero-copy to PyTorch tensors
- Optional NVJPEG decode (CUDA GPU JPEG decode)
- GPU resize via CUDA kernels

### Phase 5: Smart dataset filtering

- Perceptual hash deduplication in Rust
- CLIP-based aesthetic scoring
- `ti.agent.filter(min_aesthetic=5.0, deduplicate=True)` for dataset curation

## License

MIT
