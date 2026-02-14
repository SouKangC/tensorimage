# tensorimage

Fast image loading for Python. Built in Rust.

A drop-in replacement for `PIL.Image.open()` that decodes and resizes images **5x+ faster** using libjpeg-turbo SIMD decoding, IDCT downscaling, and hardware-accelerated resize. Returns numpy arrays with zero-copy.

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

# Choose a resize algorithm
img = ti.load("photo.jpg", size=256, algorithm="bilinear")
```

That's it. No `Image.open()`, no `.convert("RGB")`, no `np.array()` wrapper. One call, one array.

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

4000x2000 JPEG resized to 512px shortest edge, 100 iterations, Apple M4:

| Library | Median | Speedup |
|---|---|---|
| PIL (Pillow) | 41.4 ms | 1x |
| **tensorimage** | **8.0 ms** | **5.2x** |

## API

### `ti.load(path, size=None, algorithm=None)`

Load an image file and return a `numpy.ndarray` with shape `(H, W, 3)` and dtype `uint8`.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Path to image file (JPEG or PNG) |
| `size` | `int` or `None` | Target shortest edge size. Preserves aspect ratio. `None` = no resize. |
| `algorithm` | `str` or `None` | Resize algorithm. Default: `"lanczos3"` |

**Supported algorithms:** `nearest`, `bilinear`, `catmullrom`, `mitchell`, `lanczos3`

**Supported formats:** JPEG, PNG. RGBA and grayscale images are automatically converted to RGB.

**Raises:** `ValueError` on file not found, corrupt image, or invalid algorithm name.

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

# Fast nearest-neighbor for previews
thumb = ti.load("photo.jpg", size=128, algorithm="nearest")

# Works with any framework
import torch
tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
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
  |  PyArray::from_vec (zero-copy ownership transfer to numpy)
  v
numpy.ndarray (H, W, 3) uint8
```

The GIL is released during decode and resize, so other Python threads can run concurrently.

## Project structure

```
tensorimage/
├── crates/
│   ├── tensorimage-core/       # Pure Rust — decode, resize, error handling
│   │   └── src/
│   │       ├── decode.rs       # JPEG (turbojpeg) + PNG (image crate) decoding
│   │       ├── resize.rs       # SIMD resize via fast_image_resize
│   │       └── error.rs        # Error types
│   └── tensorimage-python/     # PyO3 bindings
│       └── src/
│           ├── lib.rs          # Python module definition
│           └── load.rs         # load() function with GIL release + zero-copy
├── python/tensorimage/         # Python package
│   └── __init__.py             # Re-exports load() from Rust
├── tests/
│   └── test_decode.py          # 20 tests: decode, resize, color conversion, errors
└── benches/
    └── compare.py              # Benchmark vs PIL
```

## Roadmap

### Phase 1: Core decode + resize (current)

`ti.load("image.jpg", size=512)` returns a numpy array, 5x+ faster than PIL.

- [x] JPEG decode via libjpeg-turbo with IDCT scaling
- [x] PNG decode via image crate
- [x] SIMD resize (Lanczos3, bilinear, nearest, CatmullRom, Mitchell)
- [x] Shortest-edge resize matching torchvision `Resize(size)` semantics
- [x] RGBA/grayscale auto-conversion to RGB
- [x] Zero-copy numpy output
- [x] GIL released during all Rust work
- [x] 20 correctness tests including pixel-level comparison vs PIL

### Phase 2: Fused pipeline + batch loading

Single fused pipeline, batch parallel loading.

- `ti.load("img.jpg", size=512, crop="center", normalize="imagenet")`
- Fused operations (resize+crop in one pass, avoid intermediate allocations)
- Batch loading with rayon: `ti.load_batch(paths, workers=8)`
- Normalize presets: `"imagenet"`, `"[-1,1]"`, `"clip"`, custom mean/std

### Phase 3: torchvision.transforms compatibility

Drop-in replacement — change one import line.

```python
# from torchvision import transforms
from tensorimage import transforms  # same API, 10x faster
```

- `Compose`, `Resize`, `CenterCrop`, `RandomCrop`, `ToTensor`, `Normalize`
- `RandomHorizontalFlip`, `RandomVerticalFlip`, `ColorJitter`
- Pixel-exact output matching vs torchvision

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
