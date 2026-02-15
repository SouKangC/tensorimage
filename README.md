# tensorimage

Fast image loading for Python. Built in Rust.

A drop-in replacement for `PIL.Image.open()` that decodes, resizes, crops, and normalizes images **6x+ faster** using libjpeg-turbo SIMD decoding, IDCT downscaling, and hardware-accelerated resize. Returns numpy arrays with zero-copy. Includes fused pipeline and parallel batch loading via rayon.

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

### PyTorch integration

```python
import tensorimage as ti

# Load directly as a PyTorch CPU tensor (zero-copy from numpy)
tensor = ti.load("photo.jpg", size=224, crop="center", normalize="imagenet", device="cpu")
# tensor.shape = (3, 224, 224), dtype=torch.float32

# Load to GPU (CUDA)
tensor = ti.load("photo.jpg", size=224, crop="center", normalize="imagenet", device="cuda")

# Batch loading to GPU
batch = ti.load_batch(paths, size=224, crop="center", normalize="imagenet", device="cuda")

# DLPack interop — framework-agnostic zero-copy
arr = ti.load("photo.jpg", size=224)
import torch
tensor = torch.from_dlpack(ti.to_dlpack(arr))
```

`device=None` (default) returns numpy arrays for full backward compatibility.

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

### Dataset filtering

```python
import tensorimage as ti

# Perceptual hashing — fast, computed in Rust
h = ti.phash("photo.jpg")                              # 64-bit dHash
h = ti.phash("photo.jpg", algorithm="phash")           # 64-bit pHash (more robust)
hashes = ti.phash_batch(paths, algorithm="dhash")       # parallel batch hashing
dist = ti.hamming_distance(h1, h2)                     # Hamming distance between hashes

# Deduplication — find and group near-duplicate images
result = ti.deduplicate(paths, algorithm="dhash", threshold=5)
# result = {"keep_indices": [0, 3, 7], "duplicate_groups": [[0, 1, 2], ...], "hashes": [...]}
unique_paths = [paths[i] for i in result["keep_indices"]]

# Full pipeline — dimension filter + dedup + optional aesthetic scoring
result = ti.filter_dataset(
    paths,
    min_width=512,              # remove undersized images
    min_height=512,
    deduplicate=True,           # remove near-duplicates (default)
    hash_algorithm="dhash",     # "dhash" (fast) or "phash" (robust)
    hash_threshold=5,           # Hamming distance threshold
    min_aesthetic=5.0,          # CLIP aesthetic score (requires torch + open_clip)
    verbose=True,               # print progress
)
clean_paths = result["paths"]
print(result["stats"])  # {"total": 1000, "dimension_removed": 50, "duplicate_removed": 120, ...}
```

Filters are applied cheapest-first: dimension check (header-only, no decode) → perceptual hash dedup (Rust parallel) → aesthetic scoring (CLIP, only if `min_aesthetic` is set). No torch dependency for hash-only workflows.

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
| Resize only | **7.7 ms** | 41.6 ms | **5.4x** |
| Full pipeline (resize + crop + normalize) | **6.5 ms** | 43.0 ms | **6.6x** |
| Batch (8 images, 4 workers) | **7.3 ms** | 343.8 ms | **47x** |

Transforms pipeline (Resize(256) → CenterCrop(224) → ToTensor → Normalize), 4000x2000 JPEG, 100 iterations, Apple M4:

| Pipeline | tensorimage.transforms | torchvision.transforms | Speedup |
|---|---|---|---|
| From numpy (fused ToTensor+Normalize) | **3.2 ms** | 10.4 ms | **3.2x** |
| End-to-end file → tensor (fast-path) | **3.5 ms** | 18.2 ms | **5.2x** |

Phase 4 optimizations: fat LTO + `target-cpu=native` enables cross-crate SIMD inlining; fused resize+crop eliminates an intermediate buffer by computing source-space crop coordinates in a single resampling pass; persistent rayon thread pool + contiguous batch output (`[N,3,H,W]` pre-allocated, each worker writes to its slice) cut batch overhead dramatically.

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

### `ti.phash(path_or_array, algorithm="dhash")`

Compute a 64-bit perceptual hash of an image.

| Parameter | Type | Description |
|---|---|---|
| `path_or_array` | `str` or `ndarray` | Image file path or numpy uint8 array (H, W, 3) |
| `algorithm` | `str` | `"dhash"` (default, fast) or `"phash"` (robust, uses DCT) |

**Returns:** `int` — 64-bit perceptual hash.

### `ti.phash_batch(paths, algorithm="dhash", workers=None)`

Compute perceptual hashes for multiple images in parallel.

### `ti.hamming_distance(a, b)`

Compute Hamming distance (number of differing bits) between two 64-bit hashes.

### `ti.deduplicate(paths, algorithm="dhash", threshold=None, workers=None)`

Find and group near-duplicate images by perceptual hash.

| Parameter | Type | Description |
|---|---|---|
| `paths` | `list[str]` | List of image file paths |
| `algorithm` | `str` | `"dhash"` or `"phash"` |
| `threshold` | `int` or `None` | Max Hamming distance. Default: 0 (dhash) / 10 (phash) |
| `workers` | `int` or `None` | Number of worker threads |

**Returns:** `dict` with `"keep_indices"`, `"duplicate_groups"`, `"hashes"`.

### `ti.filter_dataset(paths, ...)`

High-level dataset filtering: dimension check → dedup → aesthetic scoring.

| Parameter | Type | Description |
|---|---|---|
| `paths` | `list[str]` | Image file paths |
| `min_width` | `int` or `None` | Minimum width filter |
| `min_height` | `int` or `None` | Minimum height filter |
| `deduplicate` | `bool` | Remove near-duplicates (default: `True`) |
| `hash_algorithm` | `str` | `"dhash"` or `"phash"` |
| `hash_threshold` | `int` or `None` | Hamming distance threshold |
| `min_aesthetic` | `float` or `None` | Min CLIP aesthetic score (requires torch + open_clip) |
| `workers` | `int` or `None` | Worker threads |
| `verbose` | `bool` | Print progress |

**Returns:** `dict` with `"paths"`, `"indices"`, `"stats"`.

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
  |  fused resize+crop: SIMD Lanczos with source-space crop (single resampling pass)
  v
RGB pixels at crop size (e.g., 224×224)
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
│   ├── tensorimage-core/       # Pure Rust — decode, resize, crop, normalize, pipeline, batch, hash, dedup
│   │   └── src/
│   │       ├── decode.rs       # JPEG (turbojpeg) + PNG (image crate) decoding
│   │       ├── resize.rs       # SIMD resize via fast_image_resize
│   │       ├── crop.rs         # Center crop
│   │       ├── normalize.rs    # Fused normalize + HWC→CHW transpose
│   │       ├── pipeline.rs     # Chained decode→resize→crop→normalize
│   │       ├── batch.rs        # Parallel batch loading via rayon
│   │       ├── pool.rs         # Shared rayon thread pool (Phase 7)
│   │       ├── phash.rs        # Perceptual hashing: dHash, pHash (Phase 7)
│   │       ├── dedup.rs        # Parallel deduplication by hash (Phase 7)
│   │       ├── jpeg_info.rs    # Fast header-only dimension read (Phase 7)
│   │       └── error.rs        # Error types
│   └── tensorimage-python/     # PyO3 bindings
│       └── src/
│           ├── lib.rs          # Python module definition
│           ├── load.rs         # load(), load_batch(), _resize_array, _to_tensor_normalize, _load_pipeline
│           └── hash.rs         # phash, deduplicate, image_info bindings (Phase 7)
├── python/tensorimage/         # Python package
│   ├── __init__.py             # Re-exports from Rust + phash/deduplicate/filter_dataset
│   ├── transforms.py           # Drop-in torchvision.transforms replacement (Phase 3)
│   └── aesthetic.py            # CLIP aesthetic scoring (Phase 7, optional torch+open_clip)
├── tests/
│   ├── test_decode.py          # 45 tests: decode, resize, crop, normalize, pipeline, batch
│   ├── test_transforms.py      # 67 tests: transforms + fused optimizations (Phase 3)
│   ├── test_tensor_output.py   # 23 tests: device parameter, DLPack, zero-copy (Phase 5)
│   └── test_filter.py          # 31 tests: phash, dedup, filter_dataset (Phase 7)
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

### Phase 4: Performance optimizations ✅

Fat LTO, fused resize+crop, zero-copy bindings, persistent thread pool.

- [x] Release profile: `lto = "fat"`, `codegen-units = 1` for cross-crate SIMD inlining
- [x] `target-cpu=native` for full NEON (Apple Silicon) / AVX2 (x86) instruction selection
- [x] Fused resize+crop: single resampling pass via source-space crop coordinates (eliminates intermediate buffer)
- [x] Identity resize skip when IDCT scaling already hits target dimensions
- [x] Persistent rayon thread pool via `OnceLock` (eliminates per-batch thread spawn)
- [x] Contiguous batch output: pre-allocate `[N,3,H,W]` buffer, each worker writes directly to its slice
- [x] Slice-based normalize in PyO3 bindings (avoids `.to_vec()` copy from numpy)
- [x] `resize_exact_borrowed` using `Image::from_slice_u8` for borrowed resize path
- [x] Pipeline 6.6x vs PIL (was 5.2x), batch 47x vs PIL (was 20x), end-to-end 5.2x vs torchvision (was 4.4x)

### Phase 5: PyTorch tensor output + DLPack interop ✅

`ti.load("img.jpg", device="cpu")` — zero-copy torch.Tensor output.

- [x] `device` parameter on `load()` and `load_batch()` — `"cpu"` (zero-copy), `"cuda"` (H2D transfer)
- [x] Zero-copy `torch.from_numpy()` — removed wasteful `.copy()` calls in transforms
- [x] `to_dlpack()` utility for framework-agnostic interop (JAX, TensorFlow, etc.)
- [x] Full backward compatibility — `device=None` returns numpy arrays
- [x] 23 new tests for device parameter, DLPack, and zero-copy verification

### Phase 6: GPU decode (NVJPEG)

- NVJPEG decode (CUDA GPU JPEG decode)
- GPU resize via CUDA kernels
- End-to-end GPU pipeline: file → CUDA tensor with no CPU copies

### Phase 7: Smart dataset filtering ✅

`ti.filter_dataset(paths, deduplicate=True, min_aesthetic=5.0)` — curate ML datasets in one call.

- [x] Perceptual hashing in Rust: dHash (fast) and pHash (DCT-based, robust)
- [x] Parallel batch hashing via shared rayon thread pool
- [x] Greedy deduplication by Hamming distance threshold
- [x] Fast header-only dimension read (JPEG: turbojpeg, PNG: IHDR chunk)
- [x] `ti.phash()`, `ti.phash_batch()`, `ti.hamming_distance()` — low-level hash API
- [x] `ti.deduplicate()` — find and group near-duplicate images
- [x] `ti.filter_dataset()` — high-level pipeline: dimension filter → dedup → aesthetic scoring
- [x] CLIP aesthetic scoring via `AestheticScorer` class (optional torch + open_clip)
- [x] No new Rust dependencies — hand-rolled DCT, area-average resize, BT.601 grayscale
- [x] 31 tests (28 pass without torch, 3 skipped for aesthetic scoring)

## License

MIT
