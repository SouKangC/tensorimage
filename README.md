# tensorimage

Fast image loading for Python. Built in Rust.

A drop-in replacement for `PIL.Image.open()` that decodes, resizes, crops, and normalizes images **6x+ faster** using libjpeg-turbo SIMD decoding, IDCT downscaling, and hardware-accelerated resize. Returns numpy arrays with zero-copy. Includes parallel batch loading, a torchvision.transforms replacement, and Rust-backed dataset filtering.

## Features

- **6.6x faster** than PIL for full ML pipelines (resize + crop + normalize)
- **47x faster** batch loading via parallel rayon thread pool
- **Drop-in torchvision.transforms** replacement with auto-fused Rust fast-path
- **Zero-copy** numpy output and PyTorch tensor support (`device="cpu"` / `"cuda"`)
- **Dataset filtering** with perceptual hash dedup (3-27x faster than imagehash) and CLIP aesthetic scoring
- JPEG + PNG, RGBA/grayscale auto-converted to RGB

## Installation

```bash
pip install tensorimage
```

## Quickstart

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

No `Image.open()`, no `.convert("RGB")`, no manual normalize+transpose. One call, one tensor.

### PyTorch integration

```python
import tensorimage as ti

# Load directly as a PyTorch CPU tensor (zero-copy from numpy)
tensor = ti.load("photo.jpg", size=224, crop="center", normalize="imagenet", device="cpu")

# Load to GPU
tensor = ti.load("photo.jpg", size=224, crop="center", normalize="imagenet", device="cuda")

# Batch loading to GPU
batch = ti.load_batch(paths, size=224, crop="center", normalize="imagenet", device="cuda")

# DLPack interop — framework-agnostic zero-copy (JAX, TensorFlow, etc.)
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

Supports: `Compose`, `Resize`, `CenterCrop`, `RandomCrop`, `ToTensor`, `Normalize`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `ColorJitter`. Resize uses the Rust SIMD backend. torch is optional -- `ToTensor` returns `torch.Tensor` if available, numpy otherwise. `Compose` auto-detects common patterns and fuses operations in Rust for extra speed.

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

Filters are applied cheapest-first: dimension check (header-only, no decode) -> perceptual hash dedup (Rust parallel) -> aesthetic scoring (CLIP, only if `min_aesthetic` is set). No torch dependency for hash-only workflows.

## Benchmarks

All benchmarks on Apple M4, 100 iterations.

### Image loading (vs PIL + numpy)

4000x2000 JPEG -> 512px shortest edge:

| Task | tensorimage | PIL + numpy | Speedup |
|---|---|---|---|
| Resize only | **7.7 ms** | 41.6 ms | **5.4x** |
| Full pipeline (resize + crop + normalize) | **6.5 ms** | 43.0 ms | **6.6x** |
| Batch (8 images, 4 workers) | **7.3 ms** | 343.8 ms | **47x** |

### Transforms (vs torchvision.transforms)

Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize, 4000x2000 JPEG:

| Pipeline | tensorimage | torchvision | Speedup |
|---|---|---|---|
| From numpy (fused ToTensor+Normalize) | **3.2 ms** | 10.4 ms | **3.2x** |
| End-to-end file -> tensor (fast-path) | **3.5 ms** | 18.2 ms | **5.2x** |

### Perceptual hashing (vs imagehash)

| Task | tensorimage | imagehash | Speedup |
|---|---|---|---|
| dHash (1920x1080 JPEG) | **0.97 ms** | 3.05 ms | **3.1x** |
| pHash (1920x1080 JPEG) | **1.11 ms** | 30.00 ms | **27x** |
| dHash (4000x2000 JPEG) | **2.83 ms** | 10.24 ms | **3.6x** |
| pHash (4000x2000 JPEG) | **3.05 ms** | 10.87 ms | **3.6x** |
| dHash batch (8 images, parallel) | **4.06 ms** | 53.49 ms | **13.2x** |

pHash is especially fast because tensorimage uses IDCT-scaled JPEG decode (decodes directly at ~32px instead of full resolution) plus a hand-rolled DCT in Rust. Batch adds ~3-4x via rayon parallelism. Header-only dimension reads run at 0.015 ms/image.

## API Reference

### `ti.load(path, size=None, algorithm=None, crop=None, normalize=None, device=None)`

Load an image file and return a numpy array or torch.Tensor.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | required | Image file path (JPEG or PNG) |
| `size` | `int` | `None` | Target shortest edge size. Preserves aspect ratio. |
| `algorithm` | `str` | `"lanczos3"` | `"nearest"`, `"bilinear"`, `"catmullrom"`, `"mitchell"`, `"lanczos3"` |
| `crop` | `str` | `None` | `"center"` = center crop to `size x size`. Requires `size`. |
| `normalize` | `str` | `None` | `"imagenet"`, `"clip"`, or `"[-1,1]"`. Outputs `float32` CHW. |
| `device` | `str` | `None` | `None` = numpy, `"cpu"` = zero-copy torch, `"cuda"` = GPU tensor |

**Returns:** `ndarray` `(H, W, 3)` uint8 without normalize, `(3, H, W)` float32 with normalize. `torch.Tensor` if `device` is set.

### `ti.load_batch(paths, size=None, algorithm=None, crop=None, normalize=None, workers=None, device=None)`

Load multiple images in parallel. Same parameters as `load()`, plus `workers` (default: CPU count).

**Returns:** With `normalize` + `crop`: contiguous `(N, 3, H, W)` float32. Otherwise: list of individual arrays.

### `ti.phash(path_or_array, algorithm="dhash")`

Compute a 64-bit perceptual hash. Accepts a file path (`str`) or numpy array `(H, W, 3)` uint8.

| Algorithm | Description |
|---|---|
| `"dhash"` | Difference hash. Fast, compares adjacent pixels. |
| `"phash"` | Perceptual hash. More robust, uses DCT. |

### `ti.phash_batch(paths, algorithm="dhash", workers=None)`

Compute perceptual hashes for multiple images in parallel. Returns `list[int]`.

### `ti.hamming_distance(a, b)`

Hamming distance (number of differing bits) between two 64-bit hashes. Returns `int` (0 = identical, 64 = maximally different).

### `ti.deduplicate(paths, algorithm="dhash", threshold=None, workers=None)`

Find and group near-duplicate images by perceptual hash.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `paths` | `list[str]` | required | Image file paths |
| `algorithm` | `str` | `"dhash"` | `"dhash"` or `"phash"` |
| `threshold` | `int` | `0` / `10` | Max Hamming distance to consider as duplicate |
| `workers` | `int` | CPU count | Worker threads |

**Returns:** `dict` with `"keep_indices"` (first of each group), `"duplicate_groups"` (groups with 2+ members), `"hashes"` (all hashes).

### `ti.filter_dataset(paths, ...)`

High-level dataset filtering pipeline. Applies filters cheapest-first.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `paths` | `list[str]` | required | Image file paths |
| `min_width` | `int` | `None` | Remove images narrower than this |
| `min_height` | `int` | `None` | Remove images shorter than this |
| `deduplicate` | `bool` | `True` | Remove near-duplicates |
| `hash_algorithm` | `str` | `"dhash"` | `"dhash"` or `"phash"` |
| `hash_threshold` | `int` | `None` | Hamming distance threshold for dedup |
| `min_aesthetic` | `float` | `None` | Min CLIP aesthetic score (1-10). Requires `torch` + `open_clip_torch`. |
| `workers` | `int` | CPU count | Worker threads |
| `verbose` | `bool` | `False` | Print progress |

**Returns:** `dict` with `"paths"` (surviving paths), `"indices"` (original indices), `"stats"` (counts per stage).

### `ti.to_dlpack(array)`

Export a numpy array or torch tensor via DLPack for framework-agnostic interop.

## How it works

```
JPEG file on disk
  |  std::fs::read (single syscall)
  v
Raw bytes in memory
  |  libjpeg-turbo IDCT scaling (decode at 1/4 resolution if target is small)
  v
RGB pixels at reduced resolution
  |  fused resize+crop: SIMD Lanczos with source-space crop (single resampling pass)
  v
RGB pixels at crop size (e.g., 224x224)
  |  fused normalize + HWC->CHW transpose (single pass, pre-computed scale/bias)
  v
float32 pixels in CHW layout
  |  PyArray::from_vec (zero-copy ownership transfer to numpy)
  v
numpy.ndarray (3, H, W) float32
```

The GIL is released during all Rust work. Batch loading runs images in parallel via a persistent rayon thread pool.

Key optimizations: fat LTO + `target-cpu=native` for cross-crate SIMD inlining, fused resize+crop in a single resampling pass, persistent thread pool via `OnceLock`, and contiguous batch output (`[N,3,H,W]` pre-allocated, each worker writes to its slice).

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

## Project structure

```
tensorimage/
├── crates/
│   ├── tensorimage-core/       # Pure Rust library
│   │   └── src/
│   │       ├── decode.rs       # JPEG (turbojpeg) + PNG (image crate) decoding
│   │       ├── resize.rs       # SIMD resize via fast_image_resize
│   │       ├── crop.rs         # Center crop
│   │       ├── normalize.rs    # Fused normalize + HWC->CHW transpose
│   │       ├── pipeline.rs     # Chained decode->resize->crop->normalize
│   │       ├── batch.rs        # Parallel batch loading via rayon
│   │       ├── pool.rs         # Shared rayon thread pool
│   │       ├── phash.rs        # Perceptual hashing (dHash, pHash)
│   │       ├── dedup.rs        # Parallel deduplication
│   │       ├── jpeg_info.rs    # Header-only dimension read
│   │       └── error.rs        # Error types
│   └── tensorimage-python/     # PyO3 bindings
│       └── src/
│           ├── lib.rs          # Python module definition
│           ├── load.rs         # Image loading bindings
│           └── hash.rs         # Hash and dedup bindings
├── python/tensorimage/         # Python package
│   ├── __init__.py             # Public API
│   ├── transforms.py           # torchvision.transforms replacement
│   └── aesthetic.py            # CLIP aesthetic scoring (optional)
├── tests/                      # 166 tests
└── benches/                    # Benchmarks vs PIL and torchvision
```

## Roadmap

Phases 1-5 and 7 are complete. Upcoming:

- **Phase 8**: Extended formats (WebP, AVIF), EXIF auto-rotation, `load_bytes()` API for S3/HTTP workflows
- **Phase 9**: Rust-accelerated augmentations (GaussianBlur, RandomRotation, RandomAffine, and more)
- **Phase 10**: PyTorch Dataset & DataLoader integration (`ImageFolder`, `ImageDataset`, optimized collation)
- **Phase 6**: GPU decode via NVJPEG — end-to-end CUDA pipeline
- **Phase 11**: Streaming I/O (TAR shards, WebDataset, HTTP/URL loading) for large-scale training
- **Phase 12**: Video frame extraction via FFmpeg

See [PLAN.md](PLAN.md) for the detailed development roadmap and phase history.

## License

MIT
