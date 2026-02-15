# Development Roadmap

## Phase 1: Core decode + resize ✅

`ti.load("image.jpg", size=512)` returns a numpy array, 5x+ faster than PIL.

- [x] JPEG decode via libjpeg-turbo with IDCT scaling
- [x] PNG decode via image crate
- [x] SIMD resize (Lanczos3, bilinear, nearest, CatmullRom, Mitchell)
- [x] Shortest-edge resize matching torchvision `Resize(size)` semantics
- [x] RGBA/grayscale auto-conversion to RGB
- [x] Zero-copy numpy output
- [x] GIL released during all Rust work
- [x] 20 correctness tests including pixel-level comparison vs PIL

## Phase 2: Fused pipeline + batch loading ✅

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

## Phase 3: torchvision.transforms compatibility + smart Compose ✅

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

## Phase 4: Performance optimizations ✅

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

## Phase 5: PyTorch tensor output + DLPack interop ✅

`ti.load("img.jpg", device="cpu")` — zero-copy torch.Tensor output.

- [x] `device` parameter on `load()` and `load_batch()` — `"cpu"` (zero-copy), `"cuda"` (H2D transfer)
- [x] Zero-copy `torch.from_numpy()` — removed wasteful `.copy()` calls in transforms
- [x] `to_dlpack()` utility for framework-agnostic interop (JAX, TensorFlow, etc.)
- [x] Full backward compatibility — `device=None` returns numpy arrays
- [x] 23 new tests for device parameter, DLPack, and zero-copy verification

## Phase 6: GPU decode (NVJPEG)

End-to-end GPU pipeline: file → CUDA tensor with no CPU→GPU copies. `pip install tensorimage[cuda]`.

- [ ] NVJPEG decode via `nvjpeg` C API (CUDA GPU JPEG decode)
- [ ] GPU resize via CUDA kernels (or cuDNN)
- [ ] End-to-end GPU pipeline: file → CUDA tensor with no CPU→GPU copies
- [ ] Optional feature flag: `pip install tensorimage[cuda]`
- [ ] Fallback to CPU path when CUDA unavailable

## Phase 7: Smart dataset filtering ✅

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

## Phase 8: Extended format support + real-world robustness

`ti.load_bytes(data, size=224)` — load from bytes (S3, HTTP). WebP/AVIF decode. EXIF auto-rotation.

- [ ] Enable WebP and AVIF features in `image` crate (adds decode support via existing fallback path)
- [ ] Optimized WebP decode via `libwebp` (direct C bindings, skip `image` crate overhead)
- [ ] EXIF orientation auto-correction: read tag from JPEG, rotate/flip before returning
- [ ] `ti.load_bytes(data, ...)` Python API — same parameters as `ti.load()` but accepts `bytes`
- [ ] `ti.load_batch_bytes(data_list, ...)` for parallel bytes decoding
- [ ] `ti.image_info(path)` single-file Python API (complement to existing batch version)
- [ ] New Rust dependency: `kamadak-exif` (lightweight EXIF parser)

## Phase 9: Rust-accelerated augmentations

Rust-backed GaussianBlur, affine transforms, and more — the missing 60% of preprocessing time.

- [ ] **GaussianBlur** — separable 2D convolution in Rust (SIMD-friendly O(n*k) per axis)
- [ ] **RandomRotation** — bilinear-interpolated affine warp in Rust
- [ ] **RandomAffine** — rotation + translation + scale + shear, single resampling pass
- [ ] **RandomPerspective** — 3x3 homography warp in Rust
- [ ] **RandomErasing** / Cutout — zero-fill or noise-fill random rectangle (Python/numpy)
- [ ] **Grayscale** / **RandomGrayscale** — weighted channel average (Python/numpy)
- [ ] **GaussianNoise** — additive Gaussian noise (Python/numpy)
- [ ] **Pad** — constant/reflect/replicate padding (Python/numpy)
- [ ] **ElasticTransform** — grid-based displacement (Python/numpy)
- [ ] New module: `crates/tensorimage-core/src/augment.rs`
- [ ] New module: `crates/tensorimage-python/src/augment.rs` (PyO3 bindings)
- [ ] All exposed via `tensorimage.transforms`, same API patterns as torchvision

## Phase 10: PyTorch Dataset & DataLoader integration

`ti.ImageFolder(root, transform)` — drop-in `torchvision.datasets.ImageFolder` replacement.

- [ ] `ti.ImageFolder(root, transform=None)` — walks directory, class labels from subdirectory names
- [ ] `ti.ImageDataset(paths, labels=None, transform=None)` — generic dataset from path list
- [ ] `ti.collate_fn` — optimized collation using `ti.load_batch()` under the hood
- [ ] Integration with `torch.utils.data.DataLoader` (`num_workers`, `prefetch_factor`)
- [ ] Pure Python — no Rust changes, uses existing `ti.load()` and `ti.load_batch()`

## Phase 11: Streaming I/O & large-scale datasets

`ti.TarDataset(shards)` — sequential I/O for ImageNet/LAION-scale training. HTTP/URL loading.

- [ ] TAR shard reader in Rust — sequential `.tar` read, parallel image decode within
- [ ] `ti.load_tar(path, transform=None)` → iterator of (filename, array) pairs
- [ ] `ti.TarDataset(shard_paths, transform=None)` → PyTorch `IterableDataset`
- [ ] WebDataset compatibility — read `.tar` shards following WebDataset naming convention
- [ ] `ti.load_url(url)` — HTTP fetch + decode via `ureq` (lightweight Rust HTTP client)
- [ ] `ti.load_urls(url_list, workers=None)` — parallel fetch + decode
- [ ] Memory-mapped files via `mmap` for large local files
- [ ] New Rust dependencies: `ureq` (HTTP), `tar` (archive reading)

## Phase 12: Video frame extraction

`ti.load_video_frames(path, fps=1)` — fast frame extraction via FFmpeg. Fills the TorchVision deprecation gap.

- [ ] Video decode via `ffmpeg-next` Rust crate (FFmpeg C API bindings)
- [ ] `ti.load_video_frames(path, indices=None, fps=None, transform=None)` — extract specific frames or uniform sampling
- [ ] `ti.video_info(path)` — metadata: duration, fps, resolution, frame count
- [ ] Keyframe-only extraction mode (fast seeking, no full decode)
- [ ] Batch frame loading with parallel decode
- [ ] Returns numpy arrays or torch tensors (same `device` parameter pattern)
- [ ] New Rust dependency: `ffmpeg-next`
