"""Benchmark: tensorimage.transforms vs torchvision.transforms.

Standard ImageNet pipeline: Resize(256) → CenterCrop(224) → ToTensor → Normalize.

Tests three paths:
  1. tensorimage fast-path (PIL Image with filename → full Rust pipeline)
  2. tensorimage fused (numpy input → fused ToTensor+Normalize)
  3. torchvision baseline (PIL Image input)
"""

import statistics
import time

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("PIL required for benchmark: pip install Pillow")

from tensorimage import transforms

FIXTURE = "tests/fixtures/landscape.jpg"
ITERATIONS = 100


def bench_tensorimage_fast_path(path, iterations):
    """Fast-path: string path → full Rust pipeline (IDCT + SIMD + fused)."""
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Warmup
    for _ in range(3):
        t(path)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        out = t(path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times, out


def bench_tensorimage_fused(pil_img, iterations):
    """Fused path: numpy input → SIMD Resize + fused ToTensor+Normalize."""
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = np.array(pil_img)

    # Warmup
    for _ in range(3):
        t(img)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        out = t(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times, out


def bench_torchvision(pil_img, iterations):
    try:
        import torchvision.transforms as tv_transforms
    except ImportError:
        return None, None

    t = tv_transforms.Compose([
        tv_transforms.Resize(256),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Warmup
    for _ in range(3):
        t(pil_img)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        out = t(pil_img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times, out


def bench_torchvision_from_file(path, iterations):
    """End-to-end: file → PIL decode → torchvision transforms → tensor."""
    try:
        import torchvision.transforms as tv_transforms
    except ImportError:
        return None, None

    t = tv_transforms.Compose([
        tv_transforms.Resize(256),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Warmup
    for _ in range(3):
        img = Image.open(path).convert("RGB")
        t(img)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        img = Image.open(path).convert("RGB")
        out = t(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times, out


def main():
    print(f"Loading {FIXTURE}...")
    pil_img = Image.open(FIXTURE).convert("RGB")
    print(f"Image size: {pil_img.size[0]}x{pil_img.size[1]}")
    print(f"Pipeline: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(imagenet)")
    print(f"Iterations: {ITERATIONS}")
    print()

    # --- torchvision baseline ---
    tv_times, tv_out = bench_torchvision(pil_img, ITERATIONS)
    if tv_times is not None:
        tv_median = statistics.median(tv_times) * 1000
        print(f"torchvision.transforms:     {tv_median:.2f} ms (median)")
    else:
        tv_median = None
        print("torchvision not installed -- skipping baseline")

    # --- tensorimage fused (numpy input) ---
    fused_times, fused_out = bench_tensorimage_fused(pil_img, ITERATIONS)
    fused_median = statistics.median(fused_times) * 1000
    fused_label = f"tensorimage fused (numpy): {fused_median:.2f} ms (median)"
    if tv_median:
        fused_label += f"  [{tv_median / fused_median:.1f}x]"
    print(fused_label)

    # --- tensorimage fast-path (string path → full Rust pipeline) ---
    fast_times, fast_out = bench_tensorimage_fast_path(FIXTURE, ITERATIONS)
    fast_median = statistics.median(fast_times) * 1000
    fast_label = f"tensorimage fast (file):   {fast_median:.2f} ms (median)"
    if tv_median:
        fast_label += f"  [{tv_median / fast_median:.1f}x]"
    print(fast_label)

    # --- end-to-end: file → tensor (includes decode) ---
    print()
    tv_e2e_times, _ = bench_torchvision_from_file(FIXTURE, ITERATIONS)
    if tv_e2e_times is not None:
        tv_e2e = statistics.median(tv_e2e_times) * 1000
        speedup_e2e = tv_e2e / fast_median
        print(f"End-to-end file -> tensor:")
        print(f"  PIL + torchvision:       {tv_e2e:.2f} ms")
        print(f"  tensorimage fast-path:   {fast_median:.2f} ms  [{speedup_e2e:.1f}x]")

    # --- Verify outputs ---
    print()
    if tv_out is not None:
        fused_np = fused_out.numpy() if hasattr(fused_out, "numpy") else fused_out
        tv_np = tv_out.numpy()
        fast_np = fast_out.numpy() if hasattr(fast_out, "numpy") else fast_out
        print(f"Max diff (fused vs tv):    {np.abs(fused_np - tv_np).max():.4f}")
        print(f"Max diff (fast vs tv):     {np.abs(fast_np - tv_np).max():.4f}")


if __name__ == "__main__":
    main()
