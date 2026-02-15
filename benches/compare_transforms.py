"""Benchmark: tensorimage.transforms vs torchvision.transforms.

Standard ImageNet pipeline: Resize(256) → CenterCrop(224) → ToTensor → Normalize.
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


def bench_tensorimage(pil_img, iterations):
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


def main():
    print(f"Loading {FIXTURE}...")
    pil_img = Image.open(FIXTURE).convert("RGB")
    print(f"Image size: {pil_img.size[0]}x{pil_img.size[1]}")
    print(f"Pipeline: Resize(256) → CenterCrop(224) → ToTensor → Normalize(imagenet)")
    print(f"Iterations: {ITERATIONS}")
    print()

    ti_times, ti_out = bench_tensorimage(pil_img, ITERATIONS)
    ti_median = statistics.median(ti_times) * 1000

    print(f"tensorimage.transforms:  {ti_median:.2f} ms (median)")

    tv_times, tv_out = bench_torchvision(pil_img, ITERATIONS)
    if tv_times is not None:
        tv_median = statistics.median(tv_times) * 1000
        speedup = tv_median / ti_median
        print(f"torchvision.transforms:  {tv_median:.2f} ms (median)")
        print(f"Speedup:                 {speedup:.1f}x")

        # Verify outputs are close
        ti_np = ti_out.numpy() if hasattr(ti_out, "numpy") else ti_out
        tv_np = tv_out.numpy()
        max_diff = np.abs(ti_np - tv_np).max()
        print(f"\nMax output diff:         {max_diff:.4f}")
    else:
        print("torchvision not installed — skipping comparison")


if __name__ == "__main__":
    main()
