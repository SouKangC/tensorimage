"""Benchmark: tensorimage vs PIL for load+resize."""
import argparse
import time
import numpy as np
from PIL import Image

import tensorimage as ti


def bench_ti(path, size, n):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        arr = ti.load(path, size=size)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_pil(path, size, n):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        img = Image.open(path)
        w, h = img.size
        if w < h:
            new_w = size
            new_h = round(h * size / w)
        else:
            new_h = size
            new_w = round(w * size / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark tensorimage vs PIL")
    parser.add_argument("image", help="Path to test image")
    parser.add_argument("--size", type=int, default=512, help="Target shortest edge size")
    parser.add_argument("-n", type=int, default=100, help="Number of iterations")
    args = parser.parse_args()

    print(f"Image: {args.image}")
    print(f"Target size: {args.size}")
    print(f"Iterations: {args.n}")
    print()

    # Warmup
    ti.load(args.image, size=args.size)
    Image.open(args.image).resize((args.size, args.size), Image.LANCZOS)

    ti_times = bench_ti(args.image, args.size, args.n)
    pil_times = bench_pil(args.image, args.size, args.n)

    ti_median = sorted(ti_times)[len(ti_times) // 2] * 1000
    pil_median = sorted(pil_times)[len(pil_times) // 2] * 1000
    speedup = pil_median / ti_median

    print(f"tensorimage: {ti_median:.2f} ms (median)")
    print(f"PIL:         {pil_median:.2f} ms (median)")
    print(f"Speedup:     {speedup:.1f}x")


if __name__ == "__main__":
    main()
