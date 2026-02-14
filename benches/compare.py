"""Benchmark: tensorimage vs PIL for load+resize, pipeline, and batch."""
import argparse
import os
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


def bench_ti_pipeline(path, size, n):
    """Benchmark full pipeline: resize + center crop + imagenet normalize."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        arr = ti.load(path, size=size, crop="center", normalize="imagenet")
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_pil_pipeline(path, size, n):
    """Benchmark PIL equivalent: resize + center crop + normalize to CHW."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
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
        # Center crop
        w2, h2 = img.size
        left = (w2 - size) // 2
        top = (h2 - size) // 2
        img = img.crop((left, top, left + size, top + size))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_ti_batch(paths, size, n, workers):
    """Benchmark ti.load_batch."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = ti.load_batch(paths, size=size, crop="center", normalize="imagenet", workers=workers)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_pil_sequential(paths, size, n):
    """Benchmark sequential PIL for the same batch."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        results = []
        for path in paths:
            img = Image.open(path)
            w, h = img.size
            if w < h:
                new_w = size
                new_h = round(h * size / w)
            else:
                new_h = size
                new_w = round(w * size / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            w2, h2 = img.size
            left = (w2 - size) // 2
            top = (h2 - size) // 2
            img = img.crop((left, top, left + size, top + size))
            arr = np.array(img).astype(np.float32) / 255.0
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1)
            results.append(arr)
        batch = np.stack(results)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def median_ms(times):
    return sorted(times)[len(times) // 2] * 1000


def main():
    parser = argparse.ArgumentParser(description="Benchmark tensorimage vs PIL")
    parser.add_argument("image", help="Path to test image")
    parser.add_argument("--size", type=int, default=512, help="Target shortest edge size")
    parser.add_argument("-n", type=int, default=100, help="Number of iterations")
    parser.add_argument("--workers", type=int, default=4, help="Number of batch workers")
    args = parser.parse_args()

    print(f"Image: {args.image}")
    print(f"Target size: {args.size}")
    print(f"Iterations: {args.n}")
    print()

    # Warmup
    ti.load(args.image, size=args.size)
    Image.open(args.image).resize((args.size, args.size), Image.LANCZOS)

    # --- Resize benchmark ---
    print("=== Resize Only ===")
    ti_times = bench_ti(args.image, args.size, args.n)
    pil_times = bench_pil(args.image, args.size, args.n)
    ti_ms = median_ms(ti_times)
    pil_ms = median_ms(pil_times)
    print(f"tensorimage: {ti_ms:.2f} ms (median)")
    print(f"PIL:         {pil_ms:.2f} ms (median)")
    print(f"Speedup:     {pil_ms / ti_ms:.1f}x")
    print()

    # --- Pipeline benchmark ---
    print("=== Full Pipeline (resize + crop + normalize) ===")
    ti_pipe = bench_ti_pipeline(args.image, args.size, args.n)
    pil_pipe = bench_pil_pipeline(args.image, args.size, args.n)
    ti_pipe_ms = median_ms(ti_pipe)
    pil_pipe_ms = median_ms(pil_pipe)
    print(f"tensorimage: {ti_pipe_ms:.2f} ms (median)")
    print(f"PIL+numpy:   {pil_pipe_ms:.2f} ms (median)")
    print(f"Speedup:     {pil_pipe_ms / ti_pipe_ms:.1f}x")
    print()

    # --- Batch benchmark ---
    batch_paths = [args.image] * 8
    print(f"=== Batch ({len(batch_paths)} images, {args.workers} workers) ===")
    ti.load_batch(batch_paths[:1], size=args.size, crop="center", normalize="imagenet")  # warmup
    ti_batch = bench_ti_batch(batch_paths, args.size, args.n, args.workers)
    pil_batch = bench_pil_sequential(batch_paths, args.size, args.n)
    ti_batch_ms = median_ms(ti_batch)
    pil_batch_ms = median_ms(pil_batch)
    print(f"tensorimage: {ti_batch_ms:.2f} ms (median)")
    print(f"PIL seq:     {pil_batch_ms:.2f} ms (median)")
    print(f"Speedup:     {pil_batch_ms / ti_batch_ms:.1f}x")


if __name__ == "__main__":
    main()
