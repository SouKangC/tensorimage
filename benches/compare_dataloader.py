"""Benchmark: tensorimage DataLoader vs torchvision DataLoader.

Compares iteration throughput over an ImageFolder dataset.
Uses real images from the test fixtures, replicated to build a dataset of
configurable size.

Usage:
    python benches/compare_dataloader.py [--n-images 128] [--batch-size 32] [--epochs 5]
"""
import argparse
import os
import shutil
import tempfile
import time

import numpy as np

try:
    import torch
    import torchvision
    import torchvision.transforms as tv_transforms
    import torchvision.datasets as tv_datasets
    HAS_TV = True
except ImportError:
    HAS_TV = False

import tensorimage as ti
from tensorimage import transforms as ti_transforms
from tensorimage.data import ImageFolder, create_dataloader


FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
SOURCE_IMAGES = ["sample.jpg", "landscape.jpg", "portrait.jpg"]


def build_dataset(root, n_images, n_classes=4):
    """Build a synthetic ImageFolder dataset by replicating fixture images."""
    class_names = [f"class_{i:03d}" for i in range(n_classes)]
    for cls in class_names:
        os.makedirs(os.path.join(root, cls), exist_ok=True)

    idx = 0
    for i in range(n_images):
        cls = class_names[i % n_classes]
        src = os.path.join(FIXTURES, SOURCE_IMAGES[i % len(SOURCE_IMAGES)])
        dst = os.path.join(root, cls, f"img_{idx:05d}.jpg")
        shutil.copy2(src, dst)
        idx += 1

    return root


def bench_tensorimage(root, batch_size, size, epochs):
    """Benchmark tensorimage DataLoader."""
    transform = ti_transforms.RandomHorizontalFlip()

    dataset = ImageFolder(
        root,
        transform=transform,
        size=size,
        crop="center",
        normalize="imagenet",
    )
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)

    # Warmup
    for images, labels in loader:
        break

    times = []
    total_images = 0
    for epoch in range(epochs):
        t0 = time.perf_counter()
        epoch_images = 0
        for images, labels in loader:
            epoch_images += images.shape[0]
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total_images = epoch_images

    return times, total_images


def bench_torchvision(root, batch_size, size, epochs, num_workers):
    """Benchmark torchvision DataLoader."""
    transform = tv_transforms.Compose([
        tv_transforms.Resize(size),
        tv_transforms.CenterCrop(size),
        tv_transforms.RandomHorizontalFlip(),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = tv_datasets.ImageFolder(root, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Warmup
    for images, labels in loader:
        break

    times = []
    total_images = 0
    for epoch in range(epochs):
        t0 = time.perf_counter()
        epoch_images = 0
        for images, labels in loader:
            epoch_images += images.shape[0]
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total_images = epoch_images

    return times, total_images


def median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def main():
    parser = argparse.ArgumentParser(description="Benchmark tensorimage vs torchvision DataLoader")
    parser.add_argument("--n-images", type=int, default=128, help="Number of images in dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--size", type=int, default=224, help="Image size (resize + crop)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to iterate")
    parser.add_argument("--tv-workers", type=int, default=0,
                        help="torchvision num_workers (0=main process, try 2/4)")
    args = parser.parse_args()

    if not HAS_TV:
        print("ERROR: torchvision required for comparison benchmark")
        return

    tmpdir = tempfile.mkdtemp(prefix="ti_bench_")
    try:
        print(f"Building dataset: {args.n_images} images, {args.batch_size} batch size, "
              f"{args.size}px, {args.epochs} epochs")
        build_dataset(tmpdir, args.n_images)
        print(f"Dataset dir: {tmpdir}")
        print()

        # --- torchvision (num_workers=0, single process for fair comparison) ---
        print(f"=== torchvision DataLoader (num_workers={args.tv_workers}) ===")
        tv_times, tv_total = bench_torchvision(
            tmpdir, args.batch_size, args.size, args.epochs, args.tv_workers
        )
        tv_median = median(tv_times)
        tv_ips = tv_total / tv_median
        print(f"  Median epoch: {tv_median*1000:.1f} ms")
        print(f"  Throughput:   {tv_ips:.0f} img/s")
        print(f"  Per-epoch times: {[f'{t*1000:.1f}ms' for t in tv_times]}")
        print()

        # --- tensorimage DataLoader (num_workers=0, rayon parallelism) ---
        print("=== tensorimage DataLoader (num_workers=0, rayon) ===")
        ti_times, ti_total = bench_tensorimage(
            tmpdir, args.batch_size, args.size, args.epochs
        )
        ti_median = median(ti_times)
        ti_ips = ti_total / ti_median
        print(f"  Median epoch: {ti_median*1000:.1f} ms")
        print(f"  Throughput:   {ti_ips:.0f} img/s")
        print(f"  Per-epoch times: {[f'{t*1000:.1f}ms' for t in ti_times]}")
        print()

        # --- Also benchmark torchvision with workers for comparison ---
        for nw in [2, 4]:
            if nw == args.tv_workers:
                continue
            print(f"=== torchvision DataLoader (num_workers={nw}) ===")
            tv_times_nw, tv_total_nw = bench_torchvision(
                tmpdir, args.batch_size, args.size, args.epochs, nw
            )
            tv_median_nw = median(tv_times_nw)
            tv_ips_nw = tv_total_nw / tv_median_nw
            print(f"  Median epoch: {tv_median_nw*1000:.1f} ms")
            print(f"  Throughput:   {tv_ips_nw:.0f} img/s")
            print(f"  Per-epoch times: {[f'{t*1000:.1f}ms' for t in tv_times_nw]}")
            print()

        # --- Summary ---
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        speedup = tv_median / ti_median
        print(f"tensorimage:       {ti_median*1000:.1f} ms/epoch  ({ti_ips:.0f} img/s)")
        print(f"torchvision (w=0): {tv_median*1000:.1f} ms/epoch  ({tv_ips:.0f} img/s)")
        print(f"Speedup:           {speedup:.1f}x")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
