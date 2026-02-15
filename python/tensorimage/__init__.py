from tensorimage._tensorimage import (
    load as _rust_load,
    load_batch as _rust_load_batch,
    load_bytes as _rust_load_bytes,
    load_batch_bytes as _rust_load_batch_bytes,
    compute_phash as _rust_phash,
    phash_array as _rust_phash_array,
    phash_batch as _rust_phash_batch,
    hamming_distance,
    deduplicate as _rust_deduplicate,
    image_info as _rust_image_info,
    image_info_batch as _rust_image_info_batch,
)
from tensorimage import transforms

__all__ = [
    "load",
    "load_batch",
    "load_bytes",
    "load_batch_bytes",
    "image_info",
    "transforms",
    "to_dlpack",
    "phash",
    "phash_batch",
    "hamming_distance",
    "deduplicate",
    "filter_dataset",
]


def _numpy_to_torch(arr, device):
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required when device= is specified. "
            "Install it with: pip install torch"
        )
    tensor = torch.from_numpy(arr)  # zero-copy for CPU
    if device != "cpu":
        tensor = tensor.to(device)  # H2D transfer for CUDA
    return tensor


def load(path, size=None, algorithm=None, crop=None, normalize=None, device=None):
    """Load an image file and return a numpy array (or torch.Tensor if device is set).

    Args:
        path: Path to image file (JPEG or PNG).
        size: Target shortest edge size. None = no resize.
        algorithm: Resize algorithm. Default: "lanczos3".
        crop: Crop mode. "center" = center crop to size x size.
        normalize: Normalize preset ("imagenet", "clip", "[-1,1]").
        device: If set, return a torch.Tensor on this device.
            None = numpy (default), "cpu" = zero-copy torch CPU tensor,
            "cuda" = torch CUDA tensor.

    Returns:
        numpy.ndarray or torch.Tensor depending on device parameter.
    """
    arr = _rust_load(path, size=size, algorithm=algorithm, crop=crop, normalize=normalize)
    if device is not None:
        return _numpy_to_torch(arr, device)
    return arr


def load_batch(paths, size=None, algorithm=None, crop=None, normalize=None,
               workers=None, device=None):
    """Load multiple images in parallel using a rayon thread pool.

    Args:
        paths: List of image file paths.
        size: Target shortest edge size.
        algorithm: Resize algorithm. Default: "lanczos3".
        crop: Crop mode. "center" = center crop to size x size.
        normalize: Normalize preset.
        workers: Number of worker threads. Default: number of CPU cores.
        device: If set, return torch.Tensor(s) on this device.
            None = numpy (default), "cpu" = zero-copy torch CPU tensor,
            "cuda" = torch CUDA tensor.

    Returns:
        numpy.ndarray, list[numpy.ndarray], torch.Tensor, or list[torch.Tensor].
    """
    result = _rust_load_batch(paths, size=size, algorithm=algorithm, crop=crop,
                              normalize=normalize, workers=workers)
    if device is not None:
        if isinstance(result, list):
            return [_numpy_to_torch(arr, device) for arr in result]
        return _numpy_to_torch(result, device)
    return result


def load_bytes(data, size=None, algorithm=None, crop=None, normalize=None, device=None):
    """Load an image from raw bytes and return a numpy array (or torch.Tensor).

    Same parameters as load(), but accepts bytes instead of a file path.
    Useful for S3/HTTP workflows where image data is already in memory.

    Args:
        data: Raw image bytes (JPEG, PNG, WebP, AVIF).
        size: Target shortest edge size. None = no resize.
        algorithm: Resize algorithm. Default: "lanczos3".
        crop: Crop mode. "center" = center crop to size x size.
        normalize: Normalize preset ("imagenet", "clip", "[-1,1]").
        device: If set, return a torch.Tensor on this device.

    Returns:
        numpy.ndarray or torch.Tensor depending on device parameter.
    """
    arr = _rust_load_bytes(data, size=size, algorithm=algorithm, crop=crop, normalize=normalize)
    if device is not None:
        return _numpy_to_torch(arr, device)
    return arr


def load_batch_bytes(data_list, size=None, algorithm=None, crop=None, normalize=None,
                     workers=None, device=None):
    """Load multiple images from raw bytes in parallel.

    Same parameters as load_batch(), but accepts a list of bytes objects.

    Args:
        data_list: List of raw image bytes.
        size: Target shortest edge size.
        algorithm: Resize algorithm. Default: "lanczos3".
        crop: Crop mode. "center" = center crop to size x size.
        normalize: Normalize preset.
        workers: Number of worker threads. Default: number of CPU cores.
        device: If set, return torch.Tensor(s) on this device.

    Returns:
        numpy.ndarray, list[numpy.ndarray], torch.Tensor, or list[torch.Tensor].
    """
    result = _rust_load_batch_bytes(data_list, size=size, algorithm=algorithm, crop=crop,
                                    normalize=normalize, workers=workers)
    if device is not None:
        if isinstance(result, list):
            return [_numpy_to_torch(arr, device) for arr in result]
        return _numpy_to_torch(result, device)
    return result


def image_info(path):
    """Read image dimensions without decoding (header-only, very fast).

    Args:
        path: Path to an image file.

    Returns:
        Tuple of (width, height).
    """
    return _rust_image_info(path)


def to_dlpack(array):
    """Export a numpy array or torch tensor via the DLPack protocol.

    Enables framework-agnostic interop:
        torch.from_dlpack(ti.to_dlpack(arr))
        jax.dlpack.from_dlpack(ti.to_dlpack(arr))

    Args:
        array: numpy.ndarray or torch.Tensor.

    Returns:
        DLPack capsule.
    """
    import numpy as np
    if isinstance(array, np.ndarray):
        if hasattr(array, '__dlpack__'):
            return array.__dlpack__()
        raise TypeError(
            f"numpy >= 1.22 required for DLPack. Current: {np.__version__}"
        )
    try:
        import torch
        if isinstance(array, torch.Tensor):
            return torch.utils.dlpack.to_dlpack(array)
    except ImportError:
        pass
    raise TypeError(
        f"to_dlpack expects numpy.ndarray or torch.Tensor, got {type(array).__name__}"
    )


def phash(path_or_array, algorithm="dhash"):
    """Compute a perceptual hash of an image.

    Args:
        path_or_array: File path (str) or numpy uint8 array (H, W, 3).
        algorithm: "dhash" (default, fast) or "phash" (more robust).

    Returns:
        int: 64-bit perceptual hash.
    """
    if isinstance(path_or_array, str):
        return _rust_phash(path_or_array, algorithm=algorithm)
    else:
        return _rust_phash_array(path_or_array, algorithm=algorithm)


def phash_batch(paths, algorithm="dhash", workers=None):
    """Compute perceptual hashes for a batch of images in parallel.

    Args:
        paths: List of image file paths.
        algorithm: "dhash" or "phash".
        workers: Number of worker threads (default: CPU count).

    Returns:
        list[int]: 64-bit hashes, one per image.
    """
    return _rust_phash_batch(paths, algorithm=algorithm, workers=workers)


def deduplicate(paths, algorithm="dhash", threshold=None, workers=None):
    """Find and group near-duplicate images by perceptual hash.

    Args:
        paths: List of image file paths.
        algorithm: "dhash" or "phash".
        threshold: Max Hamming distance to consider as duplicate.
            Default: 0 for dhash (exact match), 10 for phash.
        workers: Number of worker threads (default: CPU count).

    Returns:
        dict with keys:
            - "keep_indices": list of indices to keep (first of each group)
            - "duplicate_groups": list of groups (each group is a list of indices)
            - "hashes": list of 64-bit hashes per input image
    """
    return _rust_deduplicate(
        paths, algorithm=algorithm, threshold=threshold, workers=workers
    )


def filter_dataset(
    paths,
    min_width=None,
    min_height=None,
    deduplicate=True,
    hash_algorithm="dhash",
    hash_threshold=None,
    min_aesthetic=None,
    workers=None,
    verbose=False,
):
    """Filter a dataset of images, removing undersized, duplicate, and low-quality samples.

    Applies filters cheapest-first:
    1. Dimension filter (fast header read, no decode)
    2. Perceptual hash dedup (Rust parallel)
    3. Aesthetic scoring (CLIP, only if min_aesthetic is set)

    Args:
        paths: List of image file paths.
        min_width: Minimum image width (pixels). None = no filter.
        min_height: Minimum image height (pixels). None = no filter.
        deduplicate: Whether to remove near-duplicates (default: True).
        hash_algorithm: "dhash" or "phash" for deduplication.
        hash_threshold: Max Hamming distance for dedup. Default: 0 (dhash) / 10 (phash).
        min_aesthetic: Minimum aesthetic score (1-10). Requires torch + open_clip.
            None = skip aesthetic filtering.
        workers: Number of worker threads.
        verbose: Print progress messages.

    Returns:
        dict with keys:
            - "paths": list of paths that passed all filters
            - "indices": list of original indices that passed
            - "stats": dict with counts for each filter stage
    """
    if not paths:
        return {"paths": [], "indices": [], "stats": {"total": 0}}

    paths = list(paths)
    n_total = len(paths)
    # Track which indices survive each stage
    surviving = list(range(n_total))
    stats = {"total": n_total}

    # Stage 1: Dimension filter
    if min_width is not None or min_height is not None:
        if verbose:
            print(f"[filter_dataset] Checking dimensions for {len(surviving)} images...")
        current_paths = [paths[i] for i in surviving]
        dims = _rust_image_info_batch(current_paths, workers=workers)

        new_surviving = []
        for idx, (w, h) in zip(surviving, dims):
            if (min_width is None or w >= min_width) and (
                min_height is None or h >= min_height
            ):
                new_surviving.append(idx)

        n_removed = len(surviving) - len(new_surviving)
        stats["dimension_removed"] = n_removed
        surviving = new_surviving
        if verbose:
            print(f"[filter_dataset] Dimension filter: removed {n_removed}, {len(surviving)} remain")

    # Stage 2: Deduplication
    if deduplicate and len(surviving) > 1:
        if verbose:
            print(f"[filter_dataset] Deduplicating {len(surviving)} images...")
        current_paths = [paths[i] for i in surviving]
        result = _rust_deduplicate(
            current_paths,
            algorithm=hash_algorithm,
            threshold=hash_threshold,
            workers=workers,
        )
        keep_set = set(result["keep_indices"])
        new_surviving = [surviving[i] for i in range(len(surviving)) if i in keep_set]

        n_removed = len(surviving) - len(new_surviving)
        stats["duplicate_removed"] = n_removed
        stats["duplicate_groups"] = len(result["duplicate_groups"])
        surviving = new_surviving
        if verbose:
            print(
                f"[filter_dataset] Dedup: removed {n_removed} duplicates "
                f"({len(result['duplicate_groups'])} groups), {len(surviving)} remain"
            )

    # Stage 3: Aesthetic scoring (optional, requires torch + open_clip)
    if min_aesthetic is not None and len(surviving) > 0:
        if verbose:
            print(f"[filter_dataset] Scoring aesthetics for {len(surviving)} images...")
        from tensorimage.aesthetic import AestheticScorer

        scorer = AestheticScorer()
        current_paths = [paths[i] for i in surviving]
        scores = scorer.score_batch(current_paths)

        new_surviving = []
        for idx, score in zip(surviving, scores):
            if score >= min_aesthetic:
                new_surviving.append(idx)

        n_removed = len(surviving) - len(new_surviving)
        stats["aesthetic_removed"] = n_removed
        surviving = new_surviving
        if verbose:
            print(f"[filter_dataset] Aesthetic filter: removed {n_removed}, {len(surviving)} remain")

    stats["kept"] = len(surviving)
    return {
        "paths": [paths[i] for i in surviving],
        "indices": surviving,
        "stats": stats,
    }
