from tensorimage._tensorimage import (
    load as _rust_load,
    load_batch as _rust_load_batch,
)
from tensorimage import transforms

__all__ = ["load", "load_batch", "transforms", "to_dlpack"]


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
