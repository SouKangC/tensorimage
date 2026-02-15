"""
tensorimage.transforms — drop-in replacement for torchvision.transforms.

Provides the same API as torchvision.transforms but uses Rust SIMD for resize
and numpy for spatial transforms. torch is optional: ToTensor returns
torch.Tensor if available, numpy array otherwise.

Usage:
    from tensorimage import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(Image.open("photo.jpg"))
"""

import enum
import functools
import math
import random

import numpy as np


# ---------------------------------------------------------------------------
# InterpolationMode enum
# ---------------------------------------------------------------------------

class InterpolationMode(enum.Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


# Module-level aliases matching torchvision
NEAREST = InterpolationMode.NEAREST
BILINEAR = InterpolationMode.BILINEAR
BICUBIC = InterpolationMode.BICUBIC

# Map our enum to Rust backend algorithm names
_INTERP_TO_ALGO = {
    InterpolationMode.NEAREST: "nearest",
    InterpolationMode.BILINEAR: "bilinear",
    InterpolationMode.BICUBIC: "catmullrom",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_numpy_u8_hwc(img):
    """Convert PIL Image or numpy array to uint8 HWC numpy array."""
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.dtype == np.uint8:
            return img
        raise ValueError(
            f"Expected uint8 HWC numpy array, got dtype={img.dtype} ndim={img.ndim}"
        )
    # Assume PIL Image
    try:
        arr = np.array(img)
    except Exception:
        raise TypeError(
            f"Expected PIL Image or numpy array, got {type(img).__name__}"
        )
    if arr.ndim == 2:
        # Grayscale → RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA → RGB
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _is_chw_float(img):
    """Check if image is float32 CHW layout."""
    if isinstance(img, np.ndarray):
        return img.dtype == np.float32 and img.ndim == 3 and img.shape[0] in (1, 3)
    if _has_torch():
        import torch
        if isinstance(img, torch.Tensor):
            return img.dtype == torch.float32 and img.ndim == 3 and img.shape[0] in (1, 3)
    return False


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class Compose:
    """Compose several transforms together.

    Automatically detects and applies two optimizations:

    1. **Fused ToTensor+Normalize** — when an adjacent (ToTensor, Normalize) pair
       is found, both are replaced by a single Rust call that does u8 HWC → f32 CHW
       with precomputed scale/bias in one pass (saves ~0.2 ms per call).

    2. **Full fast-path** — when transforms are exactly
       ``Resize(int) → CenterCrop(int) → ToTensor → Normalize`` and the input is
       a file path (``str``) or a PIL Image with a ``.filename`` attribute, the
       entire pipeline is routed through Rust (IDCT scaling + SIMD resize + crop +
       fused normalize). This is the big win: it skips millions of pixels during
       JPEG decode.

    Both optimizations are transparent — output matches the sequential path within
    floating-point tolerance.
    """

    def __init__(self, transforms):
        self.transforms = transforms

        # --- detect adjacent ToTensor+Normalize for fused path ---
        self._fused_idx = None
        self._fused_mean = None
        self._fused_std = None
        for i in range(len(transforms) - 1):
            if isinstance(transforms[i], ToTensor) and isinstance(transforms[i + 1], Normalize):
                self._fused_idx = i
                self._fused_mean = list(transforms[i + 1].mean)
                self._fused_std = list(transforms[i + 1].std)
                break

        # --- detect full fast-path: Resize(int), CenterCrop(int), ToTensor, Normalize ---
        self._fast_path = False
        self._fast_size = None
        self._fast_crop = None
        self._fast_mean = None
        self._fast_std = None
        if (
            len(transforms) == 4
            and isinstance(transforms[0], Resize)
            and isinstance(transforms[0].size, int)
            and isinstance(transforms[1], CenterCrop)
            and isinstance(transforms[2], ToTensor)
            and isinstance(transforms[3], Normalize)
        ):
            self._fast_path = True
            self._fast_size = transforms[0].size
            self._fast_crop = transforms[1].size[0]  # always (h, h) for int input
            self._fast_mean = list(transforms[3].mean)
            self._fast_std = list(transforms[3].std)

    def _try_fast_path(self, img):
        """Attempt the full Rust pipeline fast-path. Returns (result, True) or (None, False)."""
        from tensorimage._tensorimage import _load_pipeline

        path = None
        if isinstance(img, str):
            path = img
        else:
            # PIL Image with .filename
            fn = getattr(img, "filename", None)
            if fn and isinstance(fn, str) and len(fn) > 0:
                path = fn

        if path is None:
            return None, False

        result = _load_pipeline(
            path,
            self._fast_size,
            self._fast_crop,
            self._fast_mean,
            self._fast_std,
        )
        if _has_torch():
            import torch
            result = torch.from_numpy(result.copy())
        return result, True

    def __call__(self, img):
        # --- full fast-path ---
        if self._fast_path:
            result, ok = self._try_fast_path(img)
            if ok:
                return result

        # --- sequential with optional fused ToTensor+Normalize ---
        if self._fused_idx is not None:
            from tensorimage._tensorimage import _to_tensor_normalize

            for i, t in enumerate(self.transforms):
                if i == self._fused_idx:
                    arr = _ensure_numpy_u8_hwc(img)
                    arr = np.ascontiguousarray(arr)
                    result = _to_tensor_normalize(arr, self._fused_mean, self._fused_std)
                    if _has_torch():
                        import torch
                        img = torch.from_numpy(result.copy())
                    else:
                        img = result
                elif i == self._fused_idx + 1:
                    continue  # skip Normalize (already fused)
                else:
                    img = t(img)
            return img

        # --- plain sequential fallback ---
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        for t in self.transforms:
            lines.append(f"    {t}")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

class Resize:
    """Resize image using Rust SIMD backend.

    Args:
        size: int (shortest edge) or (h, w) tuple for exact size.
        interpolation: InterpolationMode enum value.
        max_size: If set, constrains the longer edge after shortest-edge resize.
        antialias: Accepted for API compatibility; always enabled.
    """

    def __init__(self, size, interpolation=BILINEAR, max_size=None, antialias=True):
        if isinstance(size, int):
            self.size = size
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            self.size = tuple(size)
        else:
            raise ValueError(f"size must be int or (h, w) tuple, got {size!r}")
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, img):
        from tensorimage._tensorimage import _resize_array

        img = _ensure_numpy_u8_hwc(img)
        h, w = img.shape[:2]
        algo = _INTERP_TO_ALGO[self.interpolation]

        if isinstance(self.size, tuple):
            target_h, target_w = self.size
        else:
            # Shortest-edge resize preserving aspect ratio
            if w < h:
                target_w = self.size
                target_h = int(round(h * self.size / w))
            else:
                target_h = self.size
                target_w = int(round(w * self.size / h))

            # Apply max_size constraint
            if self.max_size is not None:
                long_edge = max(target_h, target_w)
                if long_edge > self.max_size:
                    scale = self.max_size / long_edge
                    target_h = int(round(target_h * scale))
                    target_w = int(round(target_w * scale))

        # Ensure contiguous array for Rust
        img = np.ascontiguousarray(img)
        return _resize_array(img, target_h, target_w, algo)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(size={self.size}, "
            f"interpolation={self.interpolation}, max_size={self.max_size})"
        )


# ---------------------------------------------------------------------------
# CenterCrop
# ---------------------------------------------------------------------------

class CenterCrop:
    """Crop the center of the image to the given size.

    If the image is smaller than the crop size, it is zero-padded
    (matching torchvision behavior).
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            self.size = tuple(size)
        else:
            raise ValueError(f"size must be int or (h, w) tuple, got {size!r}")

    def __call__(self, img):
        img = _ensure_numpy_u8_hwc(img)
        h, w = img.shape[:2]
        crop_h, crop_w = self.size

        if h < crop_h or w < crop_w:
            # Pad with zeros if image is smaller than crop
            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            img = np.pad(
                img,
                ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            h, w = img.shape[:2]

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return img[top : top + crop_h, left : left + crop_w].copy()

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


# ---------------------------------------------------------------------------
# RandomCrop
# ---------------------------------------------------------------------------

class RandomCrop:
    """Crop at a random position.

    Args:
        size: int or (h, w) tuple.
        padding: Optional int or (left, top, right, bottom) padding.
        pad_if_needed: Pad image if smaller than crop size.
        fill: Fill value for constant padding.
        padding_mode: "constant", "edge", "reflect", or "symmetric".
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0,
                 padding_mode="constant"):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def _apply_padding(self, img):
        if self.padding is not None:
            p = self.padding
            if isinstance(p, int):
                pad_width = ((p, p), (p, p), (0, 0))
            elif len(p) == 2:
                pad_width = ((p[1], p[1]), (p[0], p[0]), (0, 0))
            elif len(p) == 4:
                # (left, top, right, bottom)
                pad_width = ((p[1], p[3]), (p[0], p[2]), (0, 0))
            else:
                raise ValueError(f"Invalid padding: {p}")

            mode_map = {
                "constant": "constant",
                "edge": "edge",
                "reflect": "reflect",
                "symmetric": "symmetric",
            }
            np_mode = mode_map.get(self.padding_mode, "constant")
            kwargs = {}
            if np_mode == "constant":
                kwargs["constant_values"] = self.fill
            img = np.pad(img, pad_width, mode=np_mode, **kwargs)
        return img

    def __call__(self, img):
        img = _ensure_numpy_u8_hwc(img)
        img = self._apply_padding(img)

        h, w = img.shape[:2]
        crop_h, crop_w = self.size

        if self.pad_if_needed:
            if h < crop_h:
                pad_h = crop_h - h
                img = np.pad(
                    img,
                    ((0, pad_h), (0, 0), (0, 0)),
                    mode="constant" if self.padding_mode == "constant" else self.padding_mode,
                    **({"constant_values": self.fill} if self.padding_mode == "constant" else {}),
                )
                h = img.shape[0]
            if w < crop_w:
                pad_w = crop_w - w
                img = np.pad(
                    img,
                    ((0, 0), (0, pad_w), (0, 0)),
                    mode="constant" if self.padding_mode == "constant" else self.padding_mode,
                    **({"constant_values": self.fill} if self.padding_mode == "constant" else {}),
                )
                w = img.shape[1]

        if h < crop_h or w < crop_w:
            raise ValueError(
                f"Image size ({h}, {w}) is smaller than crop size {self.size}. "
                "Set pad_if_needed=True to pad."
            )

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        return img[top : top + crop_h, left : left + crop_w].copy()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(size={self.size}, padding={self.padding}, "
            f"pad_if_needed={self.pad_if_needed})"
        )


# ---------------------------------------------------------------------------
# ToTensor
# ---------------------------------------------------------------------------

class ToTensor:
    """Convert uint8 HWC image to float32 CHW tensor.

    Returns torch.Tensor if torch is available, numpy array otherwise.
    """

    def __call__(self, img):
        img = _ensure_numpy_u8_hwc(img)
        # HWC uint8 → CHW float32 [0, 1]
        arr = img.astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        if _has_torch():
            import torch
            return torch.from_numpy(arr.copy())
        return arr

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class Normalize:
    """Per-channel normalization: (x - mean) / std.

    Works with both torch.Tensor and numpy arrays in CHW layout.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        if _has_torch():
            import torch
            if isinstance(tensor, torch.Tensor):
                if not self.inplace:
                    tensor = tensor.clone()
                mean = torch.tensor(self.mean, dtype=tensor.dtype).view(-1, 1, 1)
                std = torch.tensor(self.std, dtype=tensor.dtype).view(-1, 1, 1)
                tensor.sub_(mean).div_(std)
                return tensor

        # numpy path
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"Expected torch.Tensor or numpy array, got {type(tensor).__name__}")
        if not self.inplace:
            tensor = tensor.copy()
        mean = np.array(self.mean, dtype=np.float32).reshape(-1, 1, 1)
        std = np.array(self.std, dtype=np.float32).reshape(-1, 1, 1)
        tensor -= mean
        tensor /= std
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# ---------------------------------------------------------------------------
# RandomHorizontalFlip
# ---------------------------------------------------------------------------

class RandomHorizontalFlip:
    """Horizontally flip the image with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        if _has_torch():
            import torch
            if isinstance(img, torch.Tensor):
                return img.flip(-1)

        if isinstance(img, np.ndarray):
            if _is_chw_float(img):
                # CHW: flip width (axis 2)
                return img[:, :, ::-1].copy()
            else:
                # HWC: flip width (axis 1)
                return img[:, ::-1, :].copy()

        raise TypeError(f"Unsupported type: {type(img).__name__}")

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


# ---------------------------------------------------------------------------
# RandomVerticalFlip
# ---------------------------------------------------------------------------

class RandomVerticalFlip:
    """Vertically flip the image with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        if _has_torch():
            import torch
            if isinstance(img, torch.Tensor):
                return img.flip(-2)

        if isinstance(img, np.ndarray):
            if _is_chw_float(img):
                # CHW: flip height (axis 1)
                return img[:, ::-1, :].copy()
            else:
                # HWC: flip height (axis 0)
                return img[::-1, :, :].copy()

        raise TypeError(f"Unsupported type: {type(img).__name__}")

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


# ---------------------------------------------------------------------------
# ColorJitter
# ---------------------------------------------------------------------------

def _rgb_to_hsv(rgb):
    """Vectorized RGB [0,255] → HSV. H in [0,360), S in [0,1], V in [0,255]."""
    r, g, b = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc

    # Hue
    h = np.zeros_like(maxc)
    mask = diff > 0
    r_max = mask & (maxc == r)
    g_max = mask & (maxc == g)
    b_max = mask & (maxc == b)
    h[r_max] = (60.0 * ((g[r_max] - b[r_max]) / diff[r_max])) % 360.0
    h[g_max] = (60.0 * ((b[g_max] - r[g_max]) / diff[g_max]) + 120.0) % 360.0
    h[b_max] = (60.0 * ((r[b_max] - g[b_max]) / diff[b_max]) + 240.0) % 360.0

    # Saturation
    s = np.zeros_like(maxc)
    s[maxc > 0] = diff[maxc > 0] / maxc[maxc > 0]

    return h, s, maxc  # V = maxc


def _hsv_to_rgb(h, s, v):
    """Vectorized HSV → RGB [0,255]. H in [0,360), S in [0,1], V in [0,255]."""
    c = v * s
    h_prime = h / 60.0
    x = c * (1.0 - np.abs(h_prime % 2.0 - 1.0))

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    mask0 = (h_prime >= 0) & (h_prime < 1)
    mask1 = (h_prime >= 1) & (h_prime < 2)
    mask2 = (h_prime >= 2) & (h_prime < 3)
    mask3 = (h_prime >= 3) & (h_prime < 4)
    mask4 = (h_prime >= 4) & (h_prime < 5)
    mask5 = (h_prime >= 5) & (h_prime < 6)

    r[mask0] = c[mask0]; g[mask0] = x[mask0]
    r[mask1] = x[mask1]; g[mask1] = c[mask1]
    g[mask2] = c[mask2]; b[mask2] = x[mask2]
    g[mask3] = x[mask3]; b[mask3] = c[mask3]
    r[mask4] = x[mask4]; b[mask4] = c[mask4]
    r[mask5] = c[mask5]; b[mask5] = x[mask5]

    m = v - c
    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    return rgb


class ColorJitter:
    """Randomly change brightness, contrast, saturation, and hue.

    Args:
        brightness: float or (min, max). Factor range [max(0, 1-v), 1+v].
        contrast: Same as brightness.
        saturation: Same as brightness.
        hue: float or (min, max). Range [-v, v], must be in [0, 0.5].
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5))

    @staticmethod
    def _check_input(value, name, center=1, bound=(0, float("inf"))):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            if name == "hue":
                return (-value, value)
            return (max(bound[0], center - value), center + value)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            return tuple(value)
        else:
            raise TypeError(f"{name} must be float or (min, max) tuple")

    def __call__(self, img):
        img = _ensure_numpy_u8_hwc(img)
        img = img.astype(np.float32)

        # Build list of active transforms, then shuffle (torchvision behavior)
        transforms = []
        if self.brightness is not None:
            factor = random.uniform(*self.brightness)
            transforms.append(("brightness", factor))
        if self.contrast is not None:
            factor = random.uniform(*self.contrast)
            transforms.append(("contrast", factor))
        if self.saturation is not None:
            factor = random.uniform(*self.saturation)
            transforms.append(("saturation", factor))
        if self.hue is not None and (self.hue[0] != 0 or self.hue[1] != 0):
            factor = random.uniform(*self.hue)
            transforms.append(("hue", factor))

        random.shuffle(transforms)

        for name, factor in transforms:
            if name == "brightness":
                img = img * factor
            elif name == "contrast":
                # BT.601 grayscale weights
                gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
                mean_gray = gray.mean()
                img = factor * img + (1.0 - factor) * mean_gray
            elif name == "saturation":
                gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
                gray = gray[:, :, np.newaxis]
                img = factor * img + (1.0 - factor) * gray
            elif name == "hue":
                # Convert to uint8 for HSV conversion
                img_u8 = np.clip(img, 0, 255).astype(np.uint8)
                h, s, v = _rgb_to_hsv(img_u8)
                h = (h + factor * 360.0) % 360.0
                img = _hsv_to_rgb(h, s, v)

        return np.clip(img, 0, 255).astype(np.uint8)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue})"
        )


__all__ = [
    "Compose",
    "Resize",
    "CenterCrop",
    "RandomCrop",
    "ToTensor",
    "Normalize",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "ColorJitter",
    "InterpolationMode",
    "NEAREST",
    "BILINEAR",
    "BICUBIC",
]
