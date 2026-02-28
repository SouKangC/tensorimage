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
            result = torch.from_numpy(result)
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
                        img = torch.from_numpy(result)
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
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1))  # HWC → CHW
        if _has_torch():
            import torch
            return torch.from_numpy(arr)
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


# ---------------------------------------------------------------------------
# GaussianBlur
# ---------------------------------------------------------------------------

class GaussianBlur:
    """Apply Gaussian blur using the Rust SIMD backend.

    Args:
        kernel_size: int or (kh, kw) — must be positive odd number(s).
        sigma: float or (min, max) tuple for random sigma sampling.
    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
            self.kernel_size = tuple(kernel_size)
        else:
            raise ValueError(f"kernel_size must be int or (kh, kw), got {kernel_size!r}")
        for k in self.kernel_size:
            if k <= 0 or k % 2 == 0:
                raise ValueError(f"kernel_size must be positive odd, got {k}")
        if isinstance(sigma, (int, float)):
            if sigma <= 0:
                raise ValueError(f"sigma must be positive, got {sigma}")
            self.sigma = (sigma, sigma)
        elif isinstance(sigma, (list, tuple)) and len(sigma) == 2:
            self.sigma = tuple(sigma)
        else:
            raise ValueError(f"sigma must be float or (min, max), got {sigma!r}")

    def __call__(self, img):
        from tensorimage._tensorimage import _gaussian_blur

        img = _ensure_numpy_u8_hwc(img)
        img = np.ascontiguousarray(img)
        sigma = random.uniform(*self.sigma)
        ks = max(self.kernel_size)
        return _gaussian_blur(img, ks, sigma)

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma})"


# ---------------------------------------------------------------------------
# RandomRotation
# ---------------------------------------------------------------------------

class RandomRotation:
    """Rotate the image by a random angle.

    Args:
        degrees: float or (min, max) range of degrees to rotate.
        interpolation: InterpolationMode (only BILINEAR supported currently).
        expand: If True, expand output to fit the whole rotated image.
        center: Optional (x, y) center of rotation. Default is image center.
        fill: Fill value for areas outside the rotated image.
    """

    def __init__(self, degrees, interpolation=BILINEAR, expand=False,
                 center=None, fill=0):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        elif isinstance(degrees, (list, tuple)) and len(degrees) == 2:
            self.degrees = tuple(degrees)
        else:
            raise ValueError(f"degrees must be float or (min, max), got {degrees!r}")
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, img):
        from tensorimage._tensorimage import _affine_transform

        img = _ensure_numpy_u8_hwc(img)
        img = np.ascontiguousarray(img)
        h, w, c = img.shape

        angle = random.uniform(*self.degrees)
        # Negate: PIL/torchvision treat positive as CCW in screen coords (y-down)
        rad = math.radians(-angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Center of rotation (torchvision uses pixel-center convention)
        if self.center is not None:
            cx, cy = self.center
        else:
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        if self.expand:
            # Compute new bounding box
            corners = [
                (-cx, -cy),
                (w - cx, -cy),
                (w - cx, h - cy),
                (-cx, h - cy),
            ]
            rot_corners = [
                (cos_a * x - sin_a * y, sin_a * x + cos_a * y)
                for x, y in corners
            ]
            xs = [p[0] for p in rot_corners]
            ys = [p[1] for p in rot_corners]
            new_w = int(math.ceil(max(xs) - min(xs)))
            new_h = int(math.ceil(max(ys) - min(ys)))
            # New center
            new_cx = new_w / 2.0
            new_cy = new_h / 2.0
        else:
            new_w, new_h = w, h
            new_cx, new_cy = cx, cy

        # Forward affine: translate to origin, rotate, translate to new center
        # [a, b, tx, c, d, ty]
        tx = -cx * cos_a + cy * sin_a + new_cx
        ty = -cx * sin_a - cy * cos_a + new_cy
        matrix = [cos_a, -sin_a, tx, sin_a, cos_a, ty]

        fill_val = self._make_fill(c)
        return _affine_transform(img, matrix, new_h, new_w, fill_val)

    def _make_fill(self, channels):
        if isinstance(self.fill, (int, float)):
            return [int(self.fill)] * channels
        return [int(v) for v in self.fill]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(degrees={self.degrees}, "
            f"expand={self.expand}, fill={self.fill})"
        )


# ---------------------------------------------------------------------------
# RandomAffine
# ---------------------------------------------------------------------------

class RandomAffine:
    """Apply a random affine transformation.

    Args:
        degrees: float or (min, max) rotation range.
        translate: Optional (tx, ty) fraction of image size.
        scale: Optional (min, max) scale factor range.
        shear: Optional float, (min, max), or (x_min, x_max, y_min, y_max).
        interpolation: InterpolationMode.
        fill: Fill value for areas outside the image.
        center: Optional (x, y) center of rotation.
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None,
                 interpolation=BILINEAR, fill=0, center=None):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        elif isinstance(degrees, (list, tuple)) and len(degrees) == 2:
            self.degrees = tuple(degrees)
        else:
            raise ValueError(f"degrees must be float or (min, max), got {degrees!r}")
        self.translate = translate
        self.scale = scale
        if shear is not None:
            if isinstance(shear, (int, float)):
                self.shear = (-shear, shear, 0.0, 0.0)
            elif len(shear) == 2:
                self.shear = (shear[0], shear[1], 0.0, 0.0)
            elif len(shear) == 4:
                self.shear = tuple(shear)
            else:
                raise ValueError(f"Invalid shear: {shear!r}")
        else:
            self.shear = None
        self.interpolation = interpolation
        self.fill = fill
        self.center = center

    def __call__(self, img):
        from tensorimage._tensorimage import _affine_transform

        img = _ensure_numpy_u8_hwc(img)
        img = np.ascontiguousarray(img)
        h, w, c = img.shape

        # Sample parameters
        angle = random.uniform(*self.degrees)
        # Negate: match torchvision/PIL screen-coordinate convention
        rad = math.radians(-angle)

        if self.scale is not None:
            s = random.uniform(*self.scale)
        else:
            s = 1.0

        if self.translate is not None:
            max_dx = self.translate[0] * w
            max_dy = self.translate[1] * h
            dx = random.uniform(-max_dx, max_dx)
            dy = random.uniform(-max_dy, max_dy)
        else:
            dx, dy = 0.0, 0.0

        if self.shear is not None:
            shear_x = math.radians(random.uniform(self.shear[0], self.shear[1]))
            shear_y = math.radians(random.uniform(self.shear[2], self.shear[3]))
        else:
            shear_x, shear_y = 0.0, 0.0

        # Center (torchvision uses pixel-center convention)
        if self.center is not None:
            cx, cy = self.center
        else:
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        # Build forward matrix: translate(-center) -> shear -> scale -> rotate -> translate(center+d)
        # M = R * Scale * Shear where R=[[cos,-sin],[sin,cos]], Shear=[[1,shx],[shy,1]]
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        shx = math.tan(shear_x)
        shy = math.tan(shear_y)

        m00 = s * (cos_a - sin_a * shy)
        m01 = s * (cos_a * shx - sin_a)
        m10 = s * (sin_a + cos_a * shy)
        m11 = s * (sin_a * shx + cos_a)

        tx = -cx * m00 - cy * m01 + cx + dx
        ty = -cx * m10 - cy * m11 + cy + dy

        matrix = [m00, m01, tx, m10, m11, ty]

        fill_val = self._make_fill(c)
        return _affine_transform(img, matrix, h, w, fill_val)

    def _make_fill(self, channels):
        if isinstance(self.fill, (int, float)):
            return [int(self.fill)] * channels
        return [int(v) for v in self.fill]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(degrees={self.degrees}, "
            f"translate={self.translate}, scale={self.scale}, shear={self.shear})"
        )


# ---------------------------------------------------------------------------
# RandomPerspective
# ---------------------------------------------------------------------------

class RandomPerspective:
    """Apply a random perspective transformation with a given probability.

    Args:
        distortion_scale: Strength of the perspective distortion (0-1).
        p: Probability of applying the transform.
        interpolation: InterpolationMode.
        fill: Fill value for areas outside the image.
    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=BILINEAR, fill=0):
        self.distortion_scale = distortion_scale
        self.p = p
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img):
        if random.random() >= self.p:
            if isinstance(img, np.ndarray):
                return img
            return _ensure_numpy_u8_hwc(img)

        from tensorimage._tensorimage import _perspective_transform

        img = _ensure_numpy_u8_hwc(img)
        img = np.ascontiguousarray(img)
        h, w, c = img.shape
        half_h = h / 2.0
        half_w = w / 2.0
        d = self.distortion_scale

        # Random endpoint distortions (matching torchvision)
        tl = (
            random.randint(0, int(d * half_w)),
            random.randint(0, int(d * half_h)),
        )
        tr = (
            w - 1 - random.randint(0, int(d * half_w)),
            random.randint(0, int(d * half_h)),
        )
        br = (
            w - 1 - random.randint(0, int(d * half_w)),
            h - 1 - random.randint(0, int(d * half_h)),
        )
        bl = (
            random.randint(0, int(d * half_w)),
            h - 1 - random.randint(0, int(d * half_h)),
        )

        # Compute perspective coefficients mapping output corners to input corners
        # Output corners: (0,0), (w-1,0), (w-1,h-1), (0,h-1)
        # Input corners: tl, tr, br, bl
        coeffs = self._find_coeffs(
            [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)],
            [tl, tr, br, bl],
        )

        fill_val = self._make_fill(c)
        return _perspective_transform(img, coeffs, h, w, fill_val)

    @staticmethod
    def _find_coeffs(output_coords, input_coords):
        """Find 8 perspective coefficients mapping output to input coordinates.

        Solves the system: for each point pair (x,y) -> (X,Y):
          X = (a*x + b*y + c) / (g*x + h*y + 1)
          Y = (d*x + e*y + f) / (g*x + h*y + 1)
        """
        A = []
        B = []
        for (x, y), (X, Y) in zip(output_coords, input_coords):
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
            B.append(X)
            B.append(Y)
        A = np.array(A, dtype=np.float64)
        B = np.array(B, dtype=np.float64)
        coeffs = np.linalg.solve(A, B)
        return coeffs.tolist()

    def _make_fill(self, channels):
        if isinstance(self.fill, (int, float)):
            return [int(self.fill)] * channels
        return [int(v) for v in self.fill]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(distortion_scale={self.distortion_scale}, "
            f"p={self.p}, fill={self.fill})"
        )


# ---------------------------------------------------------------------------
# RandomErasing
# ---------------------------------------------------------------------------

class RandomErasing:
    """Randomly erase a rectangular region in a CHW float tensor.

    This operates on CHW float32 tensors (after ToTensor), matching torchvision.

    Args:
        p: Probability of erasing.
        scale: (min, max) fraction of image area to erase.
        ratio: (min, max) aspect ratio range of the erased region.
        value: Fill value. 0 = black, 'random' for random noise.
        inplace: If True, modify tensor in-place.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3),
                 value=0, inplace=False):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(self, img):
        if not _is_chw_float(img):
            raise TypeError(
                "RandomErasing expects a float32 CHW tensor/array (output of ToTensor). "
                f"Got shape={getattr(img, 'shape', '?')} dtype={getattr(img, 'dtype', type(img).__name__)}"
            )
        if random.random() >= self.p:
            return img

        is_torch = False
        if _has_torch():
            import torch
            if isinstance(img, torch.Tensor):
                is_torch = True
                if not self.inplace:
                    img = img.clone()
                c, h, w = img.shape
            else:
                if not self.inplace:
                    img = img.copy()
                c, h, w = img.shape
        else:
            if not self.inplace:
                img = img.copy()
            c, h, w = img.shape

        area = h * w

        for _ in range(10):  # max attempts
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            eh = int(round(math.sqrt(target_area * aspect_ratio)))
            ew = int(round(math.sqrt(target_area / aspect_ratio)))

            if eh <= h and ew <= w:
                top = random.randint(0, h - eh) if eh < h else 0
                left = random.randint(0, w - ew) if ew < w else 0

                if self.value == "random":
                    if is_torch:
                        img[:, top : top + eh, left : left + ew] = torch.rand(c, eh, ew)
                    else:
                        img[:, top : top + eh, left : left + ew] = np.random.rand(c, eh, ew).astype(np.float32)
                else:
                    img[:, top : top + eh, left : left + ew] = self.value

                return img

        return img

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, scale={self.scale}, "
            f"ratio={self.ratio}, value={self.value})"
        )


# ---------------------------------------------------------------------------
# Grayscale
# ---------------------------------------------------------------------------

class Grayscale:
    """Convert image to grayscale.

    Args:
        num_output_channels: 1 or 3. If 3, the grayscale value is replicated
            across all 3 channels.
    """

    def __init__(self, num_output_channels=1):
        if num_output_channels not in (1, 3):
            raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}")
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        img = _ensure_numpy_u8_hwc(img)
        # BT.601 luma coefficients
        gray = (
            0.2989 * img[:, :, 0].astype(np.float32)
            + 0.5870 * img[:, :, 1].astype(np.float32)
            + 0.1140 * img[:, :, 2].astype(np.float32)
        )
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        if self.num_output_channels == 1:
            return gray[:, :, np.newaxis]
        else:
            return np.stack([gray, gray, gray], axis=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


# ---------------------------------------------------------------------------
# RandomGrayscale
# ---------------------------------------------------------------------------

class RandomGrayscale:
    """Randomly convert image to grayscale with probability p.

    Args:
        p: Probability of conversion.
    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return _ensure_numpy_u8_hwc(img)
        img = _ensure_numpy_u8_hwc(img)
        num_channels = img.shape[2]
        return Grayscale(num_output_channels=num_channels)(img)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


# ---------------------------------------------------------------------------
# GaussianNoise
# ---------------------------------------------------------------------------

class GaussianNoise:
    """Add Gaussian noise to an image.

    Accepts both HWC uint8 images and CHW float32 tensors (after ToTensor).

    Args:
        mean: Mean of the noise distribution.
        sigma: Standard deviation of the noise. For uint8 images this is in
            [0, 255] scale; for float32 images it is in [0, 1] scale.
        clip: If True, clip output to valid range ([0, 255] for uint8,
            [0, 1] for float32).
    """

    def __init__(self, mean=0.0, sigma=25.0, clip=True):
        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def __call__(self, img):
        # Handle float32 CHW input (e.g. after ToTensor)
        if _is_chw_float(img):
            if _has_torch():
                import torch
                if isinstance(img, torch.Tensor):
                    noise = torch.randn_like(img) * self.sigma + self.mean
                    noisy = img + noise
                    if self.clip:
                        noisy = torch.clamp(noisy, 0.0, 1.0)
                    return noisy
            # numpy float32 CHW
            noise = np.random.normal(self.mean, self.sigma, img.shape).astype(np.float32)
            noisy = img + noise
            if self.clip:
                noisy = np.clip(noisy, 0.0, 1.0)
            return noisy

        # uint8 HWC path
        img = _ensure_numpy_u8_hwc(img)
        noise = np.random.normal(self.mean, self.sigma, img.shape)
        noisy = img.astype(np.float32) + noise.astype(np.float32)
        if self.clip:
            noisy = np.clip(noisy, 0, 255)
        return noisy.astype(np.uint8)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, sigma={self.sigma})"


# ---------------------------------------------------------------------------
# Pad
# ---------------------------------------------------------------------------

class Pad:
    """Pad the image on all sides.

    Args:
        padding: int, (left_right, top_bottom), or (left, top, right, bottom).
        fill: Fill value for constant padding.
        padding_mode: "constant", "edge", "reflect", or "symmetric".
    """

    def __init__(self, padding, fill=0, padding_mode="constant"):
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 2:
                self.padding = (padding[0], padding[1], padding[0], padding[1])
            elif len(padding) == 4:
                self.padding = tuple(padding)
            else:
                raise ValueError(f"padding must have 2 or 4 elements, got {len(padding)}")
        else:
            raise ValueError(f"padding must be int or tuple, got {type(padding).__name__}")
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        img = _ensure_numpy_u8_hwc(img)
        left, top, right, bottom = self.padding
        pad_width = ((top, bottom), (left, right), (0, 0))

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
        return np.pad(img, pad_width, mode=np_mode, **kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(padding={self.padding}, "
            f"fill={self.fill}, padding_mode='{self.padding_mode}')"
        )


# ---------------------------------------------------------------------------
# ElasticTransform
# ---------------------------------------------------------------------------

class ElasticTransform:
    """Apply elastic deformation to the image.

    Generates random displacement fields, smooths them with Gaussian blur,
    and applies the resulting deformation.

    Args:
        alpha: Intensity of the displacement field.
        sigma: Smoothing sigma for the displacement field.
        interpolation: InterpolationMode (only BILINEAR supported).
        fill: Fill value for out-of-bounds pixels.
    """

    def __init__(self, alpha=50.0, sigma=5.0, interpolation=BILINEAR, fill=0):
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img):
        img = _ensure_numpy_u8_hwc(img)
        h, w, c = img.shape

        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, (h, w)).astype(np.float64)
        dy = np.random.uniform(-1, 1, (h, w)).astype(np.float64)

        # Smooth with Gaussian filter (simple numpy convolution)
        dx = self._gaussian_smooth(dx, self.sigma) * self.alpha
        dy = self._gaussian_smooth(dy, self.sigma) * self.alpha

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        map_x = (x_coords + dx).astype(np.float64)
        map_y = (y_coords + dy).astype(np.float64)

        # Bilinear interpolation
        fill_val = self._make_fill(c)
        output = np.zeros_like(img)

        x0 = np.floor(map_x).astype(int)
        y0 = np.floor(map_y).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        fx = (map_x - x0).astype(np.float32)
        fy = (map_y - y0).astype(np.float32)

        for ch in range(c):
            plane = img[:, :, ch].astype(np.float32)
            fill_v = fill_val[ch % len(fill_val)]

            # Clamp coordinates and use fill for out-of-bounds
            def sample(xi, yi):
                valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                xi_c = np.clip(xi, 0, w - 1)
                yi_c = np.clip(yi, 0, h - 1)
                vals = plane[yi_c, xi_c]
                vals[~valid] = fill_v
                return vals

            v00 = sample(x0, y0)
            v10 = sample(x1, y0)
            v01 = sample(x0, y1)
            v11 = sample(x1, y1)

            result = (
                v00 * (1 - fx) * (1 - fy)
                + v10 * fx * (1 - fy)
                + v01 * (1 - fx) * fy
                + v11 * fx * fy
            )
            output[:, :, ch] = np.clip(result, 0, 255).astype(np.uint8)

        return output

    @staticmethod
    def _gaussian_smooth(field, sigma):
        """1D separable Gaussian smoothing on a 2D field."""
        ks = int(6 * sigma + 1)
        if ks % 2 == 0:
            ks += 1
        half = ks // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()

        h, w = field.shape
        # Horizontal pass
        padded = np.pad(field, ((0, 0), (half, half)), mode="reflect")
        temp = np.zeros_like(field)
        for i in range(ks):
            temp += padded[:, i : i + w] * kernel[i]

        # Vertical pass
        padded = np.pad(temp, ((half, half), (0, 0)), mode="reflect")
        result = np.zeros_like(field)
        for i in range(ks):
            result += padded[i : i + h, :] * kernel[i]

        return result

    def _make_fill(self, channels):
        if isinstance(self.fill, (int, float)):
            return [int(self.fill)] * channels
        return [int(v) for v in self.fill]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(alpha={self.alpha}, sigma={self.sigma}, "
            f"fill={self.fill})"
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
    "GaussianBlur",
    "RandomRotation",
    "RandomAffine",
    "RandomPerspective",
    "RandomErasing",
    "Grayscale",
    "RandomGrayscale",
    "GaussianNoise",
    "Pad",
    "ElasticTransform",
    "InterpolationMode",
    "NEAREST",
    "BILINEAR",
    "BICUBIC",
]
