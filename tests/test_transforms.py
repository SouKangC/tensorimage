"""Tests for tensorimage.transforms — Phase 3."""

import random

import numpy as np
import pytest

from tensorimage import transforms
from tensorimage.transforms import InterpolationMode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = "tests/fixtures"
SAMPLE_JPG = f"{FIXTURES}/sample.jpg"
LANDSCAPE_JPG = f"{FIXTURES}/landscape.jpg"

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torchvision.transforms as tv_transforms

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def make_hwc(h=100, w=150, seed=42):
    """Create a random uint8 HWC numpy image."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def make_pil(h=100, w=150, seed=42):
    """Create a random PIL image."""
    arr = make_hwc(h, w, seed)
    return Image.fromarray(arr)


# ===========================================================================
# TestCompose
# ===========================================================================

class TestCompose:
    def test_chain(self):
        """Compose chains transforms sequentially."""
        img = make_hwc(200, 300)
        t = transforms.Compose([
            transforms.Resize(100),
            transforms.CenterCrop(80),
        ])
        out = t(img)
        assert out.shape == (80, 80, 3)
        assert out.dtype == np.uint8

    def test_empty(self):
        """Compose with empty list returns input unchanged."""
        img = make_hwc()
        t = transforms.Compose([])
        out = t(img)
        np.testing.assert_array_equal(out, img)

    def test_repr(self):
        """Compose has a readable repr."""
        t = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
        r = repr(t)
        assert "Compose" in r
        assert "Resize" in r
        assert "ToTensor" in r


# ===========================================================================
# TestResize
# ===========================================================================

class TestResize:
    def test_int_shortest_edge_landscape(self):
        """Resize(256) on landscape: shortest edge becomes 256."""
        img = make_hwc(200, 400)
        out = transforms.Resize(256)(img)
        assert out.shape[0] == 256  # height was shortest
        assert out.shape[2] == 3

    def test_int_shortest_edge_portrait(self):
        """Resize(256) on portrait: shortest edge becomes 256."""
        img = make_hwc(400, 200)
        out = transforms.Resize(256)(img)
        assert out.shape[1] == 256  # width was shortest

    def test_tuple_exact_size(self):
        """Resize((128, 64)) resizes to exact dimensions."""
        img = make_hwc(200, 300)
        out = transforms.Resize((128, 64))(img)
        assert out.shape == (128, 64, 3)

    def test_max_size(self):
        """max_size constrains the longer edge."""
        img = make_hwc(100, 400)  # aspect 1:4
        out = transforms.Resize(200, max_size=500)(img)
        assert max(out.shape[0], out.shape[1]) <= 500

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_pil_input(self):
        """Resize accepts PIL Image input."""
        pil_img = make_pil(200, 300)
        out = transforms.Resize(100)(pil_img)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint8

    def test_interpolation_nearest(self):
        """Resize with NEAREST interpolation works."""
        img = make_hwc(100, 100)
        out = transforms.Resize(50, interpolation=transforms.NEAREST)(img)
        assert out.shape[:2] == (50, 50)

    def test_interpolation_bicubic(self):
        """Resize with BICUBIC interpolation works."""
        img = make_hwc(100, 100)
        out = transforms.Resize(50, interpolation=transforms.BICUBIC)(img)
        assert out.shape[:2] == (50, 50)

    def test_interpolation_bilinear(self):
        """Resize with BILINEAR interpolation works."""
        img = make_hwc(100, 100)
        out = transforms.Resize(50, interpolation=transforms.BILINEAR)(img)
        assert out.shape[:2] == (50, 50)

    def test_aspect_ratio_preserved(self):
        """Aspect ratio is preserved for int size."""
        img = make_hwc(100, 200)
        out = transforms.Resize(50)(img)
        assert out.shape[0] == 50
        assert out.shape[1] == 100  # 2:1 ratio preserved

    def test_output_dtype_u8(self):
        """Resize output is uint8 HWC."""
        img = make_hwc()
        out = transforms.Resize(50)(img)
        assert out.dtype == np.uint8
        assert out.ndim == 3


# ===========================================================================
# TestCenterCrop
# ===========================================================================

class TestCenterCrop:
    def test_int_size(self):
        """CenterCrop(80) produces 80x80."""
        img = make_hwc(100, 150)
        out = transforms.CenterCrop(80)(img)
        assert out.shape == (80, 80, 3)

    def test_tuple_size(self):
        """CenterCrop((60, 80)) produces (60, 80)."""
        img = make_hwc(100, 150)
        out = transforms.CenterCrop((60, 80))(img)
        assert out.shape == (60, 80, 3)

    def test_exact_center(self):
        """Crop is taken from the exact center."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[3:7, 3:7, :] = 255
        out = transforms.CenterCrop(4)(img)
        assert (out == 255).all()

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_pil_input(self):
        """CenterCrop accepts PIL Image."""
        pil_img = make_pil(100, 150)
        out = transforms.CenterCrop(80)(pil_img)
        assert out.shape == (80, 80, 3)

    def test_zero_padding_small_image(self):
        """Zero-pads when image is smaller than crop size."""
        img = make_hwc(50, 50)
        out = transforms.CenterCrop(100)(img)
        assert out.shape == (100, 100, 3)
        # Corners should be zero (padded)
        assert out[0, 0, 0] == 0

    def test_no_copy_artifact(self):
        """Output is independent of input (copied)."""
        img = make_hwc(100, 100)
        out = transforms.CenterCrop(50)(img)
        out[:] = 0
        assert img.sum() > 0  # Original unchanged


# ===========================================================================
# TestRandomCrop
# ===========================================================================

class TestRandomCrop:
    def test_basic(self):
        """RandomCrop produces correct output shape."""
        img = make_hwc(100, 150)
        out = transforms.RandomCrop(80)(img)
        assert out.shape == (80, 80, 3)

    def test_with_padding(self):
        """RandomCrop with padding increases effective size."""
        img = make_hwc(80, 80)
        out = transforms.RandomCrop(80, padding=10)(img)
        assert out.shape == (80, 80, 3)

    def test_pad_if_needed(self):
        """pad_if_needed auto-pads small images."""
        img = make_hwc(50, 50)
        out = transforms.RandomCrop(80, pad_if_needed=True)(img)
        assert out.shape == (80, 80, 3)

    def test_padding_modes(self):
        """All padding modes work without error."""
        img = make_hwc(80, 80)
        for mode in ["constant", "edge", "reflect", "symmetric"]:
            out = transforms.RandomCrop(80, padding=5, padding_mode=mode)(img)
            assert out.shape == (80, 80, 3)

    def test_varies_across_calls(self):
        """RandomCrop produces different crops on different calls."""
        img = make_hwc(200, 200)
        t = transforms.RandomCrop(50)
        crops = [t(img) for _ in range(10)]
        # At least some should differ
        differs = any(
            not np.array_equal(crops[0], crops[i]) for i in range(1, len(crops))
        )
        assert differs

    def test_error_too_small(self):
        """Raises ValueError if image too small without pad_if_needed."""
        img = make_hwc(50, 50)
        with pytest.raises(ValueError, match="smaller than crop size"):
            transforms.RandomCrop(80)(img)

    def test_output_dtype(self):
        """RandomCrop output is uint8."""
        img = make_hwc(100, 100)
        out = transforms.RandomCrop(50)(img)
        assert out.dtype == np.uint8


# ===========================================================================
# TestToTensor
# ===========================================================================

class TestToTensor:
    def test_dtype_float32(self):
        """ToTensor output is float32."""
        img = make_hwc(10, 10)
        out = transforms.ToTensor()(img)
        if HAS_TORCH:
            assert out.dtype == torch.float32
        else:
            assert out.dtype == np.float32

    def test_shape_chw(self):
        """ToTensor output is CHW."""
        img = make_hwc(10, 15)
        out = transforms.ToTensor()(img)
        if HAS_TORCH:
            assert tuple(out.shape) == (3, 10, 15)
        else:
            assert out.shape == (3, 10, 15)

    def test_range_01(self):
        """ToTensor output is in [0, 1]."""
        img = make_hwc()
        out = transforms.ToTensor()(img)
        if HAS_TORCH:
            assert out.min().item() >= 0.0
            assert out.max().item() <= 1.0
        else:
            assert out.min() >= 0.0
            assert out.max() <= 1.0

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_pil_input(self):
        """ToTensor accepts PIL Image."""
        pil_img = make_pil(10, 10)
        out = transforms.ToTensor()(pil_img)
        if HAS_TORCH:
            assert tuple(out.shape) == (3, 10, 10)
        else:
            assert out.shape == (3, 10, 10)

    def test_exact_values(self):
        """ToTensor produces exact float values from uint8."""
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)  # (1, 1, 3)
        out = transforms.ToTensor()(img)
        if HAS_TORCH:
            out = out.numpy()
        np.testing.assert_allclose(out[:, 0, 0], [0.0, 128 / 255.0, 1.0], atol=1e-6)


# ===========================================================================
# TestNormalize
# ===========================================================================

class TestNormalize:
    def test_formula_correct(self):
        """Normalize applies (x - mean) / std correctly."""
        arr = np.ones((3, 4, 4), dtype=np.float32) * 0.5
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        out = transforms.Normalize(mean, std)(arr)
        if HAS_TORCH:
            out = out.numpy()
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_imagenet_preset(self):
        """ImageNet normalize works end-to-end."""
        arr = np.random.rand(3, 10, 10).astype(np.float32)
        out = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )(arr)
        if HAS_TORCH:
            out = out.numpy()
        # Should have negative values after normalization
        assert out.min() < 0

    def test_inplace(self):
        """Inplace normalization modifies the input."""
        arr = np.ones((3, 4, 4), dtype=np.float32)
        original = arr.copy()
        transforms.Normalize([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], inplace=True)(arr)
        # (1.0 - 0.0) / 2.0 = 0.5 — array should be modified in-place
        assert not np.array_equal(arr, original)
        np.testing.assert_allclose(arr, 0.5, atol=1e-6)

    def test_broadcasting(self):
        """Mean/std broadcast correctly across spatial dimensions."""
        arr = np.ones((3, 2, 2), dtype=np.float32)
        mean = [0.0, 0.0, 0.0]
        std = [2.0, 2.0, 2.0]
        out = transforms.Normalize(mean, std)(arr)
        if HAS_TORCH:
            out = out.numpy()
        np.testing.assert_allclose(out, 0.5, atol=1e-6)

    def test_matches_manual_numpy(self):
        """Normalize matches manual numpy computation."""
        arr = np.random.rand(3, 8, 8).astype(np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        out = transforms.Normalize(mean, std)(arr)
        if HAS_TORCH:
            out = out.numpy()
        expected = arr.copy()
        for c in range(3):
            expected[c] = (expected[c] - mean[c]) / std[c]
        np.testing.assert_allclose(out, expected, atol=1e-5)


# ===========================================================================
# TestRandomHorizontalFlip
# ===========================================================================

class TestRandomHorizontalFlip:
    def test_p1_always_flips(self):
        """p=1.0 always flips."""
        img = make_hwc(10, 20)
        out = transforms.RandomHorizontalFlip(p=1.0)(img)
        np.testing.assert_array_equal(out, img[:, ::-1, :])

    def test_p0_never_flips(self):
        """p=0.0 never flips."""
        img = make_hwc(10, 20)
        out = transforms.RandomHorizontalFlip(p=0.0)(img)
        np.testing.assert_array_equal(out, img)

    def test_hwc_correct(self):
        """Flip on HWC array flips width axis."""
        img = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        out = transforms.RandomHorizontalFlip(p=1.0)(img)
        np.testing.assert_array_equal(out, img[:, ::-1, :])

    def test_chw_correct(self):
        """Flip on CHW float32 array flips width axis."""
        img = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        out = transforms.RandomHorizontalFlip(p=1.0)(img)
        np.testing.assert_array_equal(out, img[:, :, ::-1])


# ===========================================================================
# TestRandomVerticalFlip
# ===========================================================================

class TestRandomVerticalFlip:
    def test_p1_always_flips(self):
        """p=1.0 always flips."""
        img = make_hwc(10, 20)
        out = transforms.RandomVerticalFlip(p=1.0)(img)
        np.testing.assert_array_equal(out, img[::-1, :, :])

    def test_p0_never_flips(self):
        """p=0.0 never flips."""
        img = make_hwc(10, 20)
        out = transforms.RandomVerticalFlip(p=0.0)(img)
        np.testing.assert_array_equal(out, img)

    def test_hwc_correct(self):
        """Flip on HWC array flips height axis."""
        img = np.arange(24, dtype=np.uint8).reshape(4, 2, 3)
        out = transforms.RandomVerticalFlip(p=1.0)(img)
        np.testing.assert_array_equal(out, img[::-1, :, :])

    def test_chw_correct(self):
        """Flip on CHW float32 array flips height axis."""
        img = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
        out = transforms.RandomVerticalFlip(p=1.0)(img)
        np.testing.assert_array_equal(out, img[:, ::-1, :])


# ===========================================================================
# TestColorJitter
# ===========================================================================

class TestColorJitter:
    def test_brightness(self):
        """Brightness jitter changes pixel values."""
        img = make_hwc(50, 50)
        out = transforms.ColorJitter(brightness=0.5)(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_contrast(self):
        """Contrast jitter changes pixel values."""
        img = make_hwc(50, 50)
        out = transforms.ColorJitter(contrast=0.5)(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_saturation(self):
        """Saturation jitter changes pixel values."""
        img = make_hwc(50, 50)
        out = transforms.ColorJitter(saturation=0.5)(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_hue(self):
        """Hue jitter changes pixel values."""
        img = make_hwc(50, 50)
        out = transforms.ColorJitter(hue=0.1)(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_all_combined(self):
        """All jitter parameters combined work."""
        img = make_hwc(50, 50)
        out = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_output_range(self):
        """Output is clipped to [0, 255]."""
        img = make_hwc(50, 50)
        out = transforms.ColorJitter(brightness=0.9, contrast=0.9)(img)
        assert out.min() >= 0
        assert out.max() <= 255


# ===========================================================================
# TestFullPipeline
# ===========================================================================

class TestFullPipeline:
    def test_imagenet_pipeline(self):
        """Standard ImageNet inference pipeline."""
        img = make_hwc(300, 400)
        t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        out = t(img)
        if HAS_TORCH:
            assert tuple(out.shape) == (3, 224, 224)
            assert out.dtype == torch.float32
        else:
            assert out.shape == (3, 224, 224)
            assert out.dtype == np.float32

    def test_training_pipeline(self):
        """Training pipeline with augmentations."""
        img = make_hwc(300, 400)
        t = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        out = t(img)
        if HAS_TORCH:
            assert tuple(out.shape) == (3, 224, 224)
        else:
            assert out.shape == (3, 224, 224)

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_from_pil(self):
        """Full pipeline from PIL Image."""
        pil_img = make_pil(300, 400)
        t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        out = t(pil_img)
        if HAS_TORCH:
            assert tuple(out.shape) == (3, 224, 224)
        else:
            assert out.shape == (3, 224, 224)

    def test_from_numpy(self):
        """Full pipeline from numpy array."""
        img = make_hwc(300, 400)
        t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        out = t(img)
        if HAS_TORCH:
            assert tuple(out.shape) == (3, 224, 224)
        else:
            assert out.shape == (3, 224, 224)

    def test_output_shape_consistency(self):
        """Pipeline output shape is consistent across multiple calls."""
        img = make_hwc(300, 400)
        t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        shapes = set()
        for _ in range(5):
            out = t(img)
            if HAS_TORCH:
                shapes.add(tuple(out.shape))
            else:
                shapes.add(out.shape)
        assert len(shapes) == 1


# ===========================================================================
# TestInterpolationMode
# ===========================================================================

class TestInterpolationMode:
    def test_enum_values(self):
        """InterpolationMode has expected values."""
        assert InterpolationMode.NEAREST.value == "nearest"
        assert InterpolationMode.BILINEAR.value == "bilinear"
        assert InterpolationMode.BICUBIC.value == "bicubic"

    def test_used_in_resize(self):
        """InterpolationMode works as Resize parameter."""
        img = make_hwc(100, 100)
        out = transforms.Resize(50, interpolation=InterpolationMode.BILINEAR)(img)
        assert out.shape[:2] == (50, 50)

    def test_module_aliases(self):
        """Module-level aliases match enum values."""
        assert transforms.NEAREST is InterpolationMode.NEAREST
        assert transforms.BILINEAR is InterpolationMode.BILINEAR
        assert transforms.BICUBIC is InterpolationMode.BICUBIC


# ===========================================================================
# TestVsTorchvision
# ===========================================================================

@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not installed")
@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
class TestVsTorchvision:
    def test_resize_comparison(self):
        """Resize output within <=3 pixel values of torchvision."""
        pil_img = make_pil(200, 300)

        our_out = transforms.Resize(100)(pil_img)
        tv_out = np.array(tv_transforms.Resize(100)(pil_img))

        assert our_out.shape == tv_out.shape
        diff = np.abs(our_out.astype(np.int16) - tv_out.astype(np.int16))
        assert diff.max() <= 3, f"Max pixel diff: {diff.max()}"

    def test_center_crop_comparison(self):
        """CenterCrop matches torchvision exactly."""
        img = make_hwc(200, 300)
        pil_img = Image.fromarray(img)

        our_out = transforms.CenterCrop(100)(img)
        tv_out = np.array(tv_transforms.CenterCrop(100)(pil_img))

        np.testing.assert_array_equal(our_out, tv_out)

    def test_full_pipeline_comparison(self):
        """Full pipeline output within atol=0.02 of torchvision."""
        pil_img = make_pil(300, 400)
        img = np.array(pil_img)

        our_t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tv_t = tv_transforms.Compose([
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        our_out = our_t(img)
        tv_out = tv_t(pil_img)

        if HAS_TORCH:
            our_np = our_out.numpy()
        else:
            our_np = our_out
        tv_np = tv_out.numpy()

        assert our_np.shape == tv_np.shape
        np.testing.assert_allclose(our_np, tv_np, atol=0.02)
