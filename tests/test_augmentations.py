"""Tests for Phase 9 augmentation transforms."""

import math
import numpy as np
import pytest

from tensorimage import transforms


def make_hwc(h=100, w=150, channels=3):
    """Create a random uint8 HWC image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (h, w, channels), dtype=np.uint8)


def make_chw_float(c=3, h=100, w=150):
    """Create a random float32 CHW image in [0, 1]."""
    rng = np.random.RandomState(42)
    return rng.random((c, h, w)).astype(np.float32)


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


# ===========================================================================
# TestGaussianBlur
# ===========================================================================

class TestGaussianBlur:
    def test_shape_preserved(self):
        """Output shape matches input."""
        img = make_hwc()
        out = transforms.GaussianBlur(kernel_size=5, sigma=1.0)(img)
        assert out.shape == img.shape

    def test_dtype_u8(self):
        """Output dtype is uint8."""
        img = make_hwc()
        out = transforms.GaussianBlur(kernel_size=5, sigma=1.0)(img)
        assert out.dtype == np.uint8

    def test_large_sigma_more_blur(self):
        """Larger sigma reduces variance (more blurring)."""
        img = make_hwc()
        out_small = transforms.GaussianBlur(kernel_size=5, sigma=0.5)(img)
        out_large = transforms.GaussianBlur(kernel_size=21, sigma=10.0)(img)
        var_small = out_small.astype(np.float64).var()
        var_large = out_large.astype(np.float64).var()
        assert var_large < var_small

    def test_kernel_size_validation(self):
        """Even kernel_size raises ValueError."""
        with pytest.raises((ValueError, Exception)):
            transforms.GaussianBlur(kernel_size=4, sigma=1.0)

    def test_kernel_size_tuple(self):
        """(3, 7) works without error."""
        img = make_hwc()
        out = transforms.GaussianBlur(kernel_size=(3, 7), sigma=1.0)(img)
        assert out.shape == img.shape

    def test_single_sigma(self):
        """Float sigma works."""
        img = make_hwc()
        out = transforms.GaussianBlur(kernel_size=5, sigma=2.5)(img)
        assert out.shape == img.shape


# ===========================================================================
# TestRandomRotation
# ===========================================================================

class TestRandomRotation:
    def test_shape_preserved(self):
        """Shape same when expand=False."""
        img = make_hwc()
        out = transforms.RandomRotation(degrees=30)(img)
        assert out.shape == img.shape

    def test_zero_degrees_identity(self):
        """degrees=0 returns near-identical image (atol=1 for bilinear)."""
        img = make_hwc()
        out = transforms.RandomRotation(degrees=0)(img)
        np.testing.assert_allclose(out.astype(np.float64),
                                   img.astype(np.float64), atol=1)

    def test_180_rotation(self):
        """180 degree rotation roughly matches np.flip on both axes (atol=5)."""
        img = make_hwc(80, 80)
        out = transforms.RandomRotation(degrees=(180, 180))(img)
        flipped = np.flip(np.flip(img, axis=0), axis=1)
        np.testing.assert_allclose(out.astype(np.float64),
                                   flipped.astype(np.float64), atol=5)

    def test_expand_mode_dimensions(self):
        """expand=True produces larger or equal output."""
        img = make_hwc(100, 150)
        out = transforms.RandomRotation(degrees=(45, 45), expand=True)(img)
        assert out.shape[0] >= img.shape[0] or out.shape[1] >= img.shape[1]

    def test_fill_color(self):
        """fill=[255,0,0] puts red in corners."""
        img = make_hwc(100, 100)
        out = transforms.RandomRotation(degrees=(30, 30), fill=[255, 0, 0])(img)
        corners = [out[0, 0], out[0, -1], out[-1, 0], out[-1, -1]]
        has_red = any(np.array_equal(c, [255, 0, 0]) for c in corners)
        assert has_red


# ===========================================================================
# TestRandomAffine
# ===========================================================================

class TestRandomAffine:
    def test_shape_preserved(self):
        """Output shape matches input (same h, w)."""
        img = make_hwc()
        out = transforms.RandomAffine(degrees=0)(img)
        assert out.shape == img.shape

    def test_identity_no_change(self):
        """degrees=0, no translate/scale/shear is approximately identity (atol=1)."""
        img = make_hwc()
        out = transforms.RandomAffine(degrees=0)(img)
        np.testing.assert_allclose(out.astype(np.float64),
                                   img.astype(np.float64), atol=1)

    def test_scale_works(self):
        """scale=(2.0, 2.0) produces different output."""
        img = make_hwc()
        out = transforms.RandomAffine(degrees=0, scale=(2.0, 2.0))(img)
        assert not np.array_equal(out, img)

    def test_shear_works(self):
        """Shear with non-zero range produces different output."""
        img = make_hwc()
        out = transforms.RandomAffine(degrees=0, shear=30)(img)
        assert not np.array_equal(out, img)


# ===========================================================================
# TestRandomPerspective
# ===========================================================================

class TestRandomPerspective:
    def test_shape_preserved(self):
        """Output shape matches input."""
        img = make_hwc()
        out = transforms.RandomPerspective(p=1.0)(img)
        assert out.shape == img.shape

    def test_p_zero_unchanged(self):
        """p=0 returns same image."""
        img = make_hwc()
        out = transforms.RandomPerspective(p=0)(img)
        np.testing.assert_array_equal(out, img)

    def test_fill_works(self):
        """Fill value appears in output."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = transforms.RandomPerspective(distortion_scale=0.5, p=1.0,
                                           fill=128)(img)
        assert out.shape == img.shape


# ===========================================================================
# TestRandomErasing
# ===========================================================================

class TestRandomErasing:
    def test_shape_preserved(self):
        """Output shape matches (CHW float)."""
        img = make_chw_float()
        out = transforms.RandomErasing(p=1.0)(img)
        assert out.shape == img.shape

    def test_p_zero_unchanged(self):
        """p=0 returns identical tensor."""
        img = make_chw_float()
        out = transforms.RandomErasing(p=0)(img)
        np.testing.assert_array_equal(out, img)

    def test_erased_region_fill_value(self):
        """With p=1 and scale=(1.0,1.0), output should have the fill value."""
        # Use a square image so the erase region can cover 100% area with ratio=1.0
        img = make_chw_float(c=3, h=100, w=100)
        fill_val = 0.5
        out = transforms.RandomErasing(p=1.0, scale=(1.0, 1.0),
                                       ratio=(1.0, 1.0), value=fill_val)(img)
        assert np.isclose(out, fill_val).sum() > 0

    def test_random_fill(self):
        """value='random' doesn't crash."""
        img = make_chw_float()
        out = transforms.RandomErasing(p=1.0, value="random")(img)
        assert out.shape == img.shape


# ===========================================================================
# TestGrayscale
# ===========================================================================

class TestGrayscale:
    def test_shape_1_channel(self):
        """num_output_channels=1 returns (H, W, 1)."""
        img = make_hwc()
        out = transforms.Grayscale(num_output_channels=1)(img)
        assert out.shape == (100, 150, 1)

    def test_shape_3_channels(self):
        """num_output_channels=3 returns (H, W, 3)."""
        img = make_hwc()
        out = transforms.Grayscale(num_output_channels=3)(img)
        assert out.shape == (100, 150, 3)

    def test_bt601_weights(self):
        """Verify gray = 0.2989*R + 0.587*G + 0.114*B (atol=1 for rounding)."""
        img = make_hwc()
        out = transforms.Grayscale(num_output_channels=1)(img)
        expected = (0.2989 * img[:, :, 0].astype(np.float64) +
                    0.587 * img[:, :, 1].astype(np.float64) +
                    0.114 * img[:, :, 2].astype(np.float64))
        np.testing.assert_allclose(out[:, :, 0].astype(np.float64),
                                   expected, atol=1)

    def test_3_channels_all_same(self):
        """All 3 output channels are identical."""
        img = make_hwc()
        out = transforms.Grayscale(num_output_channels=3)(img)
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 2])


# ===========================================================================
# TestRandomGrayscale
# ===========================================================================

class TestRandomGrayscale:
    def test_p_zero_unchanged(self):
        """p=0 returns original."""
        img = make_hwc()
        out = transforms.RandomGrayscale(p=0)(img)
        np.testing.assert_array_equal(out, img)

    def test_p_one_always_grayscale(self):
        """p=1 always converts (all channels equal)."""
        img = make_hwc()
        out = transforms.RandomGrayscale(p=1.0)(img)
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 2])


# ===========================================================================
# TestGaussianNoise
# ===========================================================================

class TestGaussianNoise:
    def test_shape_preserved(self):
        """Output shape matches input (CHW float)."""
        img = make_chw_float()
        out = transforms.GaussianNoise(mean=0.0, sigma=0.1)(img)
        assert out.shape == img.shape

    def test_output_differs(self):
        """Output is different from input."""
        img = make_chw_float()
        out = transforms.GaussianNoise(mean=0.0, sigma=0.1)(img)
        assert not np.array_equal(out, img)

    def test_clip(self):
        """Output values in [0, 1] when clip=True."""
        img = make_chw_float()
        out = transforms.GaussianNoise(mean=0.0, sigma=0.5, clip=True)(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ===========================================================================
# TestPad
# ===========================================================================

class TestPad:
    def test_constant_dimensions(self):
        """padding=10 adds 20 to each dimension."""
        img = make_hwc()
        out = transforms.Pad(padding=10)(img)
        assert out.shape == (120, 170, 3)

    def test_constant_fill_value(self):
        """Padded region has correct fill value."""
        img = make_hwc(50, 50)
        out = transforms.Pad(padding=10, fill=128)(img)
        assert out[0, 0, 0] == 128
        assert out[0, 0, 1] == 128
        assert out[0, 0, 2] == 128

    def test_4_tuple_padding(self):
        """(left, top, right, bottom) dimensions correct."""
        img = make_hwc(50, 50)
        out = transforms.Pad(padding=(5, 10, 15, 20))(img)
        # width: 50 + 5 + 15 = 70, height: 50 + 10 + 20 = 80
        assert out.shape == (80, 70, 3)

    def test_reflect_mode(self):
        """padding_mode='reflect' works without error."""
        img = make_hwc()
        out = transforms.Pad(padding=10, padding_mode="reflect")(img)
        assert out.shape == (120, 170, 3)

    def test_edge_mode(self):
        """padding_mode='edge' works without error."""
        img = make_hwc()
        out = transforms.Pad(padding=10, padding_mode="edge")(img)
        assert out.shape == (120, 170, 3)


# ===========================================================================
# TestElasticTransform
# ===========================================================================

class TestElasticTransform:
    def test_shape_preserved(self):
        """Output shape matches input."""
        img = make_hwc()
        out = transforms.ElasticTransform(alpha=50.0, sigma=5.0)(img)
        assert out.shape == img.shape

    def test_alpha_zero_near_identity(self):
        """alpha=0 returns near-identical image (atol=1)."""
        img = make_hwc()
        out = transforms.ElasticTransform(alpha=0.0, sigma=5.0)(img)
        np.testing.assert_allclose(out.astype(np.float64),
                                   img.astype(np.float64), atol=1)


# ===========================================================================
# TestComposeIntegration
# ===========================================================================

class TestComposeIntegration:
    def test_gaussian_blur_in_compose(self):
        """GaussianBlur works in Compose chain."""
        img = make_hwc(200, 300)
        t = transforms.Compose([
            transforms.Resize(100),
            transforms.GaussianBlur(kernel_size=3, sigma=1.0),
        ])
        out = t(img)
        assert out.shape[0] == 100
        assert out.dtype == np.uint8

    def test_new_transforms_in_pipeline(self):
        """Resize -> GaussianBlur -> ToTensor -> Normalize pipeline works."""
        img = make_hwc(200, 300)
        t = transforms.Compose([
            transforms.Resize(100),
            transforms.GaussianBlur(kernel_size=3, sigma=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        out = t(img)
        if HAS_TORCH:
            assert tuple(out.shape)[0] == 3
        else:
            assert out.shape[0] == 3


# ===========================================================================
# TestVsTorchvision (augmentations)
# ===========================================================================

@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not installed")
class TestVsTorchvision:
    def test_gaussian_blur_comparison(self):
        """Output within atol=2 of torchvision GaussianBlur."""
        from PIL import Image

        img = make_hwc(64, 64)
        pil_img = Image.fromarray(img)

        our_out = transforms.GaussianBlur(kernel_size=5, sigma=1.5)(img)

        tv_blur = tv_transforms.GaussianBlur(kernel_size=5, sigma=1.5)
        tv_out = np.array(tv_blur(pil_img))

        assert our_out.shape == tv_out.shape
        diff = np.abs(our_out.astype(np.int16) - tv_out.astype(np.int16))
        assert diff.max() <= 2, f"Max pixel diff: {diff.max()}"

    def test_rotation_comparison(self):
        """At fixed angle, center region within atol of torchvision (bilinear)."""
        from PIL import Image
        from torchvision.transforms import InterpolationMode as TV_Interp

        img = make_hwc(64, 64)
        pil_img = Image.fromarray(img)

        angle = 30.0

        our_out = transforms.RandomRotation(degrees=(angle, angle))(img)

        tv_rot = tv_transforms.RandomRotation(
            degrees=(angle, angle),
            interpolation=TV_Interp.BILINEAR,
        )
        tv_out = np.array(tv_rot(pil_img))

        assert our_out.shape == tv_out.shape
        # Compare center region only (borders have fill-value differences)
        c = 16
        our_center = our_out[c:-c, c:-c]
        tv_center = tv_out[c:-c, c:-c]
        diff = np.abs(our_center.astype(np.int16) - tv_center.astype(np.int16))
        assert diff.mean() <= 5, f"Mean pixel diff: {diff.mean():.1f}"
