"""Tests for tensorimage.load() and tensorimage.load_batch()"""
import os
import numpy as np
import pytest
from PIL import Image

import tensorimage as ti

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture(name):
    return os.path.join(FIXTURES, name)


# ============================================================
# Phase 1 tests (unchanged)
# ============================================================


class TestSmoke:
    def test_load_returns_ndarray(self):
        arr = ti.load(fixture("sample.jpg"))
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint8
        assert arr.ndim == 3
        assert arr.shape[2] == 3

    def test_load_jpeg_dimensions(self):
        arr = ti.load(fixture("sample.jpg"))
        assert arr.shape == (1080, 1920, 3)

    def test_load_png(self):
        arr = ti.load(fixture("sample.png"))
        assert arr.shape == (1080, 1920, 3)
        assert arr.dtype == np.uint8


class TestResize:
    def test_resize_landscape(self):
        """Shortest edge of 4000x2000 is 2000; resize to 512 → 1024x512."""
        arr = ti.load(fixture("landscape.jpg"), size=512)
        assert arr.shape[0] == 512  # height (shortest)
        assert arr.shape[1] == 1024  # width scaled proportionally
        assert arr.shape[2] == 3

    def test_resize_portrait(self):
        """Shortest edge of 2000x4000 is 2000; resize to 512 → 512x1024."""
        arr = ti.load(fixture("portrait.jpg"), size=512)
        assert arr.shape[0] == 1024  # height scaled proportionally
        assert arr.shape[1] == 512  # width (shortest)
        assert arr.shape[2] == 3

    def test_resize_square_sample(self):
        """1920x1080: shortest edge is 1080 → size=256 gives 256xN."""
        arr = ti.load(fixture("sample.jpg"), size=256)
        assert arr.shape[0] == 256  # height (shortest)
        expected_w = round(1920 * 256 / 1080)
        assert arr.shape[1] == expected_w
        assert arr.shape[2] == 3

    def test_no_resize(self):
        """Without size=, dimensions match original."""
        arr = ti.load(fixture("sample.jpg"))
        assert arr.shape == (1080, 1920, 3)


class TestColorConversion:
    def test_rgba_to_rgb(self):
        """RGBA PNG should be converted to 3-channel RGB."""
        arr = ti.load(fixture("rgba.png"))
        assert arr.shape[2] == 3

    def test_grayscale_to_rgb(self):
        """Grayscale JPEG should be converted to 3-channel RGB."""
        arr = ti.load(fixture("gray.jpg"))
        assert arr.shape == (600, 800, 3)
        # All 3 channels should be equal for a grayscale source
        assert np.array_equal(arr[:, :, 0], arr[:, :, 1])
        assert np.array_equal(arr[:, :, 1], arr[:, :, 2])


class TestCorrectnessVsPIL:
    def test_resize_matches_pil(self):
        """Resized output should be close to PIL Lanczos (max diff ≤ 3)."""
        path = fixture("landscape.jpg")
        size = 512

        # tensorimage result
        ti_arr = ti.load(path, size=size)

        # PIL equivalent
        pil_img = Image.open(path)
        w, h = pil_img.size
        if w < h:
            new_w = size
            new_h = round(h * size / w)
        else:
            new_h = size
            new_w = round(w * size / h)
        pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        pil_arr = np.array(pil_resized)

        assert ti_arr.shape == pil_arr.shape, (
            f"Shape mismatch: ti={ti_arr.shape}, pil={pil_arr.shape}"
        )
        max_diff = np.max(np.abs(ti_arr.astype(int) - pil_arr.astype(int)))
        assert max_diff <= 3, f"Max pixel difference {max_diff} > 3"


class TestAlgorithms:
    @pytest.mark.parametrize("algo", ["nearest", "bilinear", "catmullrom", "mitchell", "lanczos3"])
    def test_valid_algorithms(self, algo):
        arr = ti.load(fixture("sample.jpg"), size=256, algorithm=algo)
        assert arr.shape[0] == 256
        assert arr.shape[2] == 3

    def test_algorithm_aliases(self):
        """catmull-rom and catmull_rom should also work."""
        arr1 = ti.load(fixture("sample.jpg"), size=256, algorithm="catmull-rom")
        arr2 = ti.load(fixture("sample.jpg"), size=256, algorithm="catmull_rom")
        assert arr1.shape == arr2.shape

    def test_lanczos_alias(self):
        arr = ti.load(fixture("sample.jpg"), size=256, algorithm="lanczos")
        assert arr.shape[0] == 256


class TestErrors:
    def test_nonexistent_file(self):
        with pytest.raises(ValueError, match="IO error"):
            ti.load("/nonexistent/path/image.jpg")

    def test_corrupt_file(self):
        with pytest.raises(ValueError, match="decode"):
            ti.load(fixture("corrupt.bin"))

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unknown resize algorithm"):
            ti.load(fixture("sample.jpg"), size=256, algorithm="invalid_algo")


# ============================================================
# Phase 2 tests
# ============================================================


class TestCrop:
    def test_center_crop_produces_square(self):
        """Center crop with size=224 produces a 224x224 output."""
        arr = ti.load(fixture("sample.jpg"), size=256, crop="center")
        assert arr.shape == (256, 256, 3)
        assert arr.dtype == np.uint8

    def test_center_crop_landscape(self):
        """Landscape image: resize to 512 shortest, crop to 512x512."""
        arr = ti.load(fixture("landscape.jpg"), size=512, crop="center")
        assert arr.shape == (512, 512, 3)

    def test_center_crop_portrait(self):
        """Portrait image: resize to 512 shortest, crop to 512x512."""
        arr = ti.load(fixture("portrait.jpg"), size=512, crop="center")
        assert arr.shape == (512, 512, 3)

    def test_crop_requires_size(self):
        """Crop without size should raise an error."""
        with pytest.raises(ValueError, match="crop requires size"):
            ti.load(fixture("sample.jpg"), crop="center")

    def test_invalid_crop_mode(self):
        """Invalid crop mode should raise an error."""
        with pytest.raises(ValueError, match="Unknown crop mode"):
            ti.load(fixture("sample.jpg"), size=256, crop="topleft")

    def test_crop_region_is_centered(self):
        """Verify the crop extracts the center region by comparing with manual slice."""
        path = fixture("landscape.jpg")
        size = 512

        # Get the resized-but-not-cropped image
        resized = ti.load(path, size=size)
        h, w, _ = resized.shape

        # Manual center crop
        y_offset = (h - size) // 2
        x_offset = (w - size) // 2
        manual_crop = resized[y_offset:y_offset + size, x_offset:x_offset + size, :]

        # ti center crop
        ti_crop = ti.load(path, size=size, crop="center")

        np.testing.assert_array_equal(ti_crop, manual_crop)


class TestNormalize:
    def test_imagenet_produces_f32_chw(self):
        """Imagenet normalization returns f32 CHW array."""
        arr = ti.load(fixture("sample.jpg"), size=256, normalize="imagenet")
        assert arr.dtype == np.float32
        assert arr.ndim == 3
        assert arr.shape[0] == 3  # CHW layout

    def test_layout_hwc_to_chw(self):
        """With normalize, shape changes from HWC to CHW."""
        arr_u8 = ti.load(fixture("sample.jpg"), size=256)
        arr_f32 = ti.load(fixture("sample.jpg"), size=256, normalize="imagenet")
        h, w, _ = arr_u8.shape
        assert arr_f32.shape == (3, h, w)

    def test_imagenet_values_in_range(self):
        """Imagenet-normalized values should be roughly in [-3, 3]."""
        arr = ti.load(fixture("sample.jpg"), size=256, normalize="imagenet")
        assert arr.min() >= -3.0
        assert arr.max() <= 3.0

    def test_pixel_exact_match_vs_numpy(self):
        """Verify normalize matches manual numpy: (arr/255.0 - mean) / std, transposed."""
        path = fixture("sample.jpg")
        size = 256

        # Get u8 HWC
        arr_u8 = ti.load(path, size=size)
        # Manual normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        manual = (arr_u8.astype(np.float32) / 255.0 - mean) / std
        manual = manual.transpose(2, 0, 1)  # HWC → CHW

        # ti normalization
        ti_arr = ti.load(path, size=size, normalize="imagenet")

        np.testing.assert_allclose(ti_arr, manual, atol=1e-5)

    def test_clip_preset(self):
        """CLIP preset should produce valid f32 CHW output."""
        arr = ti.load(fixture("sample.jpg"), size=224, normalize="clip")
        assert arr.dtype == np.float32
        assert arr.shape[0] == 3

    def test_neg1_to_1_preset(self):
        """[-1,1] preset should produce values in [-1, 1] range."""
        arr = ti.load(fixture("sample.jpg"), size=256, normalize="[-1,1]")
        assert arr.dtype == np.float32
        assert arr.min() >= -1.0 - 1e-5
        assert arr.max() <= 1.0 + 1e-5

    def test_invalid_preset(self):
        """Invalid normalize preset should raise an error."""
        with pytest.raises(ValueError, match="Unknown normalize preset"):
            ti.load(fixture("sample.jpg"), size=256, normalize="nonexistent")


class TestPipeline:
    def test_full_pipeline(self):
        """Resize + crop + normalize → f32 [3, 224, 224]."""
        arr = ti.load(
            fixture("sample.jpg"), size=224, crop="center", normalize="imagenet"
        )
        assert arr.shape == (3, 224, 224)
        assert arr.dtype == np.float32

    def test_crop_without_normalize(self):
        """Crop without normalize → u8 HWC square."""
        arr = ti.load(fixture("sample.jpg"), size=224, crop="center")
        assert arr.shape == (224, 224, 3)
        assert arr.dtype == np.uint8

    def test_normalize_without_crop(self):
        """Normalize without crop → f32 CHW, non-square."""
        arr = ti.load(fixture("sample.jpg"), size=256, normalize="imagenet")
        assert arr.shape[0] == 3
        assert arr.dtype == np.float32
        # Should not be square since no crop was applied
        assert arr.shape[1] != arr.shape[2]


class TestBatch:
    def test_basic_list_output(self):
        """Batch load without normalize returns a list of u8 arrays."""
        paths = [fixture("sample.jpg"), fixture("landscape.jpg")]
        result = ti.load_batch(paths, size=256)
        assert isinstance(result, list)
        assert len(result) == 2
        for arr in result:
            assert arr.dtype == np.uint8
            assert arr.ndim == 3

    def test_normalized_cropped_stacks_to_nchw(self):
        """Batch with normalize+crop → stacked [N, 3, H, W] ndarray."""
        paths = [fixture("sample.jpg"), fixture("landscape.jpg"), fixture("portrait.jpg")]
        result = ti.load_batch(paths, size=224, crop="center", normalize="imagenet")
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3, 224, 224)
        assert result.dtype == np.float32

    def test_workers_param(self):
        """Workers parameter is accepted."""
        paths = [fixture("sample.jpg"), fixture("landscape.jpg")]
        result = ti.load_batch(paths, size=224, crop="center", normalize="imagenet", workers=2)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2

    def test_empty_input(self):
        """Empty input returns empty list."""
        result = ti.load_batch([], size=224)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_error_propagation(self):
        """Error in one image propagates to caller."""
        paths = [fixture("sample.jpg"), "/nonexistent/image.jpg"]
        with pytest.raises(ValueError):
            ti.load_batch(paths, size=224)

    def test_varied_sizes_return_list(self):
        """Without crop, varied-size images return a list (not stacked)."""
        paths = [fixture("sample.jpg"), fixture("landscape.jpg")]
        result = ti.load_batch(paths, size=256)
        assert isinstance(result, list)
        assert len(result) == 2


class TestBackwardCompatibility:
    def test_phase1_load_returns_u8_hwc(self):
        """Phase 1 call: ti.load(path) returns u8 HWC."""
        arr = ti.load(fixture("sample.jpg"))
        assert arr.dtype == np.uint8
        assert arr.ndim == 3
        assert arr.shape == (1080, 1920, 3)

    def test_phase1_load_with_size(self):
        """Phase 1 call: ti.load(path, size=512) returns u8 HWC."""
        arr = ti.load(fixture("landscape.jpg"), size=512)
        assert arr.dtype == np.uint8
        assert arr.shape[0] == 512
        assert arr.shape[2] == 3

    def test_phase1_load_with_algorithm(self):
        """Phase 1 call: ti.load(path, size=256, algorithm='bilinear') returns u8 HWC."""
        arr = ti.load(fixture("sample.jpg"), size=256, algorithm="bilinear")
        assert arr.dtype == np.uint8
        assert arr.shape[0] == 256
        assert arr.shape[2] == 3
