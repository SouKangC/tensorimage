"""Tests for tensorimage.load()"""
import os
import numpy as np
import pytest
from PIL import Image

import tensorimage as ti

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture(name):
    return os.path.join(FIXTURES, name)


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
