"""Tests for Phase 8: Extended Format Support + Real-World Robustness."""
import os
import numpy as np
import pytest

import tensorimage as ti

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# Format support: WebP
# ---------------------------------------------------------------------------

class TestWebP:
    def test_load_webp_file(self):
        arr = ti.load(os.path.join(FIXTURE_DIR, "sample.webp"))
        assert arr.ndim == 3
        assert arr.shape[2] == 3
        assert arr.dtype == np.uint8
        # 640x480 fixture
        assert arr.shape == (480, 640, 3)

    def test_load_webp_with_resize(self):
        arr = ti.load(os.path.join(FIXTURE_DIR, "sample.webp"), size=224)
        assert arr.ndim == 3
        assert arr.shape[2] == 3
        # Shortest edge should be 224
        assert min(arr.shape[0], arr.shape[1]) == 224

    def test_load_webp_full_pipeline(self):
        arr = ti.load(
            os.path.join(FIXTURE_DIR, "sample.webp"),
            size=256, crop="center", normalize="imagenet",
        )
        assert arr.shape == (3, 256, 256)
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# EXIF orientation
# ---------------------------------------------------------------------------

class TestEXIF:
    def test_exif_orientation_6(self):
        """Orientation=6 (90 CW): 400x200 stored → 200x400 after correction."""
        arr = ti.load(os.path.join(FIXTURE_DIR, "exif_orient6.jpg"))
        # After 90 CW rotation, width and height swap: 400×200 → 200×400
        assert arr.shape == (400, 200, 3)

    def test_exif_orientation_3(self):
        """Orientation=3 (180°): dimensions unchanged, pixels rotated."""
        arr = ti.load(os.path.join(FIXTURE_DIR, "exif_orient3.jpg"))
        # 180° rotation doesn't change dimensions
        assert arr.shape == (200, 400, 3)

    def test_exif_no_tag(self):
        """JPEG without EXIF orientation returns same dimensions as stored."""
        arr = ti.load(os.path.join(FIXTURE_DIR, "sample.jpg"))
        assert arr.shape == (1080, 1920, 3)

    def test_exif_orientation_1(self):
        """Orientation=1 (normal): no transform applied."""
        arr = ti.load(os.path.join(FIXTURE_DIR, "exif_orient1.jpg"))
        assert arr.shape == (200, 400, 3)

    def test_exif_png_ignored(self):
        """PNG loading unchanged (no EXIF parsing)."""
        arr = ti.load(os.path.join(FIXTURE_DIR, "sample.png"))
        assert arr.shape == (1080, 1920, 3)


# ---------------------------------------------------------------------------
# Bytes API
# ---------------------------------------------------------------------------

class TestLoadBytes:
    def test_load_bytes_jpeg(self):
        path = os.path.join(FIXTURE_DIR, "sample.jpg")
        data = open(path, "rb").read()
        arr_bytes = ti.load_bytes(data)
        arr_file = ti.load(path)
        np.testing.assert_array_equal(arr_bytes, arr_file)

    def test_load_bytes_png(self):
        path = os.path.join(FIXTURE_DIR, "sample.png")
        data = open(path, "rb").read()
        arr_bytes = ti.load_bytes(data)
        arr_file = ti.load(path)
        np.testing.assert_array_equal(arr_bytes, arr_file)

    def test_load_bytes_webp(self):
        path = os.path.join(FIXTURE_DIR, "sample.webp")
        data = open(path, "rb").read()
        arr_bytes = ti.load_bytes(data)
        arr_file = ti.load(path)
        np.testing.assert_array_equal(arr_bytes, arr_file)

    def test_load_bytes_with_resize(self):
        data = open(os.path.join(FIXTURE_DIR, "sample.jpg"), "rb").read()
        arr = ti.load_bytes(data, size=224)
        assert arr.ndim == 3
        assert min(arr.shape[0], arr.shape[1]) == 224

    def test_load_bytes_full_pipeline(self):
        data = open(os.path.join(FIXTURE_DIR, "sample.jpg"), "rb").read()
        arr = ti.load_bytes(data, size=256, crop="center", normalize="imagenet")
        assert arr.shape == (3, 256, 256)
        assert arr.dtype == np.float32

    def test_load_bytes_invalid(self):
        with pytest.raises(ValueError):
            ti.load_bytes(b"\x00\x01\x02\x03random garbage bytes")

    def test_load_bytes_matches_file_pipeline(self):
        """Bytes pipeline with full config matches file pipeline."""
        path = os.path.join(FIXTURE_DIR, "sample.jpg")
        data = open(path, "rb").read()
        arr_bytes = ti.load_bytes(data, size=256, crop="center", normalize="imagenet")
        arr_file = ti.load(path, size=256, crop="center", normalize="imagenet")
        np.testing.assert_allclose(arr_bytes, arr_file, atol=1e-6)


class TestLoadBatchBytes:
    def test_load_batch_bytes(self):
        paths = [
            os.path.join(FIXTURE_DIR, "sample.jpg"),
            os.path.join(FIXTURE_DIR, "sample.png"),
        ]
        data_list = [open(p, "rb").read() for p in paths]
        results_bytes = ti.load_batch_bytes(data_list, size=256, crop="center", normalize="imagenet")
        results_file = ti.load_batch(paths, size=256, crop="center", normalize="imagenet")
        np.testing.assert_allclose(results_bytes, results_file, atol=1e-6)

    def test_load_batch_bytes_stacked(self):
        """With crop+normalize, returns stacked [N, 3, H, W]."""
        paths = [
            os.path.join(FIXTURE_DIR, "sample.jpg"),
            os.path.join(FIXTURE_DIR, "sample.webp"),
        ]
        data_list = [open(p, "rb").read() for p in paths]
        result = ti.load_batch_bytes(data_list, size=224, crop="center", normalize="imagenet")
        assert result.shape == (2, 3, 224, 224)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# image_info (single file)
# ---------------------------------------------------------------------------

class TestImageInfo:
    def test_image_info_jpeg(self):
        w, h = ti.image_info(os.path.join(FIXTURE_DIR, "sample.jpg"))
        assert (w, h) == (1920, 1080)

    def test_image_info_png(self):
        w, h = ti.image_info(os.path.join(FIXTURE_DIR, "sample.png"))
        assert (w, h) == (1920, 1080)

    def test_image_info_webp(self):
        w, h = ti.image_info(os.path.join(FIXTURE_DIR, "sample.webp"))
        assert (w, h) == (640, 480)


# ---------------------------------------------------------------------------
# torch device tests (skipped if torch not installed)
# ---------------------------------------------------------------------------

def _has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


class TestBytesDevice:
    @pytest.mark.skipif(
        not _has_torch(), reason="torch not installed"
    )
    def test_load_bytes_device_cpu(self):
        import torch
        data = open(os.path.join(FIXTURE_DIR, "sample.jpg"), "rb").read()
        tensor = ti.load_bytes(data, size=224, crop="center", normalize="imagenet", device="cpu")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
