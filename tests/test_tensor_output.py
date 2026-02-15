"""Tests for Phase 5: PyTorch tensor output, DLPack interop, and zero-copy transforms."""
import os
import sys
import unittest

import numpy as np
import pytest

import tensorimage as ti
from tensorimage import transforms

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture(name):
    return os.path.join(FIXTURES, name)


_has_torch = False
try:
    import torch
    _has_torch = True
except ImportError:
    pass

_has_cuda = _has_torch and torch.cuda.is_available()

requires_torch = pytest.mark.skipif(not _has_torch, reason="PyTorch not installed")
requires_cuda = pytest.mark.skipif(not _has_cuda, reason="CUDA not available")
requires_no_torch = pytest.mark.skipif(_has_torch, reason="PyTorch is installed")


# ============================================================
# Backward compatibility — device=None returns numpy
# ============================================================

class TestBackwardCompat:
    def test_load_returns_ndarray(self):
        arr = ti.load(fixture("sample.jpg"))
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint8

    def test_load_with_normalize_returns_ndarray(self):
        arr = ti.load(fixture("sample.jpg"), size=224, crop="center", normalize="imagenet")
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32

    def test_load_batch_returns_ndarray(self):
        result = ti.load_batch([fixture("sample.jpg")])
        if isinstance(result, list):
            assert isinstance(result[0], np.ndarray)
        else:
            assert isinstance(result, np.ndarray)

    def test_load_device_none_same_as_default(self):
        arr1 = ti.load(fixture("sample.jpg"), size=128)
        arr2 = ti.load(fixture("sample.jpg"), size=128, device=None)
        np.testing.assert_array_equal(arr1, arr2)


# ============================================================
# device="cpu" — zero-copy torch.Tensor on CPU
# ============================================================

@requires_torch
class TestDeviceCPU:
    def test_load_returns_tensor(self):
        tensor = ti.load(fixture("sample.jpg"), device="cpu")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"

    def test_load_dtype_uint8(self):
        tensor = ti.load(fixture("sample.jpg"), device="cpu")
        assert tensor.dtype == torch.uint8

    def test_load_shape_hwc(self):
        tensor = ti.load(fixture("sample.jpg"), device="cpu")
        assert tensor.ndim == 3
        assert tensor.shape[2] == 3  # HWC

    def test_load_normalized_returns_float(self):
        tensor = ti.load(fixture("sample.jpg"), size=224, crop="center",
                         normalize="imagenet", device="cpu")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.shape == (3, 224, 224)

    def test_load_values_match_numpy(self):
        arr = ti.load(fixture("sample.jpg"), size=128)
        tensor = ti.load(fixture("sample.jpg"), size=128, device="cpu")
        np.testing.assert_array_equal(arr, tensor.numpy())

    def test_load_batch_returns_tensors(self):
        paths = [fixture("sample.jpg"), fixture("landscape.jpg")]
        result = ti.load_batch(paths, size=224, crop="center",
                               normalize="imagenet", device="cpu")
        if isinstance(result, list):
            for t in result:
                assert isinstance(t, torch.Tensor)
                assert t.device.type == "cpu"
        else:
            assert isinstance(result, torch.Tensor)
            assert result.device.type == "cpu"
            assert result.shape[0] == 2


# ============================================================
# device="cuda" — CUDA tensor
# ============================================================

@requires_cuda
class TestDeviceCUDA:
    def test_load_returns_cuda_tensor(self):
        tensor = ti.load(fixture("sample.jpg"), device="cuda")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cuda"

    def test_load_normalized_cuda(self):
        tensor = ti.load(fixture("sample.jpg"), size=224, crop="center",
                         normalize="imagenet", device="cuda")
        assert tensor.dtype == torch.float32
        assert tensor.shape == (3, 224, 224)
        assert tensor.device.type == "cuda"

    def test_load_batch_cuda(self):
        paths = [fixture("sample.jpg")]
        result = ti.load_batch(paths, size=224, crop="center",
                               normalize="imagenet", device="cuda")
        if isinstance(result, list):
            assert result[0].device.type == "cuda"
        else:
            assert result.device.type == "cuda"


# ============================================================
# device= without torch installed raises ImportError
# ============================================================

@requires_no_torch
class TestDeviceNoTorch:
    def test_load_device_raises_import_error(self):
        with pytest.raises(ImportError, match="PyTorch is required"):
            ti.load(fixture("sample.jpg"), device="cpu")

    def test_load_batch_device_raises_import_error(self):
        with pytest.raises(ImportError, match="PyTorch is required"):
            ti.load_batch([fixture("sample.jpg")], device="cpu")


# ============================================================
# DLPack interop
# ============================================================

class TestToDlpack:
    def test_numpy_dlpack(self):
        arr = ti.load(fixture("sample.jpg"), size=64)
        if not hasattr(arr, '__dlpack__'):
            pytest.skip("numpy < 1.22, no __dlpack__ support")
        capsule = ti.to_dlpack(arr)
        assert capsule is not None

    @requires_torch
    def test_numpy_dlpack_roundtrip(self):
        arr = ti.load(fixture("sample.jpg"), size=64)
        if not hasattr(arr, '__dlpack__'):
            pytest.skip("numpy < 1.22, no __dlpack__ support")
        tensor = torch.from_dlpack(arr)  # direct from numpy dlpack
        np.testing.assert_array_equal(arr, tensor.numpy())

    @requires_torch
    def test_torch_dlpack(self):
        tensor = ti.load(fixture("sample.jpg"), size=64, device="cpu")
        capsule = ti.to_dlpack(tensor)
        assert capsule is not None

    @requires_torch
    def test_torch_dlpack_roundtrip(self):
        tensor = ti.load(fixture("sample.jpg"), size=64, device="cpu")
        capsule = ti.to_dlpack(tensor)
        tensor2 = torch.utils.dlpack.from_dlpack(capsule)
        torch.testing.assert_close(tensor, tensor2)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="to_dlpack expects"):
            ti.to_dlpack("not an array")


# ============================================================
# Transforms zero-copy verification
# ============================================================

class TestTransformsZeroCopy:
    def test_to_tensor_returns_contiguous(self):
        """ToTensor output should be contiguous (no .copy() needed by downstream)."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        t = transforms.ToTensor()
        result = t(img)
        if isinstance(result, np.ndarray):
            assert result.flags['C_CONTIGUOUS']
        elif _has_torch:
            assert result.is_contiguous()

    @requires_torch
    def test_compose_fast_path_returns_tensor(self):
        """Full fast-path returns torch.Tensor when torch is available."""
        t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        result = t(fixture("sample.jpg"))
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.is_contiguous()

    @requires_torch
    def test_compose_fused_returns_tensor(self):
        """Fused ToTensor+Normalize returns torch.Tensor when torch is available."""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        result = t(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.is_contiguous()
