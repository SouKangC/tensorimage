"""Tests for tensorimage.data — Phase 10 Dataset & DataLoader integration."""

import os

import numpy as np
import pytest

FIXTURES = "tests/fixtures"
IMAGEFOLDER_ROOT = f"{FIXTURES}/imagefolder"

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ===========================================================================
# ImageFolder tests (no torch needed for basic functionality)
# ===========================================================================

class TestImageFolder:
    def test_discovers_classes(self):
        """ImageFolder discovers class subdirectories sorted alphabetically."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT)
        assert ds.classes == ["class_a", "class_b"]

    def test_class_to_idx(self):
        """class_to_idx maps class names to integer indices."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT)
        assert ds.class_to_idx == {"class_a": 0, "class_b": 1}

    def test_sample_count(self):
        """Correct number of samples discovered."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT)
        # class_a has sample.jpg, landscape.jpg; class_b has portrait.jpg, sample.png
        assert len(ds) == 4

    def test_getitem_returns_path_label(self):
        """__getitem__ returns (path, label) tuple."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT)
        path, label = ds[0]
        assert isinstance(path, str)
        assert isinstance(label, int)
        assert os.path.isfile(path)

    def test_labels_match_classes(self):
        """Labels correspond to correct class indices."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT)
        for path, label in ds.samples:
            if "class_a" in path:
                assert label == 0
            elif "class_b" in path:
                assert label == 1

    def test_targets_attribute(self):
        """targets property returns list of all labels."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT)
        targets = ds.targets
        assert len(targets) == len(ds)
        assert all(isinstance(t, int) for t in targets)

    def test_imgs_alias(self):
        """imgs is an alias for samples."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT)
        assert ds.imgs is ds.samples

    def test_samples_sorted(self):
        """Samples within each class are sorted by filename."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT)
        class_a_paths = [p for p, l in ds.samples if l == 0]
        assert class_a_paths == sorted(class_a_paths)

    def test_nonexistent_dir_raises(self):
        """FileNotFoundError on non-existent directory."""
        from tensorimage.data import ImageFolder
        with pytest.raises(FileNotFoundError, match="Root directory not found"):
            ImageFolder("/nonexistent/path")

    def test_empty_dir_raises(self, tmp_path):
        """FileNotFoundError on directory with no class subdirs."""
        from tensorimage.data import ImageFolder
        with pytest.raises(FileNotFoundError, match="No class subdirectories"):
            ImageFolder(str(tmp_path))

    def test_stores_loading_params(self):
        """size, crop, normalize, device are stored."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT, size=224, crop="center",
                         normalize="imagenet", device="cpu")
        assert ds.size == 224
        assert ds.crop == "center"
        assert ds.normalize == "imagenet"
        assert ds.device == "cpu"


# ===========================================================================
# ImageDataset tests
# ===========================================================================

class TestImageDataset:
    def test_from_paths_with_labels(self):
        """ImageDataset with explicit labels."""
        from tensorimage.data import ImageDataset
        paths = [
            f"{FIXTURES}/sample.jpg",
            f"{FIXTURES}/landscape.jpg",
            f"{FIXTURES}/portrait.jpg",
        ]
        labels = [0, 1, 0]
        ds = ImageDataset(paths, labels)
        assert len(ds) == 3
        assert ds[1] == (paths[1], 1)

    def test_from_paths_no_labels(self):
        """ImageDataset without labels defaults to zeros."""
        from tensorimage.data import ImageDataset
        paths = [f"{FIXTURES}/sample.jpg", f"{FIXTURES}/landscape.jpg"]
        ds = ImageDataset(paths)
        assert len(ds) == 2
        assert ds[0][1] == 0
        assert ds[1][1] == 0

    def test_getitem_returns_path_label(self):
        """__getitem__ returns (path, label) tuple."""
        from tensorimage.data import ImageDataset
        paths = [f"{FIXTURES}/sample.jpg"]
        ds = ImageDataset(paths, [5])
        path, label = ds[0]
        assert path == paths[0]
        assert label == 5

    def test_mismatched_labels_raises(self):
        """ValueError when labels length doesn't match paths."""
        from tensorimage.data import ImageDataset
        with pytest.raises(ValueError, match="labels length"):
            ImageDataset(["a.jpg", "b.jpg"], [0])

    def test_targets_attribute(self):
        """targets property returns labels list."""
        from tensorimage.data import ImageDataset
        paths = [f"{FIXTURES}/sample.jpg", f"{FIXTURES}/landscape.jpg"]
        ds = ImageDataset(paths, [3, 7])
        assert ds.targets == [3, 7]

    def test_imgs_alias(self):
        """imgs is an alias for samples."""
        from tensorimage.data import ImageDataset
        paths = [f"{FIXTURES}/sample.jpg"]
        ds = ImageDataset(paths)
        assert ds.imgs is ds.samples

    def test_stores_loading_params(self):
        """size, crop, normalize, device are stored."""
        from tensorimage.data import ImageDataset
        ds = ImageDataset([f"{FIXTURES}/sample.jpg"], size=256, crop="center",
                          normalize="imagenet", device="cpu")
        assert ds.size == 256
        assert ds.crop == "center"


# ===========================================================================
# Collate and DataLoader tests (require torch)
# ===========================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestCollate:
    def test_basic_batch_loading(self):
        """collate_fn loads and returns (images, labels) tensors."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT, size=64, crop="center")
        collate_fn = ds.collate()
        batch = [ds[i] for i in range(len(ds))]
        images, labels = collate_fn(batch)
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.shape[0] == len(ds)
        assert labels.shape == (len(ds),)
        assert labels.dtype == torch.long

    def test_batch_with_normalize(self):
        """collate_fn with normalize produces CHW float tensors."""
        from tensorimage.data import ImageFolder
        ds = ImageFolder(IMAGEFOLDER_ROOT, size=64, crop="center",
                         normalize="imagenet")
        collate_fn = ds.collate()
        batch = [ds[i] for i in range(len(ds))]
        images, labels = collate_fn(batch)
        assert images.dtype == torch.float32
        # With normalize, output should be CHW: (N, 3, H, W)
        assert images.shape == (4, 3, 64, 64)

    def test_batch_with_transform(self):
        """collate_fn applies per-image transform."""
        from tensorimage.data import ImageFolder
        from tensorimage import transforms

        transform = transforms.RandomHorizontalFlip(p=1.0)
        ds = ImageFolder(IMAGEFOLDER_ROOT, size=64, crop="center",
                         transform=transform)
        collate_fn = ds.collate()
        batch = [ds[i] for i in range(len(ds))]
        images, labels = collate_fn(batch)
        assert isinstance(images, torch.Tensor)
        assert images.shape[0] == 4

    def test_image_dataset_collate(self):
        """collate works with ImageDataset too."""
        from tensorimage.data import ImageDataset
        paths = [
            f"{FIXTURES}/sample.jpg",
            f"{FIXTURES}/landscape.jpg",
        ]
        ds = ImageDataset(paths, [0, 1], size=32, crop="center",
                          normalize="imagenet")
        collate_fn = ds.collate()
        batch = [ds[i] for i in range(len(ds))]
        images, labels = collate_fn(batch)
        assert images.shape == (2, 3, 32, 32)
        assert labels.tolist() == [0, 1]


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestCreateDataLoader:
    def test_returns_dataloader(self):
        """create_dataloader returns a DataLoader instance."""
        from tensorimage.data import ImageFolder, create_dataloader
        ds = ImageFolder(IMAGEFOLDER_ROOT, size=32, crop="center",
                         normalize="imagenet")
        loader = create_dataloader(ds, batch_size=2, shuffle=False)
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_iteration_shapes(self):
        """DataLoader yields correct batch shapes."""
        from tensorimage.data import ImageFolder, create_dataloader
        ds = ImageFolder(IMAGEFOLDER_ROOT, size=32, crop="center",
                         normalize="imagenet")
        loader = create_dataloader(ds, batch_size=2, shuffle=False)
        images, labels = next(iter(loader))
        assert images.shape == (2, 3, 32, 32)
        assert labels.shape == (2,)

    def test_full_epoch(self):
        """Full epoch iteration completes without error."""
        from tensorimage.data import ImageFolder, create_dataloader
        ds = ImageFolder(IMAGEFOLDER_ROOT, size=32, crop="center",
                         normalize="imagenet")
        loader = create_dataloader(ds, batch_size=2, shuffle=False)
        total = 0
        for images, labels in loader:
            total += images.shape[0]
        assert total == len(ds)

    def test_drop_last(self):
        """drop_last=True drops incomplete final batch."""
        from tensorimage.data import ImageFolder, create_dataloader
        ds = ImageFolder(IMAGEFOLDER_ROOT, size=32, crop="center",
                         normalize="imagenet")
        # 4 samples with batch_size=3, drop_last=True -> 1 batch of 3
        loader = create_dataloader(ds, batch_size=3, shuffle=False,
                                   drop_last=True)
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0][0].shape[0] == 3

    def test_labels_correct(self):
        """Labels from DataLoader match class_to_idx."""
        from tensorimage.data import ImageFolder, create_dataloader
        ds = ImageFolder(IMAGEFOLDER_ROOT, size=32, crop="center",
                         normalize="imagenet")
        loader = create_dataloader(ds, batch_size=4, shuffle=False)
        images, labels = next(iter(loader))
        # Samples are sorted: class_a (0,0), class_b (1,1)
        expected = [ds[i][1] for i in range(4)]
        assert labels.tolist() == expected

    def test_image_dataset_dataloader(self):
        """create_dataloader works with ImageDataset."""
        from tensorimage.data import ImageDataset, create_dataloader
        paths = [
            f"{FIXTURES}/sample.jpg",
            f"{FIXTURES}/landscape.jpg",
        ]
        ds = ImageDataset(paths, [0, 1], size=32, crop="center",
                          normalize="imagenet")
        loader = create_dataloader(ds, batch_size=2, shuffle=False)
        images, labels = next(iter(loader))
        assert images.shape == (2, 3, 32, 32)
        assert labels.tolist() == [0, 1]
