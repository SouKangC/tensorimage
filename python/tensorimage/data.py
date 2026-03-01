"""PyTorch Dataset and DataLoader integration with Rust-backed batch loading."""

import os

import numpy as np


_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".avif"})


class _BaseDataset:
    """Shared base for ImageFolder and ImageDataset."""

    def __init__(self, transform=None, size=None, crop=None, normalize=None, device="cpu"):
        self.transform = transform
        self.size = size
        self.crop = crop
        self.normalize = normalize
        self.device = device
        self.samples = []  # list of (path, label)

    @property
    def targets(self):
        return [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate(self):
        """Return a collate_fn bound to this dataset's loading parameters."""
        import torch
        import tensorimage as ti

        size = self.size
        crop = self.crop
        normalize = self.normalize
        device = self.device
        transform = self.transform

        def _collate(batch):
            paths, labels = zip(*batch)
            # Rust parallel decode + resize + crop + normalize
            images = ti.load_batch(list(paths), size=size, crop=crop, normalize=normalize)
            # Per-image augmentations (if any)
            if transform is not None:
                if isinstance(images, np.ndarray):
                    image_list = [images[i] for i in range(len(images))]
                else:
                    image_list = list(images)
                image_list = [transform(img) for img in image_list]
                if isinstance(image_list[0], torch.Tensor):
                    images = torch.stack(image_list)
                else:
                    images = np.stack(image_list)
            # Convert to tensor if needed
            if not isinstance(images, torch.Tensor):
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images)
                else:
                    images = torch.stack([torch.from_numpy(img) if isinstance(img, np.ndarray) else img for img in images])
            if device is not None and device != "cpu":
                images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long)
            return images, labels

        return _collate


class ImageFolder(_BaseDataset):
    """Drop-in replacement for torchvision.datasets.ImageFolder.

    Uses Rust rayon-backed batch loading instead of Python multiprocessing.
    ``__getitem__`` returns ``(path, label)`` --- image loading is deferred to
    the collate function for efficient batch decoding.
    """

    def __init__(self, root, transform=None, size=None, crop=None, normalize=None, device="cpu"):
        super().__init__(transform=transform, size=size, crop=crop, normalize=normalize, device=device)
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Root directory not found: {self.root}")

        # Discover classes from subdirectories (sorted alphabetically)
        self.classes = sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )
        if not self.classes:
            raise FileNotFoundError(f"No class subdirectories found in: {self.root}")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Walk directory tree and discover images
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in _IMAGE_EXTENSIONS:
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

        self.imgs = self.samples  # torchvision compatibility alias


class ImageDataset(_BaseDataset):
    """Generic dataset from a list of image paths.

    No directory structure required. ``__getitem__`` returns ``(path, label)``
    --- image loading is deferred to the collate function.
    """

    def __init__(self, paths, labels=None, transform=None, size=None, crop=None, normalize=None, device="cpu"):
        super().__init__(transform=transform, size=size, crop=crop, normalize=normalize, device=device)
        self.paths = list(paths)
        if labels is None:
            labels = [0] * len(self.paths)
        else:
            labels = list(labels)
            if len(labels) != len(self.paths):
                raise ValueError(
                    f"labels length ({len(labels)}) must match paths length ({len(self.paths)})"
                )
        self.samples = [(p, l) for p, l in zip(self.paths, labels)]
        self.imgs = self.samples


def create_dataloader(dataset, batch_size=32, shuffle=True, drop_last=False, **kwargs):
    """Create a DataLoader that uses Rust-backed batch loading.

    All parallelism is handled by Rust rayon, so ``num_workers=0``.
    """
    import torch.utils.data

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        collate_fn=dataset.collate(),
        **kwargs,
    )
