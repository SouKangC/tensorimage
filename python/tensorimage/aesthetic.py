"""CLIP-based aesthetic scoring for image quality assessment.

Requires: pip install torch open_clip_torch

Uses LAION aesthetic predictor V2: a linear head (768→1) trained on top of
CLIP ViT-L/14 embeddings to predict aesthetic scores (roughly 1-10 scale).
"""

import os
from pathlib import Path

_CACHE_DIR = os.path.join(Path.home(), ".cache", "tensorimage")
_AESTHETIC_URL = (
    "https://github.com/christophschuhmann/"
    "improved-aesthetic-predictor/raw/main/"
    "sac+logos+ava1-l14-linearMSE.pth"
)


class AestheticScorer:
    """CLIP-based aesthetic scorer.

    Scores images on an aesthetic quality scale (roughly 1-10).
    Uses CLIP ViT-L/14 features with a linear regression head
    trained on the LAION aesthetic dataset.

    Requires: pip install torch open_clip_torch

    Example:
        scorer = AestheticScorer()
        score = scorer.score("photo.jpg")  # → float, e.g. 6.2
        scores = scorer.score_batch(["a.jpg", "b.jpg"])  # → [6.2, 4.1]
    """

    def __init__(self, model_name="ViT-L-14", pretrained="openai", device="cpu"):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for aesthetic scoring. "
                "Install with: pip install torch"
            )
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip is required for aesthetic scoring. "
                "Install with: pip install open_clip_torch"
            )

        self._torch = torch
        self._device = device

        # Load CLIP model
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = self._model.to(device)
        self._model.eval()

        # Load aesthetic predictor linear head
        self._aesthetic_head = self._load_aesthetic_head()

    def _load_aesthetic_head(self):
        """Load the aesthetic predictor linear head weights."""
        torch = self._torch
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(_CACHE_DIR, "aesthetic_predictor_v2.pth")

        if not os.path.exists(cache_path):
            import urllib.request
            urllib.request.urlretrieve(_AESTHETIC_URL, cache_path)

        # Linear(768, 1)
        head = torch.nn.Linear(768, 1)
        state = torch.load(cache_path, map_location=self._device, weights_only=True)
        head.load_state_dict(state)
        head = head.to(self._device)
        head.eval()
        return head

    def score(self, path_or_image):
        """Score a single image's aesthetic quality.

        Args:
            path_or_image: File path (str/Path) or PIL.Image.

        Returns:
            float: Aesthetic score (roughly 1-10 scale).
        """
        scores = self.score_batch([path_or_image], batch_size=1)
        return scores[0]

    def score_batch(self, inputs, batch_size=32):
        """Score multiple images' aesthetic quality.

        Args:
            inputs: List of file paths (str/Path) or PIL.Image objects.
            batch_size: Batch size for CLIP inference.

        Returns:
            list[float]: Aesthetic scores.
        """
        torch = self._torch
        from PIL import Image

        # Preprocess all images
        tensors = []
        for inp in inputs:
            if isinstance(inp, (str, Path)):
                img = Image.open(inp).convert("RGB")
            else:
                img = inp.convert("RGB")
            tensors.append(self._preprocess(img))

        all_scores = []
        with torch.inference_mode():
            for i in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[i : i + batch_size]).to(self._device)
                features = self._model.encode_image(batch)
                # Normalize features
                features = features / features.norm(dim=-1, keepdim=True)
                scores = self._aesthetic_head(features.float())
                all_scores.extend(scores.squeeze(-1).cpu().tolist())

        return all_scores
