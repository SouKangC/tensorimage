"""Tests for Phase 7: Smart dataset filtering — phash, dedup, filter_dataset."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import tensorimage as ti

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_JPG = os.path.join(FIXTURES, "sample.jpg")
SAMPLE_PNG = os.path.join(FIXTURES, "sample.png")
LANDSCAPE = os.path.join(FIXTURES, "landscape.jpg")
PORTRAIT = os.path.join(FIXTURES, "portrait.jpg")
CORRUPT = os.path.join(FIXTURES, "corrupt.bin")


# ============================================================
# TestPhash
# ============================================================
class TestPhash:
    """Tests for ti.phash() — perceptual hashing."""

    def test_dhash_returns_int(self):
        h = ti.phash(SAMPLE_JPG, algorithm="dhash")
        assert isinstance(h, int)

    def test_phash_returns_int(self):
        h = ti.phash(SAMPLE_JPG, algorithm="phash")
        assert isinstance(h, int)

    def test_dhash_deterministic(self):
        h1 = ti.phash(SAMPLE_JPG, algorithm="dhash")
        h2 = ti.phash(SAMPLE_JPG, algorithm="dhash")
        assert h1 == h2

    def test_phash_deterministic(self):
        h1 = ti.phash(SAMPLE_JPG, algorithm="phash")
        h2 = ti.phash(SAMPLE_JPG, algorithm="phash")
        assert h1 == h2

    def test_phash_same_content_similar(self):
        """sample.jpg and sample.png are the same content — pHash should be similar."""
        h1 = ti.phash(SAMPLE_JPG, algorithm="phash")
        h2 = ti.phash(SAMPLE_PNG, algorithm="phash")
        dist = ti.hamming_distance(h1, h2)
        assert dist <= 15, f"Same-content images should have low distance, got {dist}"

    def test_phash_from_array_dhash(self):
        """Hash from numpy array should return an int."""
        img = ti.load(SAMPLE_JPG)
        h = ti.phash(img, algorithm="dhash")
        assert isinstance(h, int)

    def test_phash_from_array_phash(self):
        """Hash from numpy array should return an int."""
        img = ti.load(SAMPLE_JPG)
        h = ti.phash(img, algorithm="phash")
        assert isinstance(h, int)

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unknown hash algorithm"):
            ti.phash(SAMPLE_JPG, algorithm="invalid")

    def test_corrupt_file(self):
        with pytest.raises(ValueError):
            ti.phash(CORRUPT)


# ============================================================
# TestPhashBatch
# ============================================================
class TestPhashBatch:
    """Tests for ti.phash_batch() — parallel hashing."""

    def test_returns_list(self):
        result = ti.phash_batch([SAMPLE_JPG, LANDSCAPE])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(h, int) for h in result)

    def test_matches_individual(self):
        """Batch hashes should match individual calls."""
        h1 = ti.phash(SAMPLE_JPG, algorithm="dhash")
        h2 = ti.phash(LANDSCAPE, algorithm="dhash")
        batch = ti.phash_batch([SAMPLE_JPG, LANDSCAPE], algorithm="dhash")
        assert batch == [h1, h2]

    def test_workers_param(self):
        result = ti.phash_batch([SAMPLE_JPG, LANDSCAPE], workers=2)
        assert len(result) == 2

    def test_empty_input(self):
        result = ti.phash_batch([])
        assert result == []


# ============================================================
# TestHammingDistance
# ============================================================
class TestHammingDistance:
    """Tests for ti.hamming_distance()."""

    def test_identical(self):
        assert ti.hamming_distance(0, 0) == 0
        assert ti.hamming_distance(12345, 12345) == 0

    def test_opposite(self):
        assert ti.hamming_distance(0, (1 << 64) - 1) == 64

    def test_single_bit_flip(self):
        assert ti.hamming_distance(0, 1) == 1
        assert ti.hamming_distance(0b1010, 0b1011) == 1


# ============================================================
# TestDeduplicate
# ============================================================
class TestDeduplicate:
    """Tests for ti.deduplicate()."""

    def test_no_duplicates_all_unique_algo(self):
        """With phash and low threshold, distinct images should be kept."""
        result = ti.deduplicate(
            [SAMPLE_JPG, LANDSCAPE, PORTRAIT],
            algorithm="phash",
            threshold=0,
        )
        assert "keep_indices" in result
        assert "duplicate_groups" in result
        assert "hashes" in result

    def test_exact_duplicates(self):
        """Same file twice should be detected as duplicates."""
        result = ti.deduplicate([SAMPLE_JPG, SAMPLE_JPG])
        assert len(result["keep_indices"]) == 1
        assert result["keep_indices"] == [0]
        assert len(result["duplicate_groups"]) == 1
        assert result["duplicate_groups"][0] == [0, 1]

    def test_result_dict_keys(self):
        result = ti.deduplicate([SAMPLE_JPG])
        assert set(result.keys()) == {"keep_indices", "duplicate_groups", "hashes"}

    def test_threshold_param(self):
        """With threshold=64, everything is a duplicate (max distance is 64)."""
        result = ti.deduplicate(
            [SAMPLE_JPG, LANDSCAPE, PORTRAIT],
            algorithm="phash",
            threshold=64,
        )
        assert len(result["keep_indices"]) == 1

    def test_empty_input(self):
        result = ti.deduplicate([])
        assert result["keep_indices"] == []
        assert result["duplicate_groups"] == []

    def test_single_image(self):
        result = ti.deduplicate([SAMPLE_JPG])
        assert result["keep_indices"] == [0]
        assert result["duplicate_groups"] == []


# ============================================================
# TestFilterDataset
# ============================================================
class TestFilterDataset:
    """Tests for ti.filter_dataset() — high-level filtering API."""

    def test_no_filters_passthrough(self):
        """With deduplicate=False and no other filters, all paths pass through."""
        paths = [SAMPLE_JPG, LANDSCAPE, PORTRAIT]
        result = ti.filter_dataset(paths, deduplicate=False)
        assert len(result["paths"]) == 3
        assert result["indices"] == [0, 1, 2]
        assert result["stats"]["total"] == 3
        assert result["stats"]["kept"] == 3

    def test_dedup_only(self):
        """Dedup should remove exact duplicates."""
        paths = [SAMPLE_JPG, SAMPLE_JPG, LANDSCAPE]
        result = ti.filter_dataset(paths, deduplicate=True)
        assert len(result["paths"]) < 3
        assert "duplicate_removed" in result["stats"]

    def test_min_dimensions(self):
        """Filter by minimum dimensions."""
        # landscape.jpg is 4000x2000, portrait is 2000x4000, sample is 1920x1080
        paths = [SAMPLE_JPG, LANDSCAPE, PORTRAIT]
        result = ti.filter_dataset(
            paths, min_width=3000, deduplicate=False
        )
        assert "dimension_removed" in result["stats"]
        # Only landscape (4000w) passes min_width=3000
        assert LANDSCAPE in result["paths"]

    def test_stats_counts(self):
        """Stats should have total and kept counts."""
        paths = [SAMPLE_JPG, LANDSCAPE]
        result = ti.filter_dataset(paths, deduplicate=False)
        assert result["stats"]["total"] == 2
        assert result["stats"]["kept"] == 2

    def test_empty_input(self):
        result = ti.filter_dataset([])
        assert result == {"paths": [], "indices": [], "stats": {"total": 0}}

    def test_verbose_runs_without_error(self, capsys):
        """Verbose mode should print progress."""
        paths = [SAMPLE_JPG, LANDSCAPE]
        ti.filter_dataset(paths, verbose=True, deduplicate=True)
        captured = capsys.readouterr()
        assert "[filter_dataset]" in captured.out


# ============================================================
# TestAestheticScoring
# ============================================================
class TestAestheticScoring:
    """Tests for AestheticScorer. Skipped if torch/open_clip not installed."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self):
        try:
            import torch
            import open_clip
        except ImportError:
            pytest.skip("Requires torch and open_clip_torch")

    def test_score_returns_float(self):
        from tensorimage.aesthetic import AestheticScorer

        scorer = AestheticScorer()
        score = scorer.score(SAMPLE_JPG)
        assert isinstance(score, float)

    def test_score_in_range(self):
        from tensorimage.aesthetic import AestheticScorer

        scorer = AestheticScorer()
        score = scorer.score(SAMPLE_JPG)
        assert 0 <= score <= 15, f"Score {score} outside expected range"

    def test_score_batch(self):
        from tensorimage.aesthetic import AestheticScorer

        scorer = AestheticScorer()
        scores = scorer.score_batch([SAMPLE_JPG, LANDSCAPE])
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
