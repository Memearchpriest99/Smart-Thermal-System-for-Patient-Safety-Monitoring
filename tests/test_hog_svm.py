"""Tests for HOGSVMDetector (Section 4.4.2.2)."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984
from thermal_algorithms.core.types import Detection, Frame
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ambient(profile, base=20.0, noise=0.3, rng=None) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng(0)
    w, h = profile.resolution
    return rng.normal(base, noise, size=(h, w)).astype(np.float32)


def _stamp_person(arr, *, top_left, h, w, value=33.0):
    y, x = top_left
    arr[y:y + h, x:x + w] = value
    return arr


def _make_training_pair(profile, *, top_left, h=9, w=5, ambient=20.0, value=33.0,
                        rng=None, timestamp=0.0, camera_id=0):
    """Synthetic training example: empty room with one person-shaped hot block."""
    arr = _ambient(profile, base=ambient, rng=rng)
    _stamp_person(arr, top_left=top_left, h=h, w=w, value=value)
    frame = Frame(data=arr, timestamp=timestamp, camera_id=camera_id)
    det = Detection(
        bbox=(float(top_left[1]), float(top_left[0]), float(w), float(h)),
        score=1.0, class_id=1, camera_id=camera_id,
    )
    return frame, [det]


def _make_small_train_set(profile, n=30, rng_seed=42):
    """Generate n synthetic frames with one person each at varying positions."""
    rng = np.random.default_rng(rng_seed)
    h, w = profile.resolution[1], profile.resolution[0]
    examples = []
    for i in range(n):
        y = int(rng.integers(0, h - 9))
        x = int(rng.integers(0, w - 5))
        # vary the ambient slightly so the SVM can't exploit absolute temp
        ambient = float(rng.uniform(18, 24))
        examples.append(_make_training_pair(
            profile, top_left=(y, x), ambient=ambient, rng=rng,
            timestamp=float(i), camera_id=i % 3,
        ))
    return examples


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_constructs_with_mlx(self):
        d = HOGSVMDetector(MLX90640)
        wh, ww = d.window_size
        ch, cw = d.cell_size
        assert wh > 0 and ww > 0
        assert wh % ch == 0 and ww % cw == 0
        # Must leave room to slide a window
        assert wh < MLX90640.height
        assert ww < MLX90640.width

    def test_constructs_with_waveshare(self):
        d = HOGSVMDetector(WAVESHARE_26984)
        wh, ww = d.window_size
        # Waveshare's higher resolution → larger pixel-space window
        assert wh > 10 and ww > 4

    def test_window_capped_by_max_fraction(self):
        # An aggressive physical_person_size should not yield a frame-spanning window
        d = HOGSVMDetector(
            MLX90640,
            physical_person_size_m=(5.0, 5.0),
            assumed_distance_m=2.0,
            max_window_fraction=0.5,
        )
        wh, ww = d.window_size
        assert wh <= int(MLX90640.height * 0.5) + 1
        assert ww <= int(MLX90640.width * 0.5) + 1

    def test_explicit_window_used(self):
        d = HOGSVMDetector(MLX90640, window_size=(12, 4))
        # 12 % 2 == 0 and 4 % 2 == 0 (default cell), so no snapping
        assert d.window_size == (12, 4)

    def test_explicit_window_snapped_to_cell_multiple(self):
        # Window not divisible by cell → snap DOWN
        d = HOGSVMDetector(MLX90640, window_size=(13, 5), cell_size=(2, 2))
        wh, ww = d.window_size
        assert wh % 2 == 0 and ww % 2 == 0

    def test_rejects_no_profile(self):
        with pytest.raises(ValueError):
            HOGSVMDetector(None)  # type: ignore[arg-type]

    def test_rejects_bad_max_window_fraction(self):
        with pytest.raises(ValueError, match="max_window_fraction"):
            HOGSVMDetector(MLX90640, max_window_fraction=0)

    def test_class_metadata(self):
        assert HOGSVMDetector.name == "hog_svm_detector"
        assert HOGSVMDetector.is_trainable is True
        assert HOGSVMDetector.resolution_behavior == "parameterized"

    def test_feature_dim_is_none_before_fit(self):
        d = HOGSVMDetector(MLX90640)
        assert d.feature_dim is None


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_predict_before_fit_raises(self):
        d = HOGSVMDetector(MLX90640)
        f = Frame(data=_ambient(MLX90640), timestamp=0.0, camera_id=0)
        with pytest.raises(RuntimeError, match="before fit"):
            d.predict(f)

    def test_fit_on_synthetic_data(self):
        d = HOGSVMDetector(MLX90640)
        examples = _make_small_train_set(MLX90640, n=30)
        d.fit(examples)
        assert d.is_fitted is True
        assert d.feature_dim is not None and d.feature_dim > 0

    def test_fit_returns_self(self):
        d = HOGSVMDetector(MLX90640)
        examples = _make_small_train_set(MLX90640, n=10)
        assert d.fit(examples) is d

    def test_class_weight_defaults_to_none(self):
        d = HOGSVMDetector(MLX90640)
        assert d.get_params()["class_weight"] is None

    def test_class_weight_reaches_underlying_svc(self):
        d = HOGSVMDetector(MLX90640, class_weight="balanced")
        examples = _make_small_train_set(MLX90640, n=10)
        d.fit(examples)
        # LinearSVC (unlike SVC) doesn't expose a class_weight_ computed
        # attribute — confirming the constructor param was threaded through
        # is what matters here.
        assert d._svm.class_weight == "balanced"

    def test_fit_with_explicit_negatives(self):
        d = HOGSVMDetector(MLX90640)
        # Only positives in examples; provide negatives explicitly
        examples = _make_small_train_set(MLX90640, n=10)
        wh, ww = d.window_size
        negatives = np.random.default_rng(0).normal(20.0, 0.3, size=(20, wh, ww)).astype(np.float32)
        d.fit(examples, negatives=negatives)
        assert d.is_fitted

    def test_fit_with_no_positives_raises(self):
        d = HOGSVMDetector(MLX90640)
        # Empty Detection lists in every frame
        rng = np.random.default_rng(0)
        empty = [
            (Frame(data=_ambient(MLX90640, rng=rng), timestamp=0.0), [])
            for _ in range(5)
        ]
        with pytest.raises(ValueError, match="no positive"):
            d.fit(empty)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    @pytest.fixture
    def trained_detector(self):
        d = HOGSVMDetector(MLX90640, score_threshold=0.5)
        d.fit(_make_small_train_set(MLX90640, n=50))
        return d

    def test_returns_list_of_detections(self, trained_detector):
        f = Frame(data=_ambient(MLX90640), timestamp=0.0, camera_id=0)
        out = trained_detector.predict(f)
        assert isinstance(out, list)
        for d in out:
            assert isinstance(d, Detection)

    def test_detection_inherits_camera_id(self, trained_detector):
        rng = np.random.default_rng(0)
        arr = _ambient(MLX90640, rng=rng)
        _stamp_person(arr, top_left=(7, 12), h=9, w=5, value=33.0)
        f = Frame(data=arr, timestamp=0.0, camera_id=2)
        out = trained_detector.predict(f)
        for d in out:
            assert d.camera_id == 2

    def test_detects_synthetic_person(self):
        # Build a clean training set, then test on an in-distribution sample.
        # Use a soft threshold — synthetic data + tiny train set yield modest
        # margins, but the SVM should still rank a real person above background.
        rng = np.random.default_rng(7)
        d = HOGSVMDetector(MLX90640, score_threshold=-1.0)
        d.fit(_make_small_train_set(MLX90640, n=80, rng_seed=7))

        arr = _ambient(MLX90640, rng=rng)
        _stamp_person(arr, top_left=(8, 12), h=9, w=5, value=33.0)
        f = Frame(data=arr, timestamp=0.0, camera_id=0)
        out = d.predict(f)
        assert len(out) >= 1
        # At least one prediction should overlap the planted person bbox.
        person_box = (12, 8, 5, 9)
        overlapping = False
        for det in out:
            x, y, w, h = det.bbox
            ix0 = max(x, person_box[0]); iy0 = max(y, person_box[1])
            ix1 = min(x + w, person_box[0] + person_box[2])
            iy1 = min(y + h, person_box[1] + person_box[3])
            if ix1 > ix0 and iy1 > iy0:
                overlapping = True
                break
        assert overlapping, f"No prediction overlapped the planted person. Got: {[d.bbox for d in out]}"

    def test_predictions_have_scores(self, trained_detector):
        rng = np.random.default_rng(0)
        arr = _ambient(MLX90640, rng=rng)
        _stamp_person(arr, top_left=(7, 12), h=9, w=5, value=33.0)
        f = Frame(data=arr, timestamp=0.0)
        out = trained_detector.predict(f)
        for det in out:
            assert 0.0 <= det.score <= 1.0


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

class TestNMS:
    def test_nms_dedups_overlapping(self):
        from thermal_algorithms.human_detection.hog_svm import _greedy_nms
        boxes = [(0, 0, 10, 10), (1, 1, 10, 10), (50, 50, 10, 10)]
        scores = [1.0, 0.9, 0.8]
        keep = _greedy_nms(boxes, scores, iou_threshold=0.3)
        assert len(keep) == 2
        assert 0 in keep      # highest score among overlaps wins
        assert 2 in keep      # far-away box survives

    def test_nms_keeps_all_non_overlapping(self):
        from thermal_algorithms.human_detection.hog_svm import _greedy_nms
        boxes = [(0, 0, 5, 5), (10, 10, 5, 5), (20, 20, 5, 5)]
        scores = [1.0, 0.9, 0.8]
        assert len(_greedy_nms(boxes, scores, iou_threshold=0.3)) == 3


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        d = HOGSVMDetector(MLX90640)
        d.fit(_make_small_train_set(MLX90640, n=30))
        path = tmp_path / "hog.thalg"
        d.save(path)

        loaded = HOGSVMDetector.load(path)
        assert loaded.is_fitted
        assert loaded.feature_dim == d.feature_dim
        assert loaded.window_size == d.window_size
        assert loaded.cell_size == d.cell_size

    def test_save_load_predictions_match(self, tmp_path):
        d = HOGSVMDetector(MLX90640, score_threshold=0.5)
        d.fit(_make_small_train_set(MLX90640, n=50))
        path = tmp_path / "hog.thalg"
        d.save(path)
        loaded = HOGSVMDetector.load(path)

        rng = np.random.default_rng(0)
        arr = _ambient(MLX90640, rng=rng)
        _stamp_person(arr, top_left=(8, 12), h=9, w=5, value=33.0)
        f = Frame(data=arr, timestamp=0.0)

        a = d.predict(f)
        b = loaded.predict(f)
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert x.bbox == y.bbox
            assert x.score == pytest.approx(y.score)

    def test_register_via_checkpoint_registry(self, tmp_path):
        from thermal_algorithms.core.checkpoints import CheckpointRegistry
        reg = CheckpointRegistry(root=tmp_path)
        d = HOGSVMDetector(MLX90640)
        d.fit(_make_small_train_set(MLX90640, n=20))
        reg.register(d)
        loaded = reg.load(HOGSVMDetector, profile_name="MLX90640")
        assert loaded.window_size == d.window_size
