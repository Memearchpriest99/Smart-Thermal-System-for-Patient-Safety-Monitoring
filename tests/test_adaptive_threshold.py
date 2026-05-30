"""Tests for AdaptiveThresholdDetector (§ 4.4.2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984
from thermal_algorithms.core.types import Frame
from thermal_algorithms.human_detection.adaptive_threshold import (
    AdaptiveThresholdDetector,
)


# ---------------------------------------------------------------------------
# Frame factories
# ---------------------------------------------------------------------------

def _ambient(profile, ambient_c=20.0, noise=0.0, rng=None) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng(0)
    w, h = profile.resolution
    return rng.normal(ambient_c, noise, size=(h, w)).astype(np.float32)


def _stamp_blob(arr, *, center, radius, value, shape="circle"):
    h, w = arr.shape
    cy, cx = center
    ys, xs = np.ogrid[:h, :w]
    if shape == "circle":
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    elif shape == "rect":
        rh, rw = (radius, radius) if isinstance(radius, int) else radius
        mask = (np.abs(ys - cy) <= rh) & (np.abs(xs - cx) <= rw)
    else:
        raise ValueError(shape)
    arr[mask] = value
    return arr


def _frame(arr, timestamp=0.0, camera_id=0) -> Frame:
    return Frame(data=arr.astype(np.float32), timestamp=timestamp, camera_id=camera_id)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_constructs_with_mlx(self):
        d = AdaptiveThresholdDetector(sensor_profile=MLX90640)
        assert d.sensor_profile is MLX90640
        assert d.block_size >= 3 and d.block_size % 2 == 1

    def test_constructs_with_waveshare(self):
        d = AdaptiveThresholdDetector(sensor_profile=WAVESHARE_26984)
        assert d.block_size >= 3 and d.block_size % 2 == 1

    def test_block_size_scales_up_for_higher_resolution(self):
        d_mlx = AdaptiveThresholdDetector(sensor_profile=MLX90640)
        d_ws = AdaptiveThresholdDetector(sensor_profile=WAVESHARE_26984)
        assert d_ws.block_size >= d_mlx.block_size

    def test_explicit_block_size_used(self):
        d = AdaptiveThresholdDetector(sensor_profile=MLX90640, block_size=11)
        assert d.block_size == 11

    def test_even_block_size_snapped_to_odd(self):
        d = AdaptiveThresholdDetector(sensor_profile=MLX90640, block_size=10)
        assert d.block_size == 11

    def test_rejects_no_profile(self):
        with pytest.raises(ValueError):
            AdaptiveThresholdDetector(sensor_profile=None)  # type: ignore[arg-type]

    def test_rejects_even_morph_kernel(self):
        with pytest.raises(ValueError, match="morph_kernel_size"):
            AdaptiveThresholdDetector(sensor_profile=MLX90640, morph_kernel_size=4)

    def test_rejects_invalid_area_fraction(self):
        with pytest.raises(ValueError, match="max_area_fraction"):
            AdaptiveThresholdDetector(sensor_profile=MLX90640, max_area_fraction=0.0)
        with pytest.raises(ValueError, match="max_area_fraction"):
            AdaptiveThresholdDetector(sensor_profile=MLX90640, max_area_fraction=2.0)

    def test_class_metadata(self):
        assert AdaptiveThresholdDetector.name == "adaptive_threshold_detector"
        assert AdaptiveThresholdDetector.is_trainable is False
        assert AdaptiveThresholdDetector.resolution_behavior == "parameterized"


# ---------------------------------------------------------------------------
# fit() — noop semantics
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_is_noop_but_marks_fitted(self):
        d = AdaptiveThresholdDetector(sensor_profile=MLX90640)
        assert d.is_fitted is False
        d.fit([])
        assert d.is_fitted is True

    def test_predict_works_before_fit(self):
        # Classical: predict doesn't require fit.
        d = AdaptiveThresholdDetector(sensor_profile=MLX90640)
        arr = _ambient(MLX90640, 20.0)
        _stamp_blob(arr, center=(12, 16), radius=3, value=35.0)
        out = d.predict(_frame(arr))
        assert isinstance(out, list)


# ---------------------------------------------------------------------------
# Detection on synthetic blobs
# ---------------------------------------------------------------------------

class TestDetection:
    def test_detects_clear_hot_blob(self):
        arr = _ambient(MLX90640, ambient_c=20.0)
        # Person-like tall rectangle: 6 tall × 3 wide
        _stamp_blob(arr, center=(12, 16), radius=(6, 3), value=33.0, shape="rect")

        d = AdaptiveThresholdDetector(
            sensor_profile=MLX90640,
            min_solidity=0.5,
            min_aspect_ratio=0.2,
            max_aspect_ratio=5.0,
        )
        out = d.predict(_frame(arr))
        assert len(out) == 1
        x, y, w, h = out[0].bbox
        # Bounding box should overlap the planted blob.
        assert x <= 16 and (x + w) >= 16
        assert y <= 12 and (y + h) >= 12

    def test_empty_scene_returns_empty_list(self):
        rng = np.random.default_rng(0)
        arr = _ambient(MLX90640, ambient_c=20.0, noise=0.1, rng=rng)
        d = AdaptiveThresholdDetector(sensor_profile=MLX90640)
        out = d.predict(_frame(arr))
        assert out == []

    def test_detection_inherits_camera_id(self):
        arr = _ambient(MLX90640, 20.0)
        _stamp_blob(arr, center=(12, 16), radius=(5, 3), value=33.0, shape="rect")
        d = AdaptiveThresholdDetector(sensor_profile=MLX90640)
        out = d.predict(_frame(arr, camera_id=2))
        assert out and out[0].camera_id == 2

    def test_thermal_features_populated(self):
        arr = _ambient(MLX90640, 20.0)
        _stamp_blob(arr, center=(12, 16), radius=(5, 3), value=35.0, shape="rect")
        d = AdaptiveThresholdDetector(sensor_profile=MLX90640)
        out = d.predict(_frame(arr))
        assert out
        f = out[0].thermal_features
        assert f is not None
        for key in ("max_temp", "mean_temp", "std_temp", "area_pixels",
                    "solidity", "aspect_ratio"):
            assert key in f
        assert f["max_temp"] == pytest.approx(35.0, abs=0.5)
        assert f["mean_temp"] > 30.0


# ---------------------------------------------------------------------------
# Geometric filters
# ---------------------------------------------------------------------------

class TestGeometricFilters:
    def test_rejects_too_small_blob(self):
        arr = _ambient(MLX90640, 20.0)
        _stamp_blob(arr, center=(12, 16), radius=1, value=35.0)  # 1 pixel
        d = AdaptiveThresholdDetector(sensor_profile=MLX90640, min_area_pixels=20)
        assert d.predict(_frame(arr)) == []

    def test_rejects_overly_wide_aspect(self):
        arr = _ambient(MLX90640, 20.0)
        _stamp_blob(arr, center=(12, 16), radius=(1, 12), value=35.0, shape="rect")
        d = AdaptiveThresholdDetector(
            sensor_profile=MLX90640, max_aspect_ratio=3.0, min_solidity=0.4,
        )
        # very wide (W/H ≈ 24/2 = 12) → filtered out
        assert d.predict(_frame(arr)) == []

    def test_rejects_too_large_blob(self):
        arr = _ambient(MLX90640, 20.0)
        arr[:, :] = 35.0   # entire frame is hot
        d = AdaptiveThresholdDetector(
            sensor_profile=MLX90640, max_area_fraction=0.5,
        )
        # Uniform hot frame: local mean ≈ value, so nothing exceeds it; mask
        # is empty. The test here is that the detector doesn't crash and
        # correctly rejects any oversize artifact.
        out = d.predict(_frame(arr))
        assert all(det.area <= MLX90640.num_pixels * 0.5 for det in out)


# ---------------------------------------------------------------------------
# Intensity gating
# ---------------------------------------------------------------------------

class TestIntensityGating:
    def test_pixel_value_bounds_filter_works(self):
        arr = _ambient(MLX90640, 20.0)
        _stamp_blob(arr, center=(12, 16), radius=(5, 3), value=80.0, shape="rect")
        # 80 °C is way above human range; gate should reject it.
        d = AdaptiveThresholdDetector(
            sensor_profile=MLX90640, pixel_value_bounds=(28.0, 42.0),
        )
        assert d.predict(_frame(arr)) == []

    def test_pixel_value_bounds_keeps_human_temp(self):
        arr = _ambient(MLX90640, 20.0)
        _stamp_blob(arr, center=(12, 16), radius=(5, 3), value=33.0, shape="rect")
        d = AdaptiveThresholdDetector(
            sensor_profile=MLX90640, pixel_value_bounds=(28.0, 42.0),
        )
        assert len(d.predict(_frame(arr))) == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        d = AdaptiveThresholdDetector(
            sensor_profile=MLX90640, c_offset=0.7, min_solidity=0.6,
        ).fit([])
        path = tmp_path / "atd.thalg"
        d.save(path)

        loaded = AdaptiveThresholdDetector.load(path)
        assert loaded.get_params() == d.get_params()
        assert loaded.block_size == d.block_size

    def test_save_load_predictions_identical(self, tmp_path):
        arr = _ambient(MLX90640, 20.0)
        _stamp_blob(arr, center=(12, 16), radius=(5, 3), value=33.0, shape="rect")

        original = AdaptiveThresholdDetector(sensor_profile=MLX90640).fit([])
        path = tmp_path / "atd.thalg"
        original.save(path)
        loaded = AdaptiveThresholdDetector.load(path)

        o = original.predict(_frame(arr))
        l = loaded.predict(_frame(arr))
        assert len(o) == len(l)
        for a, b in zip(o, l):
            assert a.bbox == b.bbox
