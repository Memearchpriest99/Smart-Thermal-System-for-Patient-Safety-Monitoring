"""Tests for GlobalNormPreprocessor — the Task 2 comparison baseline.

Mirrors test_tateno_pipeline.py's structure so the two are easy to compare
directly; differs where the algorithms genuinely differ (no learned
background here, so no `background` property and `fit()` needs no
calibration frames at all).
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984
from thermal_algorithms.core.types import Frame
from thermal_algorithms.preprocessing.global_norm_pipeline import GlobalNormPreprocessor
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline


def _ambient_frame(profile, base_temp_c=20.0, noise_std=0.5, timestamp=0.0,
                   camera_id=0, rng=None) -> Frame:
    rng = rng if rng is not None else np.random.default_rng(0)
    w, h = profile.resolution
    data = rng.normal(loc=base_temp_c, scale=noise_std, size=(h, w)).astype(np.float32)
    return Frame(data=data, timestamp=timestamp, camera_id=camera_id)


def _frame_with_hot_blob(
    profile, *, blob_center, blob_radius, blob_temp_c=35.0,
    ambient_c=20.0, noise_std=0.5, timestamp=0.0, camera_id=0, rng=None,
) -> Frame:
    rng = rng if rng is not None else np.random.default_rng(0)
    w, h = profile.resolution
    data = rng.normal(loc=ambient_c, scale=noise_std, size=(h, w)).astype(np.float32)
    cy, cx = blob_center
    ys, xs = np.ogrid[:h, :w]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= blob_radius ** 2
    data[mask] = blob_temp_c
    return Frame(data=data, timestamp=timestamp, camera_id=camera_id)


class TestConstruction:
    def test_constructs_with_mlx(self):
        p = GlobalNormPreprocessor(sensor_profile=MLX90640)
        assert p.sensor_profile is MLX90640
        assert p.sigma > 0
        assert p.kernel_size >= 3 and p.kernel_size % 2 == 1

    def test_constructs_with_waveshare(self):
        p = GlobalNormPreprocessor(sensor_profile=WAVESHARE_26984)
        assert p.sigma > 0

    def test_explicit_sigma_overrides_physical(self):
        p = GlobalNormPreprocessor(sensor_profile=MLX90640, sigma=1.5)
        assert p.sigma == 1.5

    def test_rejects_no_profile(self):
        with pytest.raises(ValueError):
            GlobalNormPreprocessor(sensor_profile=None)  # type: ignore[arg-type]

    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            GlobalNormPreprocessor(sensor_profile=MLX90640, sigma=-0.5)

    def test_rejects_even_kernel(self):
        with pytest.raises(ValueError, match="odd"):
            GlobalNormPreprocessor(sensor_profile=MLX90640, sigma=1.0, kernel_size=4)

    def test_class_metadata(self):
        assert GlobalNormPreprocessor.name == "global_norm_pipeline"
        assert GlobalNormPreprocessor.is_trainable is False
        assert GlobalNormPreprocessor.resolution_behavior == "parameterized"

    def test_same_smoothing_convention_as_tateno(self):
        """The two pipelines should resolve sigma/kernel identically given the
        same physical smoothing scale — the comparison should isolate the
        background-estimation strategy, not be confounded by different blur."""
        a = GlobalNormPreprocessor(sensor_profile=MLX90640, physical_smoothing_m=0.05)
        b = TatenoPipeline(sensor_profile=MLX90640, physical_smoothing_m=0.05)
        assert a.sigma == pytest.approx(b.sigma)
        assert a.kernel_size == b.kernel_size


class TestFit:
    def test_fit_with_no_args_returns_self_and_marks_fitted(self):
        p = GlobalNormPreprocessor(sensor_profile=MLX90640)
        assert p.is_fitted is False
        result = p.fit()
        assert result is p
        assert p.is_fitted is True

    def test_fit_ignores_any_calibration_frames_given(self):
        """Unlike TatenoPipeline, passing frames changes nothing — there's no
        state to learn from them."""
        rng = np.random.default_rng(0)
        calib = [_ambient_frame(MLX90640, base_temp_c=99.0, rng=rng) for _ in range(5)]
        p = GlobalNormPreprocessor(sensor_profile=MLX90640).fit(calib)
        assert p.is_fitted is True
        # No background attribute exists to inspect — the point is there's
        # nothing calibration-dependent to check.
        assert not hasattr(p, "background")


class TestPredict:
    def test_predict_errors_before_fit(self):
        p = GlobalNormPreprocessor(sensor_profile=MLX90640)
        f = _ambient_frame(MLX90640)
        with pytest.raises(RuntimeError, match="before fit"):
            p.predict(f)

    def test_predict_preserves_timestamp_and_camera(self):
        p = GlobalNormPreprocessor(sensor_profile=MLX90640).fit()
        f = _ambient_frame(MLX90640, timestamp=99.5, camera_id=2)
        out = p.predict(f)
        assert out.timestamp == 99.5
        assert out.camera_id == 2
        assert out.shape == f.shape

    def test_predict_metadata_tagged(self):
        p = GlobalNormPreprocessor(sensor_profile=MLX90640).fit()
        out = p.predict(_ambient_frame(MLX90640))
        assert out.metadata.get("preprocessed_by") == "global_norm_pipeline"
        assert "frame_mean_c" in out.metadata

    def test_output_is_nonnegative(self):
        p = GlobalNormPreprocessor(sensor_profile=MLX90640).fit()
        rng = np.random.default_rng(0)
        frame = _frame_with_hot_blob(
            MLX90640, blob_center=(12, 16), blob_radius=4, blob_temp_c=5.0,
            ambient_c=20.0, noise_std=0.2, rng=rng,
        )
        out = p.predict(frame)
        assert (out.data >= 0).all()

    def test_uniform_frame_yields_near_zero_residual(self):
        """No calibration needed: a spatially-uniform frame's own mean
        subtraction is near-zero everywhere by construction."""
        rng = np.random.default_rng(123)
        p = GlobalNormPreprocessor(sensor_profile=MLX90640, physical_smoothing_m=0.05).fit()
        empty = _ambient_frame(MLX90640, noise_std=0.3, rng=rng)
        out = p.predict(empty)
        assert out.data.mean() < 0.5

    def test_hot_blob_dominates_residual(self):
        """Same qualitative check as TatenoPipeline: a hot blob should
        dominate the residual and be spatially localized."""
        rng = np.random.default_rng(7)
        p = GlobalNormPreprocessor(sensor_profile=MLX90640, physical_smoothing_m=0.05).fit()
        cy, cx = 12, 16
        frame = _frame_with_hot_blob(
            MLX90640, blob_center=(cy, cx), blob_radius=3,
            blob_temp_c=35.0, ambient_c=20.0, noise_std=0.3, rng=rng,
        )
        out = p.predict(frame)
        argmax_yx = np.unravel_index(out.data.argmax(), out.data.shape)
        assert abs(argmax_yx[0] - cy) <= 3
        assert abs(argmax_yx[1] - cx) <= 3
        background_mask = np.ones_like(out.data, dtype=bool)
        ys, xs = np.ogrid[:out.data.shape[0], :out.data.shape[1]]
        background_mask[(ys - cy) ** 2 + (xs - cx) ** 2 <= 5 ** 2] = False
        assert out.data.max() > 5 * out.data[background_mask].mean()

    def test_background_is_frame_own_mean_recomputed_every_call(self):
        """The defining difference from Tateno: no persisted background — two
        frames with different ambient temperatures each get normalized
        relative to their OWN mean, not a shared learned one."""
        p = GlobalNormPreprocessor(sensor_profile=MLX90640, physical_smoothing_m=0.05).fit()
        rng = np.random.default_rng(0)
        cold_ambient = _ambient_frame(MLX90640, base_temp_c=15.0, noise_std=0.1, rng=rng)
        hot_ambient = _ambient_frame(MLX90640, base_temp_c=35.0, noise_std=0.1, rng=rng)
        # Both are uniform frames far apart in absolute temperature, but each
        # should normalize to a small residual relative to its OWN mean.
        assert p.predict(cold_ambient).data.mean() < 0.5
        assert p.predict(hot_ambient).data.mean() < 0.5


class TestPersistence:
    def test_save_load_preserves_resolved_params(self, tmp_path):
        p = GlobalNormPreprocessor(sensor_profile=MLX90640, physical_smoothing_m=0.07).fit()
        path = tmp_path / "global_norm.thalg"
        p.save(path)

        restored = GlobalNormPreprocessor.load(path)
        assert restored.is_fitted
        assert restored.sigma == p.sigma
        assert restored.kernel_size == p.kernel_size

    def test_save_load_produces_identical_residual(self, tmp_path):
        p = GlobalNormPreprocessor(sensor_profile=MLX90640).fit()
        path = tmp_path / "global_norm.thalg"
        p.save(path)
        loaded = GlobalNormPreprocessor.load(path)

        rng = np.random.default_rng(1)
        f = _frame_with_hot_blob(MLX90640, blob_center=(10, 10), blob_radius=3, rng=rng)
        np.testing.assert_array_almost_equal(p.predict(f).data, loaded.predict(f).data)


class TestPreprocessorAliases:
    def test_transform_equals_predict(self):
        p = GlobalNormPreprocessor(sensor_profile=MLX90640).fit()
        f = _ambient_frame(MLX90640)
        np.testing.assert_array_equal(p.predict(f).data, p.transform(f).data)

    def test_fit_transform_returns_processed_frames(self):
        rng = np.random.default_rng(0)
        frames = [_ambient_frame(MLX90640, rng=rng) for _ in range(5)]
        p = GlobalNormPreprocessor(sensor_profile=MLX90640)
        out = p.fit_transform(frames)
        assert len(out) == len(frames)
        for frame in out:
            assert frame.data.mean() < 1.0
