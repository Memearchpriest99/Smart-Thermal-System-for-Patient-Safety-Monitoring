"""Tests for TatenoPipeline (§ 4.4.1 reference implementation).

We exercise three planes:
    1. Construction & parameter resolution (especially the resolution-scaling
       physical→pixel σ conversion).
    2. fit() — correct mean-field background.
    3. predict() — full 3-stage pipeline, including static-scene quiescence
       and hot-blob detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984
from thermal_algorithms.core.types import Frame
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ambient_frame(profile, base_temp_c=20.0, noise_std=0.5, timestamp=0.0,
                   camera_id=0, rng=None) -> Frame:
    """Synthetic empty-room frame: uniform ambient temperature + Gaussian noise."""
    rng = rng if rng is not None else np.random.default_rng(0)
    w, h = profile.resolution
    data = rng.normal(loc=base_temp_c, scale=noise_std, size=(h, w)).astype(np.float32)
    return Frame(data=data, timestamp=timestamp, camera_id=camera_id)


def _frame_with_hot_blob(
    profile, *, blob_center, blob_radius, blob_temp_c=35.0,
    ambient_c=20.0, noise_std=0.5, timestamp=0.0, camera_id=0, rng=None,
) -> Frame:
    """Synthetic frame: ambient room + a circular hot blob (a 'person')."""
    rng = rng if rng is not None else np.random.default_rng(0)
    w, h = profile.resolution
    data = rng.normal(loc=ambient_c, scale=noise_std, size=(h, w)).astype(np.float32)
    cy, cx = blob_center
    ys, xs = np.ogrid[:h, :w]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= blob_radius ** 2
    data[mask] = blob_temp_c
    return Frame(data=data, timestamp=timestamp, camera_id=camera_id)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_constructs_with_mlx(self):
        p = TatenoPipeline(sensor_profile=MLX90640)
        assert p.sensor_profile is MLX90640
        assert p.sigma > 0
        assert p.kernel_size >= 3 and p.kernel_size % 2 == 1

    def test_constructs_with_waveshare(self):
        p = TatenoPipeline(sensor_profile=WAVESHARE_26984)
        assert p.sigma > 0
        assert p.kernel_size >= 3 and p.kernel_size % 2 == 1

    def test_explicit_sigma_overrides_physical(self):
        p = TatenoPipeline(sensor_profile=MLX90640, sigma=1.5)
        assert p.sigma == 1.5

    def test_explicit_kernel_size_overrides(self):
        p = TatenoPipeline(sensor_profile=MLX90640, sigma=1.0, kernel_size=7)
        assert p.kernel_size == 7

    def test_rejects_no_profile(self):
        with pytest.raises(ValueError):
            TatenoPipeline(sensor_profile=None)  # type: ignore[arg-type]

    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            TatenoPipeline(sensor_profile=MLX90640, sigma=-0.5)

    def test_rejects_even_kernel(self):
        with pytest.raises(ValueError, match="odd"):
            TatenoPipeline(sensor_profile=MLX90640, sigma=1.0, kernel_size=4)

    def test_class_metadata(self):
        assert TatenoPipeline.name == "tateno_pipeline"
        assert TatenoPipeline.is_trainable is False
        assert TatenoPipeline.resolution_behavior == "parameterized"


# ---------------------------------------------------------------------------
# Resolution scaling
# ---------------------------------------------------------------------------

class TestResolutionScaling:
    def test_same_physical_smoothing_yields_different_pixel_sigma(self):
        """A 5 cm smoothing scale should correspond to *more* pixels on the
        higher-resolution Waveshare than on the MLX, given roughly comparable
        FOVs. The exact ratio depends on FOV; the qualitative check is that
        the higher-res sensor gets a larger σ in pixels."""
        p_mlx = TatenoPipeline(sensor_profile=MLX90640, physical_smoothing_m=0.05)
        p_ws = TatenoPipeline(sensor_profile=WAVESHARE_26984, physical_smoothing_m=0.05)
        # Waveshare: 80 px over 60° vs MLX: 32 px over 55° → Waveshare has finer
        # angular resolution → same physical smoothing covers more pixels.
        assert p_ws.sigma > p_mlx.sigma

    def test_zero_distance_rejected(self):
        with pytest.raises(ValueError):
            TatenoPipeline(
                sensor_profile=MLX90640,
                physical_smoothing_m=0.05,
                assumed_distance_m=0.0,
            )


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_computes_mean_background(self):
        """With low-noise calibration frames, the learned background ≈ ambient
        temperature (after smoothing softens the spatial noise)."""
        rng = np.random.default_rng(42)
        calib = [
            _ambient_frame(MLX90640, base_temp_c=20.0, noise_std=0.1, timestamp=i, rng=rng)
            for i in range(30)
        ]
        p = TatenoPipeline(sensor_profile=MLX90640)
        p.fit(calib)
        assert p.is_fitted is True
        assert p.background is not None
        bg_mean = p.background.mean()
        assert bg_mean == pytest.approx(20.0, abs=0.1)

    def test_fit_accepts_ndarray(self):
        rng = np.random.default_rng(0)
        stacked = rng.normal(20.0, 0.2, size=(10, 24, 32)).astype(np.float32)
        p = TatenoPipeline(sensor_profile=MLX90640).fit(stacked)
        assert p.is_fitted

    def test_fit_rejects_wrong_shape(self):
        bad = [Frame(data=np.zeros((10, 10)), timestamp=0.0)]
        p = TatenoPipeline(sensor_profile=MLX90640)
        with pytest.raises(ValueError, match="shape"):
            p.fit(bad)

    def test_fit_rejects_empty(self):
        p = TatenoPipeline(sensor_profile=MLX90640)
        with pytest.raises(ValueError, match="zero calibration"):
            p.fit([])

    def test_fit_returns_self(self):
        p = TatenoPipeline(sensor_profile=MLX90640)
        rng = np.random.default_rng(0)
        calib = [_ambient_frame(MLX90640, rng=rng) for _ in range(5)]
        assert p.fit(calib) is p


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_errors_before_fit(self):
        p = TatenoPipeline(sensor_profile=MLX90640)
        f = _ambient_frame(MLX90640)
        with pytest.raises(RuntimeError, match="before fit"):
            p.predict(f)

    def test_predict_preserves_timestamp_and_camera(self):
        rng = np.random.default_rng(0)
        calib = [_ambient_frame(MLX90640, timestamp=i, rng=rng) for i in range(5)]
        p = TatenoPipeline(sensor_profile=MLX90640).fit(calib)
        f = _ambient_frame(MLX90640, timestamp=99.5, camera_id=2)
        out = p.predict(f)
        assert out.timestamp == 99.5
        assert out.camera_id == 2
        assert out.shape == f.shape

    def test_predict_metadata_tagged(self):
        rng = np.random.default_rng(0)
        calib = [_ambient_frame(MLX90640, rng=rng) for _ in range(5)]
        p = TatenoPipeline(sensor_profile=MLX90640).fit(calib)
        out = p.predict(_ambient_frame(MLX90640))
        assert out.metadata.get("preprocessed_by") == "tateno_pipeline"

    def test_output_is_nonnegative(self):
        """Stage 3 (L1 rectification) guarantees |·| ≥ 0 everywhere."""
        rng = np.random.default_rng(0)
        calib = [_ambient_frame(MLX90640, base_temp_c=20.0, rng=rng) for _ in range(20)]
        p = TatenoPipeline(sensor_profile=MLX90640).fit(calib)
        # Use a *cold* frame to stress the negative-residual case.
        cold = _frame_with_hot_blob(
            MLX90640, blob_center=(12, 16), blob_radius=4, blob_temp_c=5.0,
            ambient_c=20.0, noise_std=0.2, rng=rng,
        )
        out = p.predict(cold)
        assert (out.data >= 0).all()

    def test_static_scene_yields_small_residual(self):
        """Calibrate then feed back a similar empty-room frame — residual should
        be small everywhere (noise-floor level)."""
        rng = np.random.default_rng(123)
        calib = [_ambient_frame(MLX90640, noise_std=0.3, timestamp=i, rng=rng) for i in range(40)]
        p = TatenoPipeline(sensor_profile=MLX90640).fit(calib)
        empty = _ambient_frame(MLX90640, noise_std=0.3, timestamp=99, rng=rng)
        out = p.predict(empty)
        assert out.data.mean() < 0.5  # well below the 30 °C target signal in the next test

    def test_hot_blob_dominates_residual(self):
        """A 35 °C blob over a 20 °C calibrated background should produce a
        strong residual centered on the blob."""
        rng = np.random.default_rng(7)
        calib = [_ambient_frame(MLX90640, base_temp_c=20.0, noise_std=0.3, rng=rng)
                 for _ in range(40)]
        p = TatenoPipeline(sensor_profile=MLX90640).fit(calib)
        cy, cx = 12, 16
        frame = _frame_with_hot_blob(
            MLX90640, blob_center=(cy, cx), blob_radius=3,
            blob_temp_c=35.0, ambient_c=20.0, noise_std=0.3, rng=rng,
        )
        out = p.predict(frame)
        # The residual should peak inside the blob.
        argmax_yx = np.unravel_index(out.data.argmax(), out.data.shape)
        assert abs(argmax_yx[0] - cy) <= 3
        assert abs(argmax_yx[1] - cx) <= 3
        # The blob residual should be large compared to the background residual.
        background_mask = np.ones_like(out.data, dtype=bool)
        ys, xs = np.ogrid[:out.data.shape[0], :out.data.shape[1]]
        background_mask[(ys - cy) ** 2 + (xs - cx) ** 2 <= 5 ** 2] = False
        assert out.data.max() > 5 * out.data[background_mask].mean()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_preserves_background(self, tmp_path):
        rng = np.random.default_rng(0)
        calib = [_ambient_frame(MLX90640, base_temp_c=20.0, rng=rng) for _ in range(15)]
        p = TatenoPipeline(sensor_profile=MLX90640, physical_smoothing_m=0.07).fit(calib)
        path = tmp_path / "tateno.thalg"
        p.save(path)

        restored = TatenoPipeline.load(path)
        assert restored.is_fitted
        np.testing.assert_array_almost_equal(restored.background, p.background)
        assert restored.sigma == p.sigma
        assert restored.kernel_size == p.kernel_size

    def test_save_load_produces_identical_residual(self, tmp_path):
        rng = np.random.default_rng(1)
        calib = [_ambient_frame(MLX90640, rng=rng) for _ in range(20)]
        p = TatenoPipeline(sensor_profile=MLX90640).fit(calib)
        path = tmp_path / "tateno.thalg"
        p.save(path)
        loaded = TatenoPipeline.load(path)

        f = _frame_with_hot_blob(
            MLX90640, blob_center=(10, 10), blob_radius=3, rng=rng,
        )
        np.testing.assert_array_almost_equal(p.predict(f).data, loaded.predict(f).data)


# ---------------------------------------------------------------------------
# transform() / fit_transform() aliases inherited from Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessorAliases:
    def test_transform_equals_predict(self):
        rng = np.random.default_rng(0)
        calib = [_ambient_frame(MLX90640, rng=rng) for _ in range(5)]
        p = TatenoPipeline(sensor_profile=MLX90640).fit(calib)
        f = _ambient_frame(MLX90640)
        np.testing.assert_array_equal(p.predict(f).data, p.transform(f).data)

    def test_fit_transform_returns_processed_calibration(self):
        rng = np.random.default_rng(0)
        calib = [_ambient_frame(MLX90640, rng=rng) for _ in range(5)]
        p = TatenoPipeline(sensor_profile=MLX90640)
        out = p.fit_transform(calib)
        assert len(out) == len(calib)
        # Calibration frames passed back through the pipeline should yield
        # near-zero residuals.
        for frame in out:
            assert frame.data.mean() < 1.0
