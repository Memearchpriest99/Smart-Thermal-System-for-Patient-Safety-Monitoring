"""Contract tests for SensorProfile and the pre-defined presets."""

import math

import pytest

from thermal_algorithms.core.sensor_profile import (
    SensorProfile,
    MLX90640,
    WAVESHARE_26984,
    PROFILES,
)


class TestSensorProfile:
    def test_mlx_geometry(self):
        assert MLX90640.resolution == (32, 24)
        assert MLX90640.num_pixels == 768
        assert MLX90640.width == 32 and MLX90640.height == 24
        assert math.isclose(MLX90640.aspect_ratio, 32 / 24)

    def test_waveshare_geometry(self):
        assert WAVESHARE_26984.resolution == (80, 62)
        assert WAVESHARE_26984.num_pixels == 80 * 62

    def test_registry_contains_both(self):
        assert MLX90640.name in PROFILES
        assert WAVESHARE_26984.name in PROFILES
        assert PROFILES[MLX90640.name] is MLX90640

    def test_angular_resolution_positive(self):
        dx_deg, dy_deg = MLX90640.angular_resolution_deg()
        assert dx_deg > 0 and dy_deg > 0

    def test_physical_pixel_size_scales_with_distance(self):
        # At twice the range, each pixel covers twice the real-world width.
        a = MLX90640.physical_pixel_size_m(distance_m=1.0)
        b = MLX90640.physical_pixel_size_m(distance_m=2.0)
        assert math.isclose(b[0] / a[0], 2.0, rel_tol=1e-6)
        assert math.isclose(b[1] / a[1], 2.0, rel_tol=1e-6)

    def test_rejects_zero_resolution(self):
        with pytest.raises(ValueError, match="resolution"):
            SensorProfile(
                name="bad", resolution=(0, 1), fov_deg=(10, 10),
                sample_rate_hz=1.0, noise_floor_c=1.0,
            )

    def test_rejects_invalid_fov(self):
        with pytest.raises(ValueError, match="fov"):
            SensorProfile(
                name="bad", resolution=(1, 1), fov_deg=(200, 10),
                sample_rate_hz=1.0, noise_floor_c=1.0,
            )

    def test_rejects_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            SensorProfile(
                name="bad", resolution=(1, 1), fov_deg=(10, 10),
                sample_rate_hz=0.0, noise_floor_c=1.0,
            )

    def test_frozen(self):
        with pytest.raises(Exception):
            MLX90640.sample_rate_hz = 16.0  # type: ignore[misc]
