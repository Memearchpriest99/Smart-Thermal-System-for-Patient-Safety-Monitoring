"""Sensor profile — the resolution/FOV/calibration config every resolution-aware
algorithm consumes.

The library supports two hardware setups in parallel (see § 4.1.2 and the project
plan):
    * MLX90640        — 32 x 24 thermal array, 8 Hz default
    * Waveshare 26984 — 80 x 62 thermal array, 8 Hz default

Algorithms declare their behavior via `ThermalAlgorithm.resolution_behavior`:
    * "invariant"     — same code, same checkpoint, both sensors work
                        (e.g., Geometric Contact, Fire SVM, GCN body of MV-STGCN)
    * "parameterized" — same architecture, kernel/grid sizes scale with profile
                        (e.g., Tateno preprocessing, Adaptive Threshold, Otsu pipeline)
    * "fixed"         — one trained checkpoint per resolution, loaded via the
                        checkpoint registry (e.g., MobileNet-SSD, YOLO-Nano, X3D)

The profile is threaded through algorithms in their constructor or via
`.set_sensor(profile)`. Pure invariant algorithms ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import tan, radians
from typing import Final


@dataclass(frozen=True)
class SensorProfile:
    """Static description of a thermal sensor and its operating mode.

    Attributes:
        name: Stable identifier used as a key in the checkpoint registry.
            Must be unique across registered profiles.
        resolution: (width, height) in pixels.
        fov_deg: (horizontal, vertical) field of view in degrees.
        sample_rate_hz: Default capture rate (Hz). Programmable on both sensors;
            this is the value the rest of the system assumes.
        noise_floor_c: ΔT_nom — the minimum dynamic range below which a frame
            is treated as static background by the variance gate in the Otsu
            fire pipeline (§ 4.4.4 step a).
        temp_range_c: (min, max) measurable temperature in °C. Used by display
            normalization and out-of-range detection.
        time_constant_ms: Approximate thermal response time. Bounds the
            usefulness of high frame rates (capturing faster than 1/τ yields
            duplicate readings).

    Derived quantities (properties) are not stored; they're computed from the
    fields above so the profile remains a pure config object.
    """

    name: str
    resolution: tuple[int, int]
    fov_deg: tuple[float, float]
    sample_rate_hz: float
    noise_floor_c: float
    temp_range_c: tuple[float, float] = (-40.0, 300.0)
    time_constant_ms: float = 100.0

    def __post_init__(self) -> None:
        w, h = self.resolution
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive; got ({w}, {h}).")
        fh, fv = self.fov_deg
        if not (0 < fh < 180 and 0 < fv < 180):
            raise ValueError(f"fov_deg must be in (0, 180); got ({fh}, {fv}).")
        if self.sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive; got {self.sample_rate_hz}.")

    # ---- Derived geometry --------------------------------------------------

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def num_pixels(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def angular_resolution_deg(self) -> tuple[float, float]:
        """Degrees per pixel along each axis. Useful for converting physical
        kernel sizes (e.g., 'smooth over 10cm at 2m range') into pixel kernel
        sizes when parameterizing resolution-aware classical algorithms."""
        fh, fv = self.fov_deg
        return (fh / self.width, fv / self.height)

    def physical_pixel_size_m(self, distance_m: float) -> tuple[float, float]:
        """Real-world footprint of a single pixel at the given range.

        Returns (dx, dy) in meters. Used by classical algorithms to translate
        physical kernel sizes (e.g., 'smooth over 10 cm') into pixel sizes
        regardless of which sensor is mounted.
        """
        if distance_m <= 0:
            raise ValueError(f"distance_m must be positive; got {distance_m}.")
        fh, fv = self.fov_deg
        scene_w = 2.0 * distance_m * tan(radians(fh) / 2.0)
        scene_h = 2.0 * distance_m * tan(radians(fv) / 2.0)
        return (scene_w / self.width, scene_h / self.height)


# ---------------------------------------------------------------------------
# Pre-defined profiles
# ---------------------------------------------------------------------------
# These values reflect the project's hardware choices (§ 4.1.2). FOV values
# correspond to the variants used in the prototype; revisit if the team
# switches to a different optical variant.

MLX90640: Final[SensorProfile] = SensorProfile(
    name="MLX90640",
    resolution=(32, 24),
    fov_deg=(55.0, 35.0),            # standard BAA variant (55° × 35°)
    sample_rate_hz=8.0,              # § 4.4.3 baseline operating rate
    noise_floor_c=1.5,               # ΔT_nom from § 4.4.4 step (a)
    temp_range_c=(-40.0, 300.0),
    time_constant_ms=124.0,          # MLX90640 datasheet, refresh-rate dependent
)


WAVESHARE_26984: Final[SensorProfile] = SensorProfile(
    name="Waveshare_26984",
    resolution=(80, 62),
    fov_deg=(60.0, 45.0),            # Waveshare 26984 product spec — confirm with vendor
    sample_rate_hz=8.0,
    noise_floor_c=0.7,               # ~half the MLX floor; refine with bench measurement
    temp_range_c=(-20.0, 400.0),
    time_constant_ms=80.0,
)


PROFILES: Final[dict[str, SensorProfile]] = {
    MLX90640.name: MLX90640,
    WAVESHARE_26984.name: WAVESHARE_26984,
}
"""Registry of all known sensor profiles, keyed by name.

Algorithm code that needs to resolve a profile by name (e.g., when loading a
checkpoint tagged with a sensor name) should look it up here rather than
hard-coding the constants.
"""
