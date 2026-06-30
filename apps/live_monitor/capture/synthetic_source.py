"""Procedural warm-blob source for smoke tests and dependency-free demos.

Generates a cool background with one or more moving warm "people" blobs and an
optional hot "fire" blob, at the sensor profile's native resolution, paced to
``sample_rate_hz``. Useful to exercise the full UI/threading/render path with no
hardware and no dataset on disk.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Frame
from apps.live_monitor.capture.base import FrameSource


class SyntheticSource(FrameSource):
    """A deterministic-ish moving warm-blob generator.

    Args:
        camera_id: 0/1/2.
        profile: sensor geometry (sets frame shape + frame rate).
        n_actors: number of moving warm blobs.
        with_fire: if True, add a small very-hot blob (for fire-path testing).
        ambient_c / body_c / fire_c: temperatures for background / people / fire.
        seed: jitters phase per camera so the three views differ.
    """

    def __init__(
        self,
        camera_id: int,
        profile: SensorProfile,
        *,
        n_actors: int = 2,
        with_fire: bool = False,
        ambient_c: float = 24.0,
        body_c: float = 34.0,
        fire_c: float = 120.0,
        seed: int = 0,
    ) -> None:
        super().__init__(camera_id, profile)
        self._n_actors = max(0, int(n_actors))
        self._with_fire = bool(with_fire)
        self._ambient_c = float(ambient_c)
        self._body_c = float(body_c)
        self._fire_c = float(fire_c)
        self._rng = np.random.default_rng(seed + camera_id * 1000)
        self._phase0 = (seed + camera_id) * 0.7
        self._t0: Optional[float] = None
        self._min_dt = 1.0 / float(profile.sample_rate_hz)
        self._last_emit: Optional[float] = None

    def _open(self) -> None:
        self._t0 = time.monotonic()
        self._last_emit = None
        self._health.connected = True

    def _close(self) -> None:
        self._t0 = None

    def _read_raw(self) -> Optional[Frame]:
        now = time.monotonic()
        # Throttle to the sensor frame rate so we behave like real hardware
        # (return None when no new frame is due yet).
        if self._last_emit is not None and (now - self._last_emit) < self._min_dt:
            return None
        self._last_emit = now
        t = now - (self._t0 or now)
        data = self._render(t)
        return Frame(data=data, timestamp=t, camera_id=self._camera_id)

    def _render(self, t: float) -> np.ndarray:
        w, h = self._profile.width, self._profile.height
        img = np.full((h, w), self._ambient_c, dtype=np.float32)
        img += self._rng.normal(0.0, self._profile.noise_floor_c * 0.3, size=(h, w)).astype(np.float32)

        # Moving people: blobs that drift and (for 2 actors) periodically meet,
        # which lets the contact/touch path fire.
        for a in range(self._n_actors):
            phase = self._phase0 + a * math.pi
            if self._n_actors == 2:
                # Two actors oscillate toward/away from the centre → contact.
                sep = 0.30 + 0.28 * (0.5 + 0.5 * math.cos(0.5 * t))
                cx = w * (0.5 + (sep if a == 0 else -sep) * 0.5)
                cy = h * (0.5 + 0.06 * math.sin(0.7 * t + phase))
            else:
                cx = w * (0.5 + 0.30 * math.cos(0.4 * t + phase))
                cy = h * (0.5 + 0.30 * math.sin(0.3 * t + phase))
            self._add_blob(img, cx, cy, radius=max(2.0, w * 0.06), peak=self._body_c)

        if self._with_fire:
            fx = w * (0.18 + 0.02 * math.sin(2.0 * t))
            fy = h * 0.78
            self._add_blob(img, fx, fy, radius=max(1.5, w * 0.03), peak=self._fire_c)

        return img

    @staticmethod
    def _add_blob(img: np.ndarray, cx: float, cy: float, radius: float, peak: float) -> None:
        h, w = img.shape
        ys, xs = np.ogrid[0:h, 0:w]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        gauss = np.exp(-d2 / (2.0 * radius * radius))
        ambient = img.min()
        img[:] = np.maximum(img, ambient + (peak - ambient) * gauss).astype(np.float32)
