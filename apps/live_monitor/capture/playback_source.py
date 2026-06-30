"""Replay a recorded session as if it were a live camera.

Reads ``<session>/ch{camera_id}_raw_data.npz`` (key ``"frames"``, shape
``(N, H, W)`` in °C — the layout produced by ``scripts/reorganize_waveshare.py``
and consumed across the repo) and emits frames paced to the profile frame rate,
looping at the end. This is what lets the entire app run and be verified on a
machine with no thermal hardware.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Frame
from apps.live_monitor.capture.base import FrameSource


class PlaybackSource(FrameSource):
    """Replays one channel's recorded frames at the sensor frame rate.

    Args:
        camera_id: 0/1/2 — also selects ``ch{camera_id}_raw_data.npz``.
        profile: sensor geometry (used for the pacing rate and a sanity check).
        session_dir: directory holding the ``chN_raw_data.npz`` files.
        loop: restart from frame 0 at the end (default True).
        speed: playback speed multiplier (1.0 = real time).
        npz_key: array key inside the npz (default ``"frames"``).
    """

    def __init__(
        self,
        camera_id: int,
        profile: SensorProfile,
        *,
        session_dir: str | Path,
        loop: bool = True,
        speed: float = 1.0,
        npz_key: str = "frames",
    ) -> None:
        super().__init__(camera_id, profile)
        self._session_dir = Path(session_dir)
        self._loop = bool(loop)
        self._speed = max(1e-3, float(speed))
        self._npz_key = npz_key
        self._frames: Optional[np.ndarray] = None
        self._idx = 0
        self._t0: Optional[float] = None
        self._min_dt = 1.0 / (float(profile.sample_rate_hz) * self._speed)
        self._last_emit: Optional[float] = None

    @property
    def npz_path(self) -> Path:
        return self._session_dir / f"ch{self._camera_id}_raw_data.npz"

    @property
    def n_frames(self) -> int:
        return 0 if self._frames is None else int(self._frames.shape[0])

    def _open(self) -> None:
        path = self.npz_path
        if not path.is_file():
            raise FileNotFoundError(
                f"No recording for camera {self._camera_id} at {path}. "
                f"Expected a '{self._npz_key}' array of shape (N, H, W)."
            )
        with np.load(path) as npz:
            if self._npz_key not in npz:
                raise KeyError(
                    f"{path} has keys {list(npz.keys())}; expected '{self._npz_key}'."
                )
            self._frames = npz[self._npz_key].astype(np.float32)
        if self._frames.ndim != 3 or self._frames.shape[0] == 0:
            raise ValueError(
                f"{path}: expected (N, H, W) with N>0, got {self._frames.shape}."
            )
        fh, fw = self._frames.shape[1:]
        if (fw, fh) != (self._profile.width, self._profile.height):
            # Not fatal — just flag it; algorithms key off the actual array.
            self._health.extra["resolution_warning"] = (
                f"recording is {fw}x{fh}, profile {self._profile.name} expects "
                f"{self._profile.width}x{self._profile.height}"
            )
        self._idx = 0
        self._t0 = time.monotonic()
        self._last_emit = None
        self._health.connected = True
        self._health.extra["n_frames"] = self.n_frames
        self._health.extra["session"] = self._session_dir.name

    def _close(self) -> None:
        self._frames = None

    def _read_raw(self) -> Optional[Frame]:
        if self._frames is None:
            return None
        now = time.monotonic()
        if self._last_emit is not None and (now - self._last_emit) < self._min_dt:
            return None

        if self._idx >= self.n_frames:
            if not self._loop:
                return None
            self._idx = 0

        arr = self._frames[self._idx]
        # Timestamp from the frame index so downstream pacing/Kalman dt is
        # consistent regardless of wall-clock jitter.
        ts = self._idx / float(self._profile.sample_rate_hz)
        frame = Frame(data=np.array(arr, dtype=np.float32), timestamp=ts, camera_id=self._camera_id)
        self._idx += 1
        self._last_emit = now
        self._health.extra["frame_index"] = self._idx
        return frame
