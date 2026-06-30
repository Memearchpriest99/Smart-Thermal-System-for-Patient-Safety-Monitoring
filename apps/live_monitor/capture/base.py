"""``FrameSource`` — the capture abstraction.

A source produces single-camera thermal frames on demand. The app runs one
source per physical camera, each polled from its own thread, and a downstream
worker assembles synchronized triplets.

Contract:
    * :meth:`start` — open the device / file / generator. Idempotent.
    * :meth:`read`  — return the *latest* available :class:`Frame`, or ``None``
                      if no new frame is ready. Must be non-blocking-ish
                      (bounded wait); never raise on a transient read miss.
    * :meth:`stop`  — release resources. Idempotent.

Sources expose ``profile`` (the :class:`SensorProfile` describing geometry) and
``camera_id`` (0/1/2), plus a small :class:`CameraHealth` snapshot used by the
debug HUD.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Frame


@dataclass
class CameraHealth:
    """Lightweight liveness snapshot for one camera (read by the debug HUD)."""

    camera_id: int
    connected: bool = False
    frames_read: int = 0
    last_frame_monotonic: Optional[float] = None
    measured_fps: float = 0.0
    last_error: str = ""
    extra: dict = field(default_factory=dict)

    def age_s(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds since the last successful frame, or ``None`` if never."""
        if self.last_frame_monotonic is None:
            return None
        return (now if now is not None else time.monotonic()) - self.last_frame_monotonic


class FrameSource(ABC):
    """Abstract single-camera thermal frame source."""

    def __init__(self, camera_id: int, profile: SensorProfile) -> None:
        self._camera_id = int(camera_id)
        self._profile = profile
        self._health = CameraHealth(camera_id=self._camera_id)
        self._started = False
        # Exponential-moving-average FPS estimate.
        self._last_read_monotonic: Optional[float] = None

    # ---- Identity -------------------------------------------------------

    @property
    def camera_id(self) -> int:
        return self._camera_id

    @property
    def profile(self) -> SensorProfile:
        return self._profile

    @property
    def health(self) -> CameraHealth:
        return self._health

    @property
    def started(self) -> bool:
        return self._started

    # ---- Lifecycle ------------------------------------------------------

    @abstractmethod
    def _open(self) -> None:
        """Backend-specific open. Set ``self._health.connected``."""

    @abstractmethod
    def _read_raw(self) -> Optional[Frame]:
        """Backend-specific read of the newest frame, or ``None`` if not ready."""

    @abstractmethod
    def _close(self) -> None:
        """Backend-specific resource release."""

    def start(self) -> "FrameSource":
        if self._started:
            return self
        self._open()
        self._started = True
        return self

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self._close()
        finally:
            self._started = False
            self._health.connected = False

    def read(self) -> Optional[Frame]:
        """Return the latest frame and update health bookkeeping."""
        if not self._started:
            return None
        try:
            frame = self._read_raw()
        except Exception as exc:  # never let a transient read kill the thread
            self._health.last_error = f"{type(exc).__name__}: {exc}"
            return None
        if frame is None:
            return None
        self._record_read()
        return frame

    # ---- Bookkeeping ----------------------------------------------------

    def _record_read(self) -> None:
        now = time.monotonic()
        h = self._health
        h.frames_read += 1
        if self._last_read_monotonic is not None:
            dt = now - self._last_read_monotonic
            if dt > 1e-6:
                inst = 1.0 / dt
                # EMA so the HUD number is stable.
                h.measured_fps = (0.8 * h.measured_fps + 0.2 * inst) if h.measured_fps else inst
        self._last_read_monotonic = now
        h.last_frame_monotonic = now

    # ---- Context manager sugar -----------------------------------------

    def __enter__(self) -> "FrameSource":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
