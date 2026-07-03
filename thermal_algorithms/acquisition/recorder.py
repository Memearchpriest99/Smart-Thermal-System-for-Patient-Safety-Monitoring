"""Session recorder — writes MI48 streams in the repo's raw capture layout.

Produces exactly the on-disk format the rest of the stack already consumes
(see ``scripts/reorganize_waveshare.py``)::

    <root>/<scene>/<YYYYMMDD_HHMMSS>/
        ch0_thermal.npz     (key 'frames', shape (N, 62, 80) float32 °C)
        ch0/frame_00000.png ...   (320x240 colormapped previews, optional)
        ch1_thermal.npz / ch1/ ...
        ch2_thermal.npz / ch2/ ...

Unlike the previous recorder, channels are read in lockstep — one frame per
channel per tick — so every ``ch{N}_thermal.npz`` has the SAME length and the
per-channel misalignment that ``reorganize_waveshare.py`` had to repair at
the tail can no longer occur.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from thermal_algorithms.core.types import Frame

PNG_PREVIEW_SIZE = (320, 240)  # (W, H), matches the existing dataset previews


class SessionRecorder:
    """Record synchronized multi-camera sessions to the raw capture layout.

    Args:
        cameras: Opened camera objects exposing ``read_frame() -> Frame``
            and ``config.camera_id`` (e.g. `MI48Camera`). Channel numbers
            in the output follow each camera's ``camera_id``.
        root: Dataset root, e.g. ``datasets/waveshare``.
        scene: Scene name, e.g. ``2ppl_walk`` — becomes the folder name.
        write_pngs: Also render 320x240 colormapped preview PNGs per frame
            (requires OpenCV; silently skipped if cv2 is unavailable).
        session: Session folder name; defaults to the current local time
            formatted ``YYYYMMDD_HHMMSS``.
    """

    def __init__(
        self,
        cameras: Sequence[object],
        root: str | Path,
        scene: str,
        *,
        write_pngs: bool = True,
        session: Optional[str] = None,
    ) -> None:
        if not cameras:
            raise ValueError("At least one camera is required.")
        self.cameras = list(cameras)
        self.root = Path(root)
        self.scene = scene
        self.session = session or time.strftime("%Y%m%d_%H%M%S")
        self.write_pngs = write_pngs
        self._buffers: dict[int, list[np.ndarray]] = {
            cam.config.camera_id: [] for cam in self.cameras
        }

    @property
    def session_dir(self) -> Path:
        return self.root / self.scene / self.session

    @property
    def n_frames(self) -> int:
        """Frames captured so far (identical across channels by design)."""
        return min(len(b) for b in self._buffers.values())

    # -- capture --------------------------------------------------------

    def capture_tick(self) -> dict[int, Frame]:
        """Read one frame from every camera (lockstep) and buffer them."""
        frames: dict[int, Frame] = {}
        for cam in self.cameras:
            frame = cam.read_frame()
            self._buffers[cam.config.camera_id].append(frame.data)
            frames[cam.config.camera_id] = frame
        return frames

    def record(
        self,
        *,
        duration_s: Optional[float] = None,
        n_frames: Optional[int] = None,
        on_tick: Optional[Callable[[int, dict[int, Frame]], None]] = None,
    ) -> Path:
        """Capture until the duration or frame budget is reached, then save.

        A KeyboardInterrupt mid-capture still saves what was recorded.

        Returns:
            The session directory that was written.
        """
        if (duration_s is None) == (n_frames is None):
            raise ValueError("Specify exactly one of duration_s / n_frames.")

        deadline = None if duration_s is None else time.monotonic() + duration_s
        try:
            i = 0
            while True:
                if n_frames is not None and i >= n_frames:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                frames = self.capture_tick()
                if on_tick is not None:
                    on_tick(i, frames)
                i += 1
        except KeyboardInterrupt:
            pass
        return self.save()

    # -- persistence ------------------------------------------------------

    def save(self) -> Path:
        """Write buffered frames to ``<root>/<scene>/<session>/``."""
        out = self.session_dir
        out.mkdir(parents=True, exist_ok=True)
        n = self.n_frames  # lockstep ⇒ equal, but truncate defensively
        for ch, frames in self._buffers.items():
            stack = (
                np.stack(frames[:n]).astype(np.float32)
                if n
                else np.empty((0, 0, 0), dtype=np.float32)
            )
            np.savez_compressed(out / f"ch{ch}_thermal.npz", frames=stack)
            if self.write_pngs and n:
                self._write_previews(out / f"ch{ch}", stack)
        return out

    @staticmethod
    def _write_previews(ch_dir: Path, frames: np.ndarray) -> None:
        try:
            import cv2
        except ImportError:
            return
        ch_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            img8 = cv2.normalize(
                frame, None, 0, 255, norm_type=cv2.NORM_MINMAX
            ).astype(np.uint8)
            img = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
            img = cv2.resize(img, PNG_PREVIEW_SIZE, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(ch_dir / f"frame_{i:05d}.png"), img)
