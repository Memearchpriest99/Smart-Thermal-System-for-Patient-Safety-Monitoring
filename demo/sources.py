"""Frame sources for the demo: live MI48 cameras or recorded-session replay.

Both sources push synchronized 3-camera ticks to a callback:
    on_tick(frames: list[np.ndarray HxW float32], tick: int, timestamp: float)
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

TickCallback = Callable[[list, int, float], None]


class ReplaySource:
    """Loops one or more recorded sessions at the requested frame rate.

    Accepts session folders in either on-disk layout:
        ch{N}_raw_data.npz   (waveshare_work)
        ch{N}_thermal.npz    (raw capture)
    """

    name = "REPLAY"

    def __init__(self, session_dirs: Sequence[str | Path], fps: float = 8.0) -> None:
        self._sessions = []
        for d in session_dirs:
            d = Path(d)
            chans = []
            for ch in range(3):
                for pattern in (f"ch{ch}_raw_data.npz", f"ch{ch}_thermal.npz"):
                    p = d / pattern
                    if p.exists():
                        chans.append(np.load(p)["frames"].astype(np.float32))
                        break
            if len(chans) != 3:
                raise FileNotFoundError(f"{d}: expected 3 channels of npz frames")
            n = min(c.shape[0] for c in chans)
            self._sessions.append((d.name, [c[:n] for c in chans], n))
        if not self._sessions:
            raise ValueError("No replay sessions given.")
        self.fps = fps
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.current_session = self._sessions[0][0]

    def start(self, on_tick: TickCallback) -> None:
        def run() -> None:
            tick = 0
            period = 1.0 / self.fps
            next_t = time.monotonic()
            while not self._stop.is_set():
                for name, chans, n in self._sessions:
                    self.current_session = name
                    for i in range(n):
                        if self._stop.is_set():
                            return
                        on_tick([chans[c][i] for c in range(3)], tick, time.time())
                        tick += 1
                        next_t += period
                        delay = next_t - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                        else:
                            next_t = time.monotonic()

        self._thread = threading.Thread(target=run, name="replay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


class LiveMI48Source:
    """Three Waveshare MI48 cameras (wiki SPI/I2C pipeline), one thread each.

    Each camera thread blocks on its DATA_READY pin at the configured FPS and
    stores its latest frame; a ticker thread assembles synchronized triplets.
    The small inter-camera phase offset is inherent to the hardware and is
    tolerated by all downstream detectors.
    """

    name = "LIVE"

    def __init__(self, configs, fps: float = 8.0) -> None:
        from thermal_algorithms.acquisition import MI48Camera

        self.fps = fps
        self._cams = [MI48Camera(cfg) for cfg in configs]
        if len(self._cams) != 3:
            raise ValueError("Live demo needs exactly 3 camera configs.")
        self._latest: list[Optional[np.ndarray]] = [None, None, None]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.current_session = "live"

    def _cam_loop(self, idx: int) -> None:
        cam = self._cams[idx]
        cam.open()
        cam.start()
        try:
            while not self._stop.is_set():
                frame = cam.read_frame()
                with self._lock:
                    self._latest[idx] = frame.data
        finally:
            cam.close()

    def start(self, on_tick: TickCallback) -> None:
        for i in range(3):
            t = threading.Thread(target=self._cam_loop, args=(i,),
                                 name=f"mi48-{i}", daemon=True)
            t.start()
            self._threads.append(t)

        def ticker() -> None:
            tick = 0
            period = 1.0 / self.fps
            next_t = time.monotonic()
            while not self._stop.is_set():
                with self._lock:
                    frames = list(self._latest)
                if all(f is not None for f in frames):
                    on_tick(frames, tick, time.time())
                    tick += 1
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()

        t = threading.Thread(target=ticker, name="ticker", daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
