"""Qt threading layer around :class:`PipelineRunner`.

Threading model
---------------
* One **capture thread per camera** (``_CaptureThread``) continuously reads its
  source into a mutex-guarded "latest frame" slot. A camera that produces faster
  than we consume simply overwrites its slot (drop-to-latest — we never grow a
  queue and never block a fast camera on a slow one). Overwrites are counted.
* One **pipeline thread** (``PipelineWorker``) wakes on a timer, grabs the
  current triplet, runs ``ThermalPipeline.process`` via the runner, and emits the
  result to the UI thread through a Qt signal. Under load it processes the newest
  available frames and lets intermediate ones drop (adaptive frame-skip).
* The **UI thread** only renders.

CPU optimization
----------------
``configure_cpu`` caps OpenCV and PyTorch thread pools so three detectors plus
the UI don't oversubscribe the Pi 5's 4 cores and thrash. Call it once at start.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Sequence

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from thermal_algorithms.core.types import Frame

from apps.live_monitor.capture.base import FrameSource
from apps.live_monitor.runner import PipelineRunner, RunnerResult


def configure_cpu(n_threads: Optional[int] = None) -> dict:
    """Tune library thread pools for the host. Returns what was applied.

    On a 4-core Pi 5 we leave a core for capture + UI and give the detectors the
    rest. Pass ``n_threads`` to override.
    """
    import os

    cores = os.cpu_count() or 4
    threads = n_threads if n_threads else max(1, cores - 1)
    applied: dict = {"cores": cores, "threads": threads}
    try:
        import cv2
        cv2.setNumThreads(threads)
        applied["cv2"] = threads
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(threads)
        applied["torch"] = threads
    except Exception:
        pass
    return applied


class _CaptureThread(threading.Thread):
    """Continuously reads one source into a latest-frame slot (drop-to-latest)."""

    def __init__(self, source: FrameSource, poll_sleep: float = 0.001) -> None:
        super().__init__(daemon=True, name=f"capture-cam{source.camera_id}")
        self._source = source
        self._poll_sleep = poll_sleep
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._fresh = False
        self._overwrites = 0
        self._stop_evt = threading.Event()

    @property
    def source(self) -> FrameSource:
        return self._source

    @property
    def overwrites(self) -> int:
        return self._overwrites

    def run(self) -> None:
        self._source.start()
        while not self._stop_evt.is_set():
            frame = self._source.read()
            if frame is not None:
                with self._lock:
                    if self._fresh:
                        self._overwrites += 1  # previous frame never consumed
                    self._latest = frame
                    self._fresh = True
            else:
                time.sleep(self._poll_sleep)

    def take_latest(self) -> tuple[Optional[Frame], bool]:
        """Return ``(frame, was_fresh)`` and clear the fresh flag."""
        with self._lock:
            frame, fresh = self._latest, self._fresh
            self._fresh = False
            return frame, fresh

    def stop(self) -> None:
        self._stop_evt.set()


class PipelineWorker(QObject):
    """Drives the runner on its own QThread and emits results to the UI."""

    resultReady = pyqtSignal(object)        # RunnerResult
    statsReady = pyqtSignal(dict)           # periodic health/timing snapshot
    error = pyqtSignal(str)

    def __init__(
        self,
        runner: PipelineRunner,
        sources: Sequence[FrameSource],
        *,
        target_fps: float = 8.0,
        stats_interval_s: float = 0.5,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._sources = list(sources)
        self._target_dt = 1.0 / max(1.0, target_fps)
        self._stats_interval = stats_interval_s
        self._capture_threads: list[_CaptureThread] = []
        self._timer = None
        self._paused = False
        self._step_once = False
        self._last_stats = 0.0
        self._proc_fps = 0.0
        self._last_proc_t: Optional[float] = None

    @property
    def runner(self) -> PipelineRunner:
        return self._runner

    # ---- lifecycle (called in the worker thread via started signal) -----

    @pyqtSlot()
    def start(self) -> None:
        from PyQt6.QtCore import QTimer

        try:
            self._capture_threads = [_CaptureThread(s) for s in self._sources]
            for t in self._capture_threads:
                t.start()
            self._timer = QTimer()
            self._timer.setInterval(int(self._target_dt * 1000))
            self._timer.timeout.connect(self._tick)
            self._timer.start()
        except Exception as exc:
            self.error.emit(f"worker start failed: {type(exc).__name__}: {exc}")

    @pyqtSlot()
    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        for t in self._capture_threads:
            t.stop()
        for t in self._capture_threads:
            t.join(timeout=1.0)
        self._runner.stop()

    # ---- controls (thread-safe enough: simple flag writes) --------------

    @pyqtSlot(bool)
    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    @pyqtSlot()
    def request_step(self) -> None:
        self._step_once = True

    # ---- main loop ------------------------------------------------------

    def _tick(self) -> None:
        now = time.monotonic()
        try:
            if self._paused and not self._step_once:
                self._maybe_emit_stats(now)
                return
            self._step_once = False

            # Grab the newest frame from each camera. Reuse the last frame for a
            # camera that hasn't produced a new one (keeps the triplet flowing).
            triplet, any_fresh = self._collect_triplet()
            if triplet is None or not any_fresh:
                self._maybe_emit_stats(now)
                return

            rr = self._runner.process_triplet(triplet)
            self._update_proc_fps(now)
            self.resultReady.emit(rr)
            self._maybe_emit_stats(now, rr)
        except Exception as exc:
            self.error.emit(f"pipeline error: {type(exc).__name__}: {exc}")

    def _collect_triplet(self) -> tuple[Optional[tuple[Frame, Frame, Frame]], bool]:
        frames: list[Optional[Frame]] = [None, None, None]
        any_fresh = False
        for t in self._capture_threads:
            f, fresh = t.take_latest()
            frames[t.source.camera_id] = f
            any_fresh = any_fresh or fresh
        if any(f is None for f in frames):
            return None, False
        return (frames[0], frames[1], frames[2]), any_fresh  # type: ignore[return-value]

    def _update_proc_fps(self, now: float) -> None:
        if self._last_proc_t is not None:
            dt = now - self._last_proc_t
            if dt > 1e-6:
                inst = 1.0 / dt
                self._proc_fps = 0.8 * self._proc_fps + 0.2 * inst if self._proc_fps else inst
        self._last_proc_t = now

    def _maybe_emit_stats(self, now: float, rr: Optional[RunnerResult] = None) -> None:
        if now - self._last_stats < self._stats_interval:
            return
        self._last_stats = now
        cams = []
        for t in self._capture_threads:
            h = t.source.health
            cams.append({
                "camera_id": h.camera_id,
                "connected": h.connected,
                "fps": round(h.measured_fps, 1),
                "frames": h.frames_read,
                "age_s": round(h.age_s(), 2) if h.age_s() is not None else None,
                "overwrites": t.overwrites,
                "error": h.last_error,
                "extra": dict(h.extra),
            })
        self.statsReady.emit({
            "proc_fps": round(self._proc_fps, 1),
            "frames_processed": self._runner.frames_processed,
            "last_pipeline_ms": round(rr.timings_ms.get("pipeline", 0.0), 1) if rr else None,
            "cameras": cams,
        })


def make_worker_thread(worker: PipelineWorker) -> QThread:
    """Move ``worker`` onto a fresh QThread and wire start/stop. Returns the
    thread (not yet started)."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.start)
    return thread
