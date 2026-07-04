"""Multithreaded detection engine.

Thread layout (all inference off the UI thread):
    camera-0..2   one worker per camera: rolling-p25 background, Tateno
                  residual, MobileNet-SSD (person), FireSVM (fire)
    contact       assembles per-tick residual triplets and runs Thermo-X3D

Real-time discipline: every queue has size 1 and newer frames replace older
ones, so the demo never falls behind the sensor — it drops instead.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from demo.detectors import BackgroundModel, FireDetector, OnnxContact, OnnxSSD


def _put_latest(q: "queue.Queue", item) -> None:
    """Replace the queue's content with the newest item (never blocks)."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


class DetectionEngine:
    def __init__(self, checkpoint_dir: str | Path, *,
                 ssd_threads: int = 1, x3d_threads: int = 4) -> None:
        from thermal_algorithms.core.sensor_profile import WAVESHARE_26984

        ckpt = Path(checkpoint_dir)
        ssd_dir = ckpt / "mobilenet_ssd_detector"
        x3d_dir = ckpt / "thermo_x3d_detector"

        self._profile = WAVESHARE_26984
        self._ssd_args = (ssd_dir / "Waveshare_26984_raw.onnx",
                          ssd_dir / "Waveshare_26984_raw.anchors.npy",
                          ssd_dir / "Waveshare_26984_raw.meta.json")
        self._ssd_threads = ssd_threads
        self._fire_path = ckpt / "fire_svm_detector" / "_default.thalg"
        self._contact = OnnxContact(x3d_dir / "Waveshare_26984_T5_v2.ftz.onnx",
                                    x3d_dir / "Waveshare_26984_T5_v2.meta.json",
                                    threads=x3d_threads)

        self._cam_queues = [queue.Queue(maxsize=1) for _ in range(3)]
        self._contact_queue: "queue.Queue" = queue.Queue(maxsize=1)
        self._pending: dict[int, dict[int, np.ndarray]] = {}
        self._pending_lock = threading.Lock()

        self._state_lock = threading.Lock()
        self._state = {
            "cams": [
                {"raw": None, "boxes": [], "fire": False, "fire_conf": 0.0,
                 "fire_bbox": None, "bg_ready": False, "proc_ms": 0.0}
                for _ in range(3)
            ],
            "contact": {"conf": None, "alarmed": False, "history": deque(maxlen=240),
                        "proc_ms": 0.0, "status": "warming up"},
            "tick": 0,
            "tick_fps": 0.0,
        }
        self._last_tick_t: Optional[float] = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ---- source callback -------------------------------------------------

    def submit(self, frames: list, tick: int, ts: float) -> None:
        with self._state_lock:
            self._state["tick"] = tick
            if self._last_tick_t is not None:
                dt = ts - self._last_tick_t
                if dt > 0:
                    ema = self._state["tick_fps"]
                    self._state["tick_fps"] = (0.9 * ema + 0.1 / dt) if ema else 1 / dt
            self._last_tick_t = ts
        for cam in range(3):
            _put_latest(self._cam_queues[cam], (frames[cam], tick))

    # ---- workers ------------------------------------------------------------

    def _camera_worker(self, cam: int) -> None:
        ssd = OnnxSSD(*self._ssd_args, threads=self._ssd_threads)
        fire = FireDetector(self._fire_path)
        bg = BackgroundModel(self._profile)
        while not self._stop.is_set():
            try:
                raw, tick = self._cam_queues[cam].get(timeout=0.25)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            bg.add(raw)
            resid = bg.residual(raw)
            boxes = ssd.predict(raw)
            is_fire, fire_conf, fire_bbox = fire.predict(raw)
            ms = (time.perf_counter() - t0) * 1e3

            with self._state_lock:
                c = self._state["cams"][cam]
                c.update(raw=raw, boxes=boxes, fire=is_fire, fire_conf=fire_conf,
                         fire_bbox=fire_bbox, bg_ready=bg.ready)
                c["proc_ms"] = 0.8 * c["proc_ms"] + 0.2 * ms if c["proc_ms"] else ms

            if resid is not None:
                triplet = None
                with self._pending_lock:
                    slot = self._pending.setdefault(tick, {})
                    slot[cam] = resid
                    if len(slot) == 3:
                        triplet = tuple(slot[i] for i in range(3))
                        del self._pending[tick]
                    for old in [t for t in self._pending if t < tick - 8]:
                        del self._pending[old]
                if triplet is not None:
                    _put_latest(self._contact_queue, (tick, triplet))

    def _contact_worker(self) -> None:
        while not self._stop.is_set():
            try:
                tick, triplet = self._contact_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            conf, alarmed = self._contact.update(triplet)
            ms = (time.perf_counter() - t0) * 1e3
            with self._state_lock:
                s = self._state["contact"]
                s["conf"] = conf
                s["alarmed"] = alarmed
                s["proc_ms"] = 0.8 * s["proc_ms"] + 0.2 * ms if s["proc_ms"] else ms
                if conf is not None:
                    s["history"].append(conf)
                    s["status"] = "monitoring"
                else:
                    s["status"] = "buffering"

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        for cam in range(3):
            t = threading.Thread(target=self._camera_worker, args=(cam,),
                                 name=f"cam-worker-{cam}", daemon=True)
            t.start()
            self._threads.append(t)
        t = threading.Thread(target=self._contact_worker, name="contact-worker",
                             daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)

    def snapshot(self) -> dict:
        with self._state_lock:
            cams = [dict(c) for c in self._state["cams"]]
            contact = dict(self._state["contact"])
            contact["history"] = list(contact["history"])
            return {"cams": cams, "contact": contact,
                    "tick": self._state["tick"],
                    "tick_fps": self._state["tick_fps"]}
