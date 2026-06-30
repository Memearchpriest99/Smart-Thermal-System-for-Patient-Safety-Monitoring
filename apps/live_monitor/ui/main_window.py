"""Main application window — wires the worker, feeds, selectors, banner,
bird's-eye dock, calibration, and the flight recorder."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QMetaObject, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from thermal_algorithms.core.types import Detection
from apps.live_monitor import rendering
from apps.live_monitor.calibration import load_calibration
from apps.live_monitor.detectors import DetectorUnavailable, TASKS
from apps.live_monitor.runner import PipelineRunner, RunnerResult
from apps.live_monitor.pipeline_worker import PipelineWorker
from apps.live_monitor.ui.birdseye_widget import BirdsEyeWidget
from apps.live_monitor.ui.calibration_dialog import CalibrationDialog
from apps.live_monitor.ui.camera_view import CameraView
from apps.live_monitor.ui.debug_panel import DebugPanel

_TASK_LABELS = {"fire": "Fire", "person": "Person", "touch": "Touch"}
_FIXED_SCALE = (15.0, 45.0)  # °C band when "fixed temp scale" is on


class MainWindow(QMainWindow):
    def __init__(
        self,
        runner: PipelineRunner,
        worker: PipelineWorker,
        *,
        default_save_dir: Optional[Path] = None,
        recordings_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Live Thermal Monitor — Patient Safety")
        self.runner = runner
        self.worker = worker
        self._default_save_dir = Path(default_save_dir) if default_save_dir else Path.cwd()
        self._recordings_dir = Path(recordings_dir) if recordings_dir else (self._default_save_dir / "recordings")
        self._fixed_scale = False
        self._last_alert_dump = 0.0
        # flight recorder: ~last 10 s of triplets at 8 Hz
        self._ring: deque = deque(maxlen=80)

        self._build_ui()
        self._connect_worker()
        self._refresh_birdseye_visibility()

    # ---- UI construction ------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        # selector bars
        selectors = QHBoxLayout()
        self._combos: dict[str, QComboBox] = {}
        for task in TASKS:
            selectors.addWidget(QLabel(f"{_TASK_LABELS[task]} algo:"))
            combo = self._make_combo(task)
            self._combos[task] = combo
            selectors.addWidget(combo, 1)
        root.addLayout(selectors)

        # camera row
        cam_row = QHBoxLayout()
        self._views = [CameraView(i) for i in range(3)]
        for v in self._views:
            cam_row.addWidget(v, 1)
        root.addLayout(cam_row, 1)

        # touch alert banner
        self._banner = QLabel("Monitoring…")
        self._banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._banner.setFont(QFont("sans", 16, QFont.Weight.Bold))
        self._banner.setFixedHeight(54)
        self._set_banner(active=False, text="No contact detected")
        root.addWidget(self._banner)

        self.setCentralWidget(central)

        # bird's-eye dock (right)
        self._birdseye = BirdsEyeWidget()
        self._be_dock = QDockWidget("Bird's-eye (geometric touch)", self)
        self._be_dock.setWidget(self._birdseye)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._be_dock)

        # debug dock (left)
        self._debug = DebugPanel()
        dbg_dock = QDockWidget("Debug", self)
        dbg_dock.setWidget(self._debug)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dbg_dock)
        self._debug.pauseToggled.connect(self._on_pause)
        self._debug.stepRequested.connect(self._on_step)
        self._debug.snapshotRequested.connect(lambda: self._dump_flight_recorder("manual"))
        self._debug.fixedScaleToggled.connect(self._on_fixed_scale)
        self._debug.thresholdsChanged.connect(self.runner.set_thresholds)

        self._build_menu()
        # seed initial bird's-eye geometry from the runner's homography
        self._apply_homography_to_birdseye()

    def _make_combo(self, task: str) -> QComboBox:
        combo = QComboBox()
        ctx = self.runner.build_context()
        current = self.runner.selection.get(task)
        for idx, opt in enumerate(self.runner.options[task]):
            available, reason = opt.availability(ctx)
            text = opt.display if available else f"{opt.display}  —  [{reason}]"
            combo.addItem(text, userData=opt.key)
            if not available:
                # grey out the unavailable item
                model = combo.model()
                model.item(idx).setEnabled(False)
            if opt.key == current:
                combo.setCurrentIndex(idx)
        combo.activated.connect(lambda _i, t=task: self._on_select(t))
        return combo

    def _build_menu(self) -> None:
        bar = self.menuBar()
        cal_menu = bar.addMenu("&Calibration")
        act_cal = QAction("Calibrate room…", self)
        act_cal.triggered.connect(self._open_calibration)
        act_load = QAction("Load calibration…", self)
        act_load.triggered.connect(self._load_calibration)
        cal_menu.addAction(act_cal)
        cal_menu.addAction(act_load)

        mode_menu = bar.addMenu("&Mode")
        self._act_restricted = QAction("Restricted-area mode", self, checkable=True)
        self._act_restricted.toggled.connect(self._on_restricted)
        mode_menu.addAction(self._act_restricted)

    # ---- worker wiring --------------------------------------------------

    def _connect_worker(self) -> None:
        self.worker.resultReady.connect(self._on_result)
        self.worker.statsReady.connect(self._debug.update_stats)
        self.worker.error.connect(self._on_error)

    # ---- selection / mode ----------------------------------------------

    def _on_select(self, task: str) -> None:
        combo = self._combos[task]
        key = combo.currentData()
        prev = self.runner.selection.get(task)
        if key == prev:
            return
        try:
            self.runner.select(task, key)
        except DetectorUnavailable as exc:
            QMessageBox.information(self, "Algorithm unavailable", str(exc))
            self._reselect_previous(combo, prev)
            return
        if task == "touch":
            self._refresh_birdseye_visibility()

    def _reselect_previous(self, combo: QComboBox, prev_key: Optional[str]) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == prev_key:
                combo.setCurrentIndex(i)
                return

    def _on_restricted(self, checked: bool) -> None:
        self.runner.set_restricted(checked)

    def _on_pause(self, paused: bool) -> None:
        self.worker.set_paused(paused)

    def _on_step(self) -> None:
        self.worker.request_step()

    def _on_fixed_scale(self, fixed: bool) -> None:
        self._fixed_scale = fixed

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 5000)

    # ---- results --------------------------------------------------------

    def _on_result(self, rr: RunnerResult) -> None:
        res = rr.result
        self._ring.append(rr)

        vmin, vmax = (_FIXED_SCALE if self._fixed_scale else (None, None))
        for cam in range(3):
            frame = rr.frames[cam]
            rgb = rendering.colorize(frame.data, vmin=vmin, vmax=vmax)
            person = list(res.detections[cam]) if not self.runner.pipeline.restricted else []
            # In restricted mode the human detections live in res.detections too.
            if self.runner.pipeline.restricted:
                person = list(res.detections[cam])
            fire = self._fire_boxes(res, cam)
            self._views[cam].set_frame(
                rgb, temp=frame.data, person=person, fire=fire,
                status=f"{frame.shape[1]}x{frame.shape[0]}  t={frame.timestamp:.2f}s",
            )

        # touch banner (primary), with fire/restricted reflected in text
        self._update_banner(res)

        # bird's-eye for geometric touch
        if self.runner.is_geometric_touch():
            self._birdseye.update_frame(res.detections, res.contact_event)

        # auto-dump a clip on any alarm (debounced)
        if res.any_alarm:
            now = time.monotonic()
            if now - self._last_alert_dump > 5.0:
                self._last_alert_dump = now
                self._dump_flight_recorder("alert")

    @staticmethod
    def _fire_boxes(res, cam: int) -> list[Detection]:
        fa = res.fire_alerts[cam]
        bbox = fa.blob_features.get("bbox") if fa.is_alarm else None
        if bbox is None:
            return []
        return [Detection(bbox=tuple(float(x) for x in bbox), score=float(fa.confidence),
                          class_id=0, camera_id=cam)]

    def _update_banner(self, res) -> None:
        if res.contact_alarm:
            self._set_banner(True, "⚠  TOUCH / CONTACT DETECTED")
        elif res.restricted_area_alarm:
            self._set_banner(True, "⚠  UNAUTHORIZED PRESENCE (restricted area)")
        elif res.fire_alarm:
            self._set_banner(True, "🔥  FIRE DETECTED", color="#b8860b")
        else:
            self._set_banner(False, "No contact detected")

    def _set_banner(self, active: bool, text: str, color: str = "#c0392b") -> None:
        if active:
            self._banner.setStyleSheet(f"background:{color}; color:white;")
        else:
            self._banner.setStyleSheet("background:#1e3d2f; color:#9fdfb5;")
        self._banner.setText(text)

    # ---- bird's-eye + calibration --------------------------------------

    def _refresh_birdseye_visibility(self) -> None:
        self._be_dock.setVisible(self.runner.is_geometric_touch())

    def _apply_homography_to_birdseye(self, corners=None, unit: str = "px") -> None:
        H = self.runner.homography
        self._birdseye.set_geometry(H, corners=corners, unit=unit,
                                    delta_m=self._debug.current_thresholds().get("delta_m", 0.5))

    def _open_calibration(self) -> None:
        dlg = CalibrationDialog(default_save_dir=self._default_save_dir, parent=self)
        if dlg.exec() and dlg.homography is not None:
            self.runner.set_homography(dlg.homography)
            corners = None
            try:
                _H, extras = load_calibration(dlg.result_path)
                corners = extras.get("world_positions")
            except Exception:
                pass
            self._apply_homography_to_birdseye(corners=corners, unit="m")
            self._refresh_birdseye_visibility()
            self.statusBar().showMessage(f"Calibration loaded from {dlg.result_path}", 4000)

    def _load_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load calibration", str(self._default_save_dir), "NumPy (*.npz)")
        if not path:
            return
        try:
            H, extras = load_calibration(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load calibration", f"Failed:\n{exc}")
            return
        self.runner.set_homography(H)
        self._apply_homography_to_birdseye(corners=extras.get("world_positions"), unit="m")
        self._refresh_birdseye_visibility()
        self.statusBar().showMessage(f"Calibration loaded from {path}", 4000)

    # ---- flight recorder ------------------------------------------------

    def _dump_flight_recorder(self, reason: str) -> None:
        if not self._ring:
            self.statusBar().showMessage("Nothing recorded yet.", 2000)
            return
        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = self._recordings_dir / f"clip_{stamp}_{reason}"
        per_cam = [[], [], []]
        events = []
        for rr in list(self._ring):
            for cam in range(3):
                per_cam[cam].append(rr.frames[cam].data)
            events.append({
                "t": rr.frames[2].timestamp,
                "alerts": [a.type.value for a in rr.result.alerts],
            })
        for cam in range(3):
            np.savez_compressed(f"{base}_ch{cam}.npz", frames=np.asarray(per_cam[cam], dtype=np.float32))
        (Path(f"{base}_events.json")).write_text(json.dumps({
            "reason": reason, "n_frames": len(self._ring),
            "selection": self.runner.selection, "events": events,
        }, indent=2))
        self.statusBar().showMessage(f"Saved flight-recorder clip → {base.name}_*.npz", 5000)

    # ---- shutdown -------------------------------------------------------

    def closeEvent(self, event) -> None:
        thread = self.worker.thread()
        running = thread is not None and thread.isRunning()
        try:
            if running:
                # Stop in the worker thread; blocking is safe only when its
                # event loop is live (otherwise this would deadlock).
                QMetaObject.invokeMethod(self.worker, "stop", Qt.ConnectionType.BlockingQueuedConnection)
            else:
                self.worker.stop()
        except Exception:
            pass
        if running:
            thread.quit()
            thread.wait(2000)
        super().closeEvent(event)
