"""Calibration mode: enter room geometry + correspondences, see a 3-D room,
solve homographies, and save an annotator-compatible calibration file."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from thermal_algorithms.core.types import HomographyMatrices
from apps.live_monitor.calibration import (
    CalibrationInput,
    rectangle_world_corners,
    save_calibration,
    solve_homographies,
)
from apps.live_monitor.ui.room3d import HAS_GL, ROOM_MAX_M, CameraPose, populate_gl_view

_FOV_PRESETS = {"Standard (45°)": (45.0, 34.0), "Wide (90°)": (90.0, 65.0)}


def _spin(lo: float, hi: float, val: float, step: float = 0.1, suffix: str = " m") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    s.setDecimals(2)
    s.setSuffix(suffix)
    return s


class _CameraTab(QWidget):
    def __init__(self, camera_id: int, on_change) -> None:
        super().__init__()
        self.camera_id = camera_id
        lay = QVBoxLayout(self)

        pose = QGroupBox("Mounting pose (room coords)")
        form = QFormLayout(pose)
        self.x = _spin(0.0, ROOM_MAX_M, min(ROOM_MAX_M, [0.2, 2.5, 4.8][camera_id]))
        self.y = _spin(0.0, ROOM_MAX_M, min(ROOM_MAX_M, [2.5, 0.2, 2.5][camera_id]))
        self.z = _spin(0.0, ROOM_MAX_M, min(ROOM_MAX_M, 2.6))
        self.yaw = _spin(-180.0, 180.0, [0.0, 90.0, 180.0][camera_id], step=5.0, suffix="°")
        self.tilt = _spin(0.0, 90.0, 30.0, step=5.0, suffix="°")
        self.fov = QComboBox()
        self.fov.addItems(list(_FOV_PRESETS.keys()))
        form.addRow("x", self.x)
        form.addRow("y", self.y)
        form.addRow("z (height)", self.z)
        form.addRow("yaw", self.yaw)
        form.addRow("tilt (down)", self.tilt)
        form.addRow("lens FOV", self.fov)
        lay.addWidget(pose)
        for w in (self.x, self.y, self.z, self.yaw, self.tilt):
            w.valueChanged.connect(on_change)
        self.fov.currentIndexChanged.connect(on_change)

        corr = QGroupBox("Pixel ↔ floor correspondences  (u px, v px, X m, Y m)")
        cl = QVBoxLayout(corr)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["u (px)", "v (px)", "X (m)", "Y (m)"])
        cl.addWidget(self.table)
        btns = QHBoxLayout()
        add = QPushButton("+ row")
        rem = QPushButton("− row")
        fill = QPushButton("Fill rectangle corners")
        add.clicked.connect(lambda: self._add_row())
        rem.clicked.connect(self._remove_row)
        fill.clicked.connect(self._fill_requested)
        for b in (add, rem, fill):
            btns.addWidget(b)
        cl.addLayout(btns)
        lay.addWidget(corr)
        self._fill_callback = None

    def _add_row(self, values=(0.0, 0.0, 0.0, 0.0)) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate(values):
            self.table.setItem(r, c, QTableWidgetItem(f"{v:g}"))

    def _remove_row(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            r = self.table.rowCount() - 1
        if r >= 0:
            self.table.removeRow(r)

    def _fill_requested(self) -> None:
        if self._fill_callback:
            self._fill_callback(self)

    def fill_rectangle(self, rect_wh: tuple[float, float]) -> None:
        for (x, y) in rectangle_world_corners(rect_wh):
            self._add_row((0.0, 0.0, x, y))

    def fov_value(self) -> tuple[float, float]:
        return _FOV_PRESETS[self.fov.currentText()]

    def pose(self) -> CameraPose:
        return CameraPose(
            camera_id=self.camera_id,
            x=self.x.value(), y=self.y.value(), z=self.z.value(),
            yaw_deg=self.yaw.value(), tilt_deg=self.tilt.value(),
            fov_deg=self.fov_value(),
        )

    def correspondences(self) -> list[tuple[float, float, float, float]]:
        rows = []
        for r in range(self.table.rowCount()):
            try:
                vals = tuple(float(self.table.item(r, c).text()) for c in range(4))
            except (AttributeError, ValueError):
                continue
            rows.append(vals)  # type: ignore[arg-type]
        return rows


class CalibrationDialog(QDialog):
    def __init__(self, default_save_dir: Optional[Path] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Room & Homography Calibration")
        self.resize(1000, 640)
        self._default_save_dir = Path(default_save_dir) if default_save_dir else Path.cwd()
        self.homography: Optional[HomographyMatrices] = None
        self.result_path: Optional[Path] = None

        root = QHBoxLayout(self)

        # --- left: inputs ---
        left = QVBoxLayout()
        room = QGroupBox(f"Room (each side ≤ {ROOM_MAX_M:g} m)")
        rform = QFormLayout(room)
        self.length = _spin(0.1, ROOM_MAX_M, 4.0)
        self.width = _spin(0.1, ROOM_MAX_M, 4.0)
        self.height = _spin(0.1, ROOM_MAX_M, 2.8)
        self.rect_w = _spin(0.1, ROOM_MAX_M, 2.0)
        self.rect_h = _spin(0.1, ROOM_MAX_M, 1.0)
        rform.addRow("Length (X)", self.length)
        rform.addRow("Width (Y)", self.width)
        rform.addRow("Height (Z)", self.height)
        rform.addRow("Floor rect W", self.rect_w)
        rform.addRow("Floor rect H", self.rect_h)
        left.addWidget(room)
        for w in (self.length, self.width, self.height, self.rect_w, self.rect_h):
            w.valueChanged.connect(self._refresh_3d)

        self.tabs = QTabWidget()
        self._cam_tabs = []
        for cam in range(3):
            tab = _CameraTab(cam, self._refresh_3d)
            tab._fill_callback = lambda t: t.fill_rectangle((self.rect_w.value(), self.rect_h.value()))
            self._cam_tabs.append(tab)
            self.tabs.addTab(tab, f"Camera {cam}")
        left.addWidget(self.tabs, 1)

        self._status = QLabel("Enter ≥ 4 correspondences per camera, then Solve & Save.")
        self._status.setWordWrap(True)
        left.addWidget(self._status)

        buttons = QDialogButtonBox()
        solve_btn = buttons.addButton("Solve && Save", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        solve_btn.clicked.connect(self._solve_and_save)
        buttons.rejected.connect(self.reject)
        left.addWidget(buttons)

        root.addLayout(left, 1)

        # --- right: 3-D room ---
        self._gl_view = None
        if HAS_GL:
            import pyqtgraph.opengl as gl
            self._gl_view = gl.GLViewWidget()
            self._gl_view.setBackgroundColor("#0d0d0d")
            self._gl_view.opts["distance"] = 12
            root.addWidget(self._gl_view, 1)
        else:
            fallback = QLabel("3-D view unavailable\n(install PyOpenGL for the room model)")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("background:#0d0d0d; color:#888;")
            root.addWidget(fallback, 1)

        self._refresh_3d()

    # ---- 3-D refresh ----------------------------------------------------

    def _refresh_3d(self) -> None:
        if self._gl_view is None:
            return
        L, W, H = self.length.value(), self.width.value(), self.height.value()
        cams = [t.pose() for t in self._cam_tabs]
        try:
            populate_gl_view(self._gl_view, L, W, H,
                             (self.rect_w.value(), self.rect_h.value()), cams)
        except Exception as exc:
            self._status.setText(f"3-D refresh error: {exc}")

    # ---- solve ----------------------------------------------------------

    def build_input(self) -> CalibrationInput:
        poses = {}
        corr = {}
        for t in self._cam_tabs:
            p = t.pose()
            poses[t.camera_id] = (p.x, p.y, p.z, p.yaw_deg, p.tilt_deg, p.fov_deg[0], p.fov_deg[1])
            corr[t.camera_id] = t.correspondences()
        return CalibrationInput(
            room_lwh=(self.length.value(), self.width.value(), self.height.value()),
            rect_wh=(self.rect_w.value(), self.rect_h.value()),
            correspondences=corr,
            camera_poses=poses,
            ref_camera=0,
        )

    def _solve_and_save(self) -> None:
        inp = self.build_input()
        try:
            H, info = solve_homographies(inp)
        except Exception as exc:
            QMessageBox.warning(self, "Calibration", f"Could not solve homographies:\n{exc}")
            return

        summary = "  ".join(
            f"cam{c}:n={i['n_pairs']}"
            + (f",res={i['median_residual']:.3f}" if i["median_residual"] is not None else "")
            for c, i in sorted(info.items())
        )
        default = str(self._default_save_dir / "homography_calibration.npz")
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration", default, "NumPy (*.npz)")
        if not path:
            return
        try:
            self.result_path = save_calibration(path, inp, H)
            self.homography = H
        except Exception as exc:
            QMessageBox.warning(self, "Calibration", f"Could not save:\n{exc}")
            return
        self._status.setText(f"Saved {self.result_path.name}.  {summary}")
        self.accept()
