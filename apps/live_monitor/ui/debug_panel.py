"""Debug dock: live stats HUD, threshold sliders, and playback controls.

Emits signals the main window forwards to the worker/runner. Keeping the panel
signal-only (no direct references to the worker) keeps the UI decoupled and the
panel trivially testable.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class _Slider(QWidget):
    """A labelled float slider backed by an int Qt slider."""

    changed = pyqtSignal(float)

    def __init__(self, name: str, lo: float, hi: float, value: float, step: float = 0.01):
        super().__init__()
        self._name = name
        self._lo, self._hi, self._step = lo, hi, step
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(int(round((hi - lo) / step)))
        self._slider.setValue(int(round((value - lo) / step)))
        self._label = QLabel()
        self._label.setMinimumWidth(120)
        self._label.setFont(QFont("monospace", 8))
        self._update_label()
        self._slider.valueChanged.connect(self._on_change)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self._label)
        lay.addWidget(self._slider)

    def value(self) -> float:
        return self._lo + self._slider.value() * self._step

    def _update_label(self) -> None:
        self._label.setText(f"{self._name} = {self.value():.2f}")

    def _on_change(self) -> None:
        self._update_label()
        self.changed.emit(self.value())


class DebugPanel(QWidget):
    pauseToggled = pyqtSignal(bool)
    stepRequested = pyqtSignal()
    snapshotRequested = pyqtSignal()
    fixedScaleToggled = pyqtSignal(bool)
    thresholdsChanged = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        # --- stats HUD ---
        stats_box = QGroupBox("Telemetry")
        sgrid = QGridLayout(stats_box)
        self._lbl_proc = QLabel("–")
        self._lbl_cams = QLabel("–")
        self._lbl_cams.setWordWrap(True)
        for lbl in (self._lbl_proc, self._lbl_cams):
            lbl.setFont(QFont("monospace", 8))
        sgrid.addWidget(QLabel("Pipeline:"), 0, 0)
        sgrid.addWidget(self._lbl_proc, 0, 1)
        sgrid.addWidget(QLabel("Cameras:"), 1, 0)
        sgrid.addWidget(self._lbl_cams, 1, 1)
        root.addWidget(stats_box)

        # --- playback controls ---
        ctrl_box = QGroupBox("Controls")
        cl = QHBoxLayout(ctrl_box)
        self._pause = QCheckBox("Pause")
        self._pause.toggled.connect(self.pauseToggled.emit)
        step = QPushButton("Step")
        step.clicked.connect(self.stepRequested.emit)
        snap = QPushButton("Snapshot")
        snap.clicked.connect(self.snapshotRequested.emit)
        fixed = QCheckBox("Fixed temp scale")
        fixed.toggled.connect(self.fixedScaleToggled.emit)
        for w in (self._pause, step, snap, fixed):
            cl.addWidget(w)
        root.addWidget(ctrl_box)

        # --- threshold sliders ---
        thr_box = QGroupBox("Live thresholds")
        tl = QVBoxLayout(thr_box)
        self._sliders = {
            "t_ign": _Slider("fire t_ign °C", 20.0, 120.0, 45.0, 1.0),
            "t_fire": _Slider("fire t_fire °C", 30.0, 200.0, 60.0, 1.0),
            "score_threshold": _Slider("person score", 0.0, 1.0, 0.5, 0.01),
            "delta_m": _Slider("touch δ (m)", 0.1, 2.0, 0.5, 0.01),
            "epsilon_m": _Slider("fuse ε (m)", 0.05, 1.0, 0.5, 0.01),
        }
        for s in self._sliders.values():
            s.changed.connect(lambda _v: self._emit_thresholds())
            tl.addWidget(s)
        root.addWidget(thr_box)
        root.addStretch(1)

    # ---- updates from the worker ---------------------------------------

    def update_stats(self, stats: dict) -> None:
        proc = stats.get("proc_fps")
        ms = stats.get("last_pipeline_ms")
        n = stats.get("frames_processed")
        self._lbl_proc.setText(f"{proc} fps   {ms} ms   #{n}")
        lines = []
        for c in stats.get("cameras", []):
            state = "OK" if c["connected"] else "DOWN"
            extra = ""
            if c.get("overwrites"):
                extra = f" drop={c['overwrites']}"
            err = f"  !{c['error']}" if c.get("error") else ""
            lines.append(f"cam{c['camera_id']}: {state} {c['fps']}fps age={c['age_s']}{extra}{err}")
        self._lbl_cams.setText("\n".join(lines) if lines else "–")

    def _emit_thresholds(self) -> None:
        self.thresholdsChanged.emit({k: s.value() for k, s in self._sliders.items()})

    def current_thresholds(self) -> dict:
        return {k: s.value() for k, s in self._sliders.items()}
