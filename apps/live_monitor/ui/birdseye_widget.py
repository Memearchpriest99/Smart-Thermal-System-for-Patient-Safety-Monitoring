"""Live bird's-eye floor-plane view for the geometric touch detector.

Draws, in real time, what ``scripts/visualize_birdseye.py`` renders to a GIF:
the calibrated floor rectangle, each camera's projected foot-points, the fused
actor positions (from the ``ContactEvent``), a rolling trail, and — when a pair
is flagged in contact — a red link annotated with the floor distance.

Reuses ``multi_view.homography.project_foot_point`` for the per-camera markers.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from thermal_algorithms.core.types import ContactEvent, Detection, HomographyMatrices
from thermal_algorithms.contact_detection.multi_view.homography import project_foot_point

CAM_COLORS = [QColor("#ff6b6b"), QColor("#4ecdc4"), QColor("#ffe066")]
ACTOR_COLOR = QColor("#ffd740")
TRAIL_LEN = 30


class BirdsEyeWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._H: Optional[HomographyMatrices] = None
        self._corners: Optional[np.ndarray] = None     # (4,2) world coords
        self._unit = "px"
        self._delta_m = 0.5
        self._cam_points: list[list[tuple[float, float]]] = [[], [], []]
        self._actors: list[tuple[float, float]] = []
        self._pairs: list[tuple[int, int]] = []
        self._trail: list[tuple[float, float]] = []
        self.setMinimumSize(280, 280)
        self.setStyleSheet("background:#111;")

    # ---- configuration --------------------------------------------------

    def set_geometry(
        self,
        homography: Optional[HomographyMatrices],
        *,
        corners: Optional[np.ndarray] = None,
        unit: str = "px",
        delta_m: float = 0.5,
    ) -> None:
        self._H = homography
        self._corners = None if corners is None else np.asarray(corners, dtype=float)
        self._unit = unit
        self._delta_m = delta_m
        self._trail.clear()
        self.update()

    def set_delta_m(self, delta_m: float) -> None:
        self._delta_m = delta_m
        self.update()

    # ---- per-frame update ----------------------------------------------

    def update_frame(
        self,
        detections_per_cam: Sequence[Sequence[Detection]],
        contact_event: Optional[ContactEvent],
    ) -> None:
        # Per-camera projected foot-points (for debugging the homography).
        self._cam_points = [[], [], []]
        if self._H is not None:
            for cam in range(3):
                H = self._H[cam]
                for det in detections_per_cam[cam]:
                    self._cam_points[cam].append(project_foot_point(det.foot_point, H))

        # Fused actors + contact pairs come straight from the detector output.
        self._actors = []
        self._pairs = []
        if contact_event is not None:
            self._actors = [a.world_xy for a in contact_event.actors]
            self._pairs = list(contact_event.pairs_in_contact)

        if self._actors:
            cx = float(np.mean([p[0] for p in self._actors]))
            cy = float(np.mean([p[1] for p in self._actors]))
            self._trail.append((cx, cy))
            del self._trail[:-TRAIL_LEN]
        self.update()

    # ---- view transform -------------------------------------------------

    def _window(self) -> tuple[float, float, float, float]:
        xs: list[float] = []
        ys: list[float] = []
        if self._corners is not None:
            xs += list(self._corners[:, 0])
            ys += list(self._corners[:, 1])
        for grp in self._cam_points:
            for (x, y) in grp:
                if abs(x) < 1e4 and abs(y) < 1e4:
                    xs.append(x); ys.append(y)
        for (x, y) in self._actors:
            xs.append(x); ys.append(y)
        if not xs or not ys:
            return (0.0, 1.0, 0.0, 1.0)
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mx = 0.2 * max(x1 - x0, 1e-3)
        my = 0.2 * max(y1 - y0, 1e-3)
        return (x0 - mx, x1 + mx, y0 - my, y1 + my)

    def _make_transform(self):
        x0, x1, y0, y1 = self._window()
        pad = 24.0
        w = self.width() - 2 * pad
        h = self.height() - 2 * pad
        sx = w / max(1e-6, (x1 - x0))
        sy = h / max(1e-6, (y1 - y0))
        s = min(sx, sy)  # equal aspect

        def to_widget(x: float, y: float) -> QPointF:
            px = pad + (x - x0) * s
            # invert Y for an image-like top-down view
            py = pad + (y1 - y) * s
            return QPointF(px, py)

        return to_widget, s

    # ---- painting -------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d0d0d"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._H is None:
            painter.setPen(QPen(QColor("#888")))
            painter.setFont(QFont("sans", 9))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Bird's-eye view\n(select Geometric touch)")
            return

        to_widget, scale = self._make_transform()

        # Floor rectangle
        if self._corners is not None and len(self._corners) >= 3:
            poly = QPolygonF([to_widget(x, y) for (x, y) in self._corners])
            painter.setBrush(QColor(42, 42, 42, 140))
            painter.setPen(QPen(QColor("#888"), 1.5))
            painter.drawPolygon(poly)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        # Per-camera projected foot-points
        for cam in range(3):
            painter.setPen(QPen(CAM_COLORS[cam], 1.0))
            painter.setBrush(CAM_COLORS[cam])
            for (x, y) in self._cam_points[cam]:
                painter.drawEllipse(to_widget(x, y), 3.5, 3.5)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Trail
        if len(self._trail) > 1:
            painter.setPen(QPen(ACTOR_COLOR, 1.2, Qt.PenStyle.SolidLine))
            pts = [to_widget(x, y) for (x, y) in self._trail]
            for a, b in zip(pts[:-1], pts[1:]):
                painter.drawLine(a, b)

        # Contact links (red) with distance labels
        painter.setFont(QFont("monospace", 8))
        for (i, j) in self._pairs:
            if i < len(self._actors) and j < len(self._actors):
                pa, pb = self._actors[i], self._actors[j]
                wa, wb = to_widget(*pa), to_widget(*pb)
                painter.setPen(QPen(QColor("#ff3b3b"), 2.0))
                painter.drawLine(wa, wb)
                d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                mid = QPointF((wa.x() + wb.x()) / 2, (wa.y() + wb.y()) / 2)
                painter.drawText(mid, f"{d:.2f}{self._unit}")

        # Fused actors
        painter.setPen(QPen(QColor("#000"), 1.0))
        painter.setBrush(ACTOR_COLOR)
        for (x, y) in self._actors:
            c = to_widget(x, y)
            painter.drawEllipse(c, 6.0, 6.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Scale/legend
        painter.setPen(QPen(QColor("#aaa")))
        painter.setFont(QFont("monospace", 8))
        painter.drawText(QPointF(6, self.height() - 6),
                         f"unit={self._unit}  δ={self._delta_m:g}  actors={len(self._actors)}")
