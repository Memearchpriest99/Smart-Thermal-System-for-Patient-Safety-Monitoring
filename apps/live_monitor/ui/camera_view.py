"""One camera panel: thermal image + detection/fire overlays + a hover readout."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from thermal_algorithms.core.types import Detection
from apps.live_monitor import rendering


class CameraView(QWidget):
    """Displays the latest RGB frame for one camera with overlaid boxes.

    ``set_frame`` takes the colorized RGB array (so the heavy colormap work
    happens once, off the paint path), the source-frame temperature array (for
    the hover readout), and the per-camera detections/fire boxes to draw.
    """

    def __init__(self, camera_id: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._camera_id = camera_id
        self._pixmap: Optional[QPixmap] = None
        self._temp: Optional[np.ndarray] = None       # source °C frame
        self._src_w = 1
        self._src_h = 1
        self._person: list[Detection] = []
        self._fire: list[Detection] = []
        self._title = f"Camera {camera_id}"
        self._status = "waiting…"
        self._hover: Optional[tuple[int, int]] = None
        self.setMinimumSize(160, 124)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#111;")

    # ---- data in --------------------------------------------------------

    def set_frame(
        self,
        rgb: np.ndarray,
        *,
        temp: Optional[np.ndarray] = None,
        person: Sequence[Detection] = (),
        fire: Sequence[Detection] = (),
        status: str = "",
    ) -> None:
        self._src_h, self._src_w = rgb.shape[:2]
        self._temp = temp
        self._pixmap = QPixmap.fromImage(rendering.to_qimage(rgb))
        self._person = list(person)
        self._fire = list(fire)
        if status:
            self._status = status
        self.update()

    def set_status(self, status: str) -> None:
        self._status = status
        self.update()

    # ---- geometry helpers ----------------------------------------------

    def _image_rect(self) -> QRectF:
        """Letterboxed destination rect preserving the source aspect ratio."""
        ww, wh = self.width(), self.height()
        if self._src_w == 0 or self._src_h == 0:
            return QRectF(0, 0, ww, wh)
        scale = min(ww / self._src_w, wh / self._src_h)
        dw, dh = self._src_w * scale, self._src_h * scale
        return QRectF((ww - dw) / 2.0, (wh - dh) / 2.0, dw, dh)

    # ---- painting -------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111"))
        rect = self._image_rect()

        if self._pixmap is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))

            sx = rect.width() / max(1, self._src_w)
            sy = rect.height() / max(1, self._src_h)
            painter.save()
            painter.translate(rect.topLeft())
            painter.setFont(QFont("monospace", 8))
            rendering.draw_detections(painter, self._person, scale_x=sx, scale_y=sy,
                                      color=rendering.CLASS_COLORS[1], label="person")
            rendering.draw_detections(painter, self._fire, scale_x=sx, scale_y=sy,
                                      color=rendering.CLASS_COLORS[0], label="FIRE")
            painter.restore()

        # Title + status banner
        painter.setPen(QPen(QColor("#ddd")))
        painter.setFont(QFont("sans", 9, QFont.Weight.Bold))
        painter.drawText(QPointF(6, 16), self._title)
        painter.setPen(QPen(QColor("#7fd")))
        painter.setFont(QFont("monospace", 8))
        painter.drawText(QPointF(6, self.height() - 6), self._status)

        # Hover temperature readout
        if self._hover is not None and self._temp is not None:
            self._draw_hover(painter, rect)

    def _draw_hover(self, painter: QPainter, rect: QRectF) -> None:
        mx, my = self._hover
        if not rect.contains(QPointF(mx, my)):
            return
        u = int((mx - rect.left()) / max(1e-6, rect.width()) * self._src_w)
        v = int((my - rect.top()) / max(1e-6, rect.height()) * self._src_h)
        if 0 <= v < self._temp.shape[0] and 0 <= u < self._temp.shape[1]:
            t = float(self._temp[v, u])
            painter.setPen(QPen(QColor("#fff")))
            painter.setFont(QFont("monospace", 8))
            painter.drawText(QPointF(mx + 8, my - 6), f"({u},{v}) {t:.1f}°C")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        p = event.position()
        self._hover = (int(p.x()), int(p.y()))
        self.update()

    def leaveEvent(self, _event) -> None:
        self._hover = None
        self.update()
