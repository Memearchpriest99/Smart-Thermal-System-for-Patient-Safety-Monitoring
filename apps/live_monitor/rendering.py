"""Thermal-frame rendering and overlays.

The RGB generation is Qt-free (returns numpy ``(H, W, 3)`` uint8) so it can be
unit-tested headlessly. ``to_qimage`` and the QPainter overlay helpers import
PyQt6 lazily and are only used by the UI.

The colormap is the same baked **inferno** LUT the rest of the repo uses
(``image_annotator``), which also matches ``senxor.proc``'s default colormap —
so the on-screen feed looks identical whether frames come from the MI48 driver
or a recording.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from thermal_algorithms.core.types import Detection

if TYPE_CHECKING:  # avoid importing PyQt at module load
    from PyQt6.QtGui import QImage


# Exact matplotlib 'inferno' 256-entry RGB LUT, baked so rendering needs neither
# matplotlib nor cv2 (copied from image_annotator/annotator.py — the colormap
# the dataset PNGs and senxor.proc both use).
_INFERNO_LUT_B64 = (
    "AAAEAQAFAQEGAQEIAgEKAgIMAgIOAwIQBAMSBAMUBQQXBgQZBwUbCAUdCQYfCgciCwckDAgmDQgp"
    "DgkrEAktEQowEgoyFAs0FQs3Fgs5GAw8GQw+GwxBHAxDHgxFHwxIIQxKIwxMJAxPJgxRKAtTKQtV"
    "KwtXLQtZLwpbMQpcMgpeNApfNglhOAliOQljOwlkPQllPglmQApnQgpoRApoRQppRwtqSQtqSgxr"
    "TAxrTQ1sTw1sUQ5sUg5tVA9tVQ9tVxBuWRBuWhFuXBJuXRJuXxNuYRNuYhRuZBVuZRVuZxZuaRZu"
    "ahdubBhubRhubxlucRluchpudBpudRtudxxteBxteh1tfB1tfR5tfx5sgB9sgiBshCBrhSFrhyFr"
    "iCJqiiJqjCNpjSNpjyRpkCVokiVokyZnlSZnlydmmCdmmihlmylknSlknypjoCpjoitioyxhpSxg"
    "pi1gqC5fqS5eqy9erTBdrjBcsDFbsTJaszJatDNZtjRYtzVXuTVWujZVvDdUvThTvzlSwDpRwTpQ"
    "wztPxDxOxj1Nxz5MyD9LykBKy0FJzEJIzkNHz0RG0EVF0kZE00dD1EhC1UpB10s/2Ew+2U092k48"
    "21A73VE63lI431M34FU24VY14lc041kz5Fox5Vww5l0v514u6GAt6WEr6mMq62Qp62Yo7Gcm7Wkl"
    "7mok72wj724h8G8g8XEf8XMd8nQc83Yb83gZ9HkY9XsX9X0V9n4U9oAT94IS94QQ+IUP+IcO+IkM"
    "+YsL+YwK+Y4J+pAI+pIH+pQH+5YG+5cG+5kG+5sG+50H/J8H/KEI/KMJ/KUK/KYM/KgN/KoP/KwR"
    "/K4S/LAU/LIW/LQY+7Ya+7gd+7of+7wh+74j+sAm+sIo+sQq+sYt+ccv+cky+cs1+M03+M8699E9"
    "99NA9tVD9tdG9dlJ9dtM9N1P9N9T9OFW8+Na8+Vd8uZh8uhl8upp8ext8e1x8e918fF58vJ98vSC"
    "8/WG8/aK9PiO9fmS9vqW+Pua+fyd+v2h/P+k"
)
INFERNO_LUT: np.ndarray = np.frombuffer(
    base64.b64decode(_INFERNO_LUT_B64), dtype=np.uint8
).reshape(256, 3)


def colorize(
    frame: np.ndarray,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Map a 2-D thermal frame (°C) to an ``(H, W, 3)`` uint8 RGB image.

    Per-frame min/max normalization by default (matches the dataset previews);
    pass ``vmin``/``vmax`` for a fixed temperature scale (steadier visuals and
    better for spotting absolute hot spots).
    """
    f = np.asarray(frame, dtype=np.float32)
    lo = float(np.nanmin(f)) if vmin is None else float(vmin)
    hi = float(np.nanmax(f)) if vmax is None else float(vmax)
    if hi - lo < 1e-6:
        idx = np.zeros(f.shape, dtype=np.uint8)
    else:
        idx = ((f - lo) / (hi - lo) * 255.0).round().clip(0, 255).astype(np.uint8)
    return INFERNO_LUT[idx]


def residual_view(residual: np.ndarray) -> np.ndarray:
    """Colorize a Tateno residual (already L1 magnitude) with a fixed floor at 0
    so a static background reads as uniformly dark."""
    return colorize(residual, vmin=0.0)


def to_qimage(rgb: np.ndarray) -> "QImage":
    """Wrap an ``(H, W, 3)`` uint8 RGB array as a QImage (deep-copied)."""
    from PyQt6.QtGui import QImage

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return img.copy()  # detach from the numpy buffer


# --- Overlay drawing (UI-side; imports QtGui lazily) ----------------------

# Per-class colors for detection boxes (RGB).
CLASS_COLORS = {
    0: (255, 80, 80),     # fire / ignition source → red
    1: (80, 200, 255),    # person → cyan
}
DEFAULT_BOX_COLOR = (255, 215, 64)


def draw_detections(
    painter,
    detections: Sequence[Detection],
    *,
    scale_x: float,
    scale_y: float,
    label: Optional[str] = None,
    color: Optional[tuple] = None,
    draw_foot: bool = True,
) -> None:
    """Draw detection boxes + foot-points onto a QPainter already mapped to the
    displayed image. ``scale_x/scale_y`` convert source-frame pixels to widget
    pixels. Pass ``color`` to force a category color (fire vs person) rather
    than keying off ``class_id`` (which is 0 for *person* in this project).
    """
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtGui import QColor, QPen

    for det in detections:
        x, y, w, h = det.bbox
        r, g, b = color if color is not None else CLASS_COLORS.get(det.class_id, DEFAULT_BOX_COLOR)
        pen = QPen(QColor(r, g, b))
        pen.setWidthF(2.0)
        painter.setPen(pen)
        rect = QRectF(x * scale_x, y * scale_y, w * scale_x, h * scale_y)
        painter.drawRect(rect)

        tag = label if label is not None else f"{det.score:.2f}"
        painter.drawText(QPointF(rect.left(), max(10.0, rect.top() - 3.0)), tag)

        if draw_foot:
            fx, fy = det.foot_point
            painter.setBrush(QColor(r, g, b))
            painter.drawEllipse(QPointF(fx * scale_x, fy * scale_y), 3.0, 3.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
