"""3-D room model rendering helpers (pyqtgraph.opengl).

Pure geometry + GL item construction, isolated from the dialog so the math is
testable without a display. ``HAS_GL`` tells callers whether the OpenGL backend
is importable; if not, the dialog shows a 2-D fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import pyqtgraph.opengl as gl  # noqa: F401
    HAS_GL = True
except Exception:  # pragma: no cover - depends on optional dep / GPU
    HAS_GL = False

ROOM_MAX_M = 5.0  # hard cap per the spec (max volume 5x5x5)


@dataclass
class CameraPose:
    """A camera mounted in the room. Coordinates in metres, room origin at a
    floor corner; +X length, +Y width, +Z up."""

    camera_id: int
    x: float
    y: float
    z: float
    yaw_deg: float            # 0 → looking along +X
    tilt_deg: float           # downward depression below horizontal
    fov_deg: tuple[float, float] = (60.0, 45.0)

    def forward(self) -> np.ndarray:
        yaw, tilt = math.radians(self.yaw_deg), math.radians(self.tilt_deg)
        return np.array([
            math.cos(tilt) * math.cos(yaw),
            math.cos(tilt) * math.sin(yaw),
            -math.sin(tilt),
        ], dtype=float)

    def frustum_corners(self, length: float = 2.0) -> np.ndarray:
        """Four ray endpoints approximating the FOV pyramid, for drawing."""
        f = self.forward()
        up0 = np.array([0.0, 0.0, 1.0])
        right = np.cross(f, up0)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        up /= np.linalg.norm(up)
        th = math.radians(self.fov_deg[0] / 2.0)
        tv = math.radians(self.fov_deg[1] / 2.0)
        origin = np.array([self.x, self.y, self.z])
        corners = []
        for sh in (-1, 1):
            for sv in (-1, 1):
                d = f + math.tan(th) * sh * right + math.tan(tv) * sv * up
                d /= np.linalg.norm(d)
                corners.append(origin + d * length)
        return np.array(corners)


def clamp_room(length: float, width: float, height: float) -> tuple[float, float, float]:
    """Clamp each dimension to (0, ROOM_MAX_M]."""
    def c(v: float) -> float:
        return max(0.1, min(ROOM_MAX_M, float(v)))
    return c(length), c(width), c(height)


def box_edges(length: float, width: float, height: float) -> list[np.ndarray]:
    """Return line segments (each (2,3)) for the 12 edges of the room box."""
    L, W, H = length, width, height
    v = {
        0: (0, 0, 0), 1: (L, 0, 0), 2: (L, W, 0), 3: (0, W, 0),
        4: (0, 0, H), 5: (L, 0, H), 6: (L, W, H), 7: (0, W, H),
    }
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # floor
        (4, 5), (5, 6), (6, 7), (7, 4),   # ceiling
        (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
    ]
    return [np.array([v[a], v[b]], dtype=float) for a, b in edges]


def populate_gl_view(view, length: float, width: float, height: float,
                     rect_wh: tuple[float, float] | None,
                     cameras: list[CameraPose]) -> None:
    """(Re)build the GL scene: room box, floor rectangle, camera frustums."""
    import pyqtgraph.opengl as gl

    view.clear()
    L, W, H = length, width, height

    # Room wireframe (grey)
    for seg in box_edges(L, W, H):
        view.addItem(gl.GLLinePlotItem(pos=seg, color=(0.6, 0.6, 0.6, 1.0), width=1.5, antialias=True))

    # Floor reference rectangle (cyan), centred on the floor
    if rect_wh is not None:
        rw, rh = rect_wh
        cx, cy = L / 2.0, W / 2.0
        rect = np.array([
            [cx - rw / 2, cy - rh / 2, 0.01],
            [cx + rw / 2, cy - rh / 2, 0.01],
            [cx + rw / 2, cy + rh / 2, 0.01],
            [cx - rw / 2, cy + rh / 2, 0.01],
            [cx - rw / 2, cy - rh / 2, 0.01],
        ])
        view.addItem(gl.GLLinePlotItem(pos=rect, color=(0.3, 0.9, 0.9, 1.0), width=2.0, antialias=True))

    # Cameras: position dot + forward ray + FOV pyramid
    cam_colors = [(1.0, 0.42, 0.42, 1.0), (0.3, 0.8, 0.78, 1.0), (1.0, 0.88, 0.4, 1.0)]
    for cam in cameras:
        color = cam_colors[cam.camera_id % 3]
        origin = np.array([[cam.x, cam.y, cam.z]])
        view.addItem(gl.GLScatterPlotItem(pos=origin, color=color, size=10.0))
        corners = cam.frustum_corners(length=min(L, W) * 0.6 + 0.5)
        o = origin[0]
        for c in corners:
            view.addItem(gl.GLLinePlotItem(pos=np.array([o, c]), color=color, width=1.0, antialias=True))
        # connect the 4 corners into a rectangle
        ring = np.array([corners[0], corners[1], corners[3], corners[2], corners[0]])
        view.addItem(gl.GLLinePlotItem(pos=ring, color=color, width=1.0, antialias=True))
