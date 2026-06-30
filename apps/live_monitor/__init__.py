"""Live Thermal Monitor — real-time Raspberry Pi 5 operator app.

A PyQt6 desktop app that captures three live Waveshare/MI48 thermal cameras,
runs the ``thermal_algorithms`` detectors in parallel (fire / person / touch),
lets the operator hot-swap the algorithm per task at runtime, draws live
bounding boxes and alerts, shows a live bird's-eye homography view for the
geometric touch detector, and provides a room-geometry calibration mode with a
3-D model.

The non-UI layers (``capture``, ``detectors``, ``rendering``, ``runner``) are
deliberately Qt-free so they can be unit-tested without a display. PyQt only
appears in :mod:`apps.live_monitor.pipeline_worker` and :mod:`apps.live_monitor.ui`.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
