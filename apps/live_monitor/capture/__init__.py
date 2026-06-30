"""Frame capture backends.

Everything here yields :class:`thermal_algorithms.core.types.Frame` objects —
the lingua franca the whole algorithm stack already consumes — so the rest of
the app is agnostic to *how* frames arrive.

Backends:
    * :class:`~apps.live_monitor.capture.senxor_source.SenxorSource`
        Real Raspberry Pi hardware (Waveshare/MI48 via ``pysenxor-lite``).
    * :class:`~apps.live_monitor.capture.playback_source.PlaybackSource`
        Replays recorded ``chN_raw_data.npz`` sessions — works on any machine.
    * :class:`~apps.live_monitor.capture.synthetic_source.SyntheticSource`
        Procedural warm-blob generator for smoke tests / CI.
"""

from apps.live_monitor.capture.base import CameraHealth, FrameSource
from apps.live_monitor.capture.playback_source import PlaybackSource
from apps.live_monitor.capture.synthetic_source import SyntheticSource

__all__ = [
    "CameraHealth",
    "FrameSource",
    "PlaybackSource",
    "SyntheticSource",
    # SenxorSource is imported lazily (needs pysenxor-lite, Pi-only).
]
