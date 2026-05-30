"""File I/O helpers for thermal data.

The Data Acquisition Software (§ 5.1.2) records each session as raw CSV
thermal matrices — one CSV per frame, named with a timestamp. This module
loads those files into `Frame` objects so the algorithm pipeline can consume
them directly.

It also provides simple sequence loaders (a whole recording → an iterator of
Frames), and writers for produced/preprocessed frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Frame


# ---------------------------------------------------------------------------
# Single-frame I/O
# ---------------------------------------------------------------------------

def load_frame_csv(
    path: str | Path,
    *,
    timestamp: Optional[float] = None,
    camera_id: Optional[int] = None,
    expected_profile: Optional[SensorProfile] = None,
    delimiter: str = ",",
) -> Frame:
    """Load a single thermal frame from a CSV file.

    The DataRecorder GUI (§ 5.1.2) writes one CSV per frame with H rows and
    W columns of float temperatures.

    Args:
        path: CSV file path.
        timestamp: If omitted, parsed from the filename (expects an integer
            or float prefix before the extension, e.g. `1715600000.123.csv`).
            Falls back to the file's mtime if no numeric prefix is found.
        camera_id: Source camera index, in {0, 1, 2}.
        expected_profile: If provided, raises if the loaded shape does not
            match `expected_profile.resolution`.
        delimiter: CSV delimiter (default ',').
    """
    path = Path(path)
    data = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2-D CSV, got shape {data.shape} from {path}."
        )

    if expected_profile is not None:
        ew, eh = expected_profile.resolution
        if data.shape != (eh, ew):
            raise ValueError(
                f"Frame shape {data.shape} does not match "
                f"{expected_profile.name} expected (H, W) = ({eh}, {ew}). File: {path}"
            )

    if timestamp is None:
        timestamp = _infer_timestamp_from_path(path)

    return Frame(data=data, timestamp=timestamp, camera_id=camera_id)


def save_frame_csv(
    frame: Frame,
    path: str | Path,
    *,
    delimiter: str = ",",
    fmt: str = "%.4f",
) -> None:
    """Write a Frame's data array to CSV (matches DataRecorder format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, frame.data, delimiter=delimiter, fmt=fmt)


# ---------------------------------------------------------------------------
# Sequence I/O
# ---------------------------------------------------------------------------

def load_recording(
    directory: str | Path,
    *,
    camera_id: Optional[int] = None,
    expected_profile: Optional[SensorProfile] = None,
    glob_pattern: str = "*.csv",
) -> Iterator[Frame]:
    """Iterate all frames in a recording directory, sorted by timestamp.

    Args:
        directory: Folder containing per-frame CSV files.
        camera_id: Camera index to stamp on every yielded Frame.
        expected_profile: Optional shape check applied to every frame.
        glob_pattern: Filename pattern (default '*.csv').

    Yields:
        Frame instances in chronological order.
    """
    directory = Path(directory)
    files = sorted(directory.glob(glob_pattern), key=_infer_timestamp_from_path)
    for f in files:
        yield load_frame_csv(
            f,
            camera_id=camera_id,
            expected_profile=expected_profile,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_timestamp_from_path(path: Path) -> float:
    """Best-effort timestamp extraction from a filename.

    Tries to parse the file stem as a float; falls back to file mtime.
    """
    try:
        return float(path.stem)
    except ValueError:
        return float(path.stat().st_mtime) if path.exists() else 0.0
