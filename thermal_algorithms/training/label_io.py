"""Label parsers for the project's dataset format.

Two label artifacts are supported:

1. **YOLO-format bounding box labels** — one `.txt` file per labeled frame at
   `ch{N}_frames/frame_XXXXX.txt`. Each non-empty line is:

       class_idx  cx  cy  w  h

   with `cx, cy, w, h` normalized to [0, 1] over the frame dimensions. The
   class indices are defined by a sibling `classes.txt` file (one class name
   per line; line index = class id).

   Missing `.txt` files mean "no detections in this frame" — i.e. a valid
   negative example, not a corruption.

2. **Activity-level multi-labels** — a single `labels.xlsx` workbook with
   one row per (channel, frame) and binary columns for activities
   ("single human", "two humans", "three humans", "fire", "touch", ...).

This module returns `Detection` instances (with bboxes denormalized to pixel
coords matching `Frame.data` shape) and lightweight dataclasses for the
activity labels.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Optional

import numpy as np

from thermal_algorithms.core.types import Detection


# ---------------------------------------------------------------------------
# Class-index convention (user-confirmed, matches annotation tool config)
# ---------------------------------------------------------------------------

FIRE_CLASS_ID: Final[int] = 0
"""YOLO class index for fire / ignition-source annotations."""

PERSON_CLASS_ID: Final[int] = 1
"""YOLO class index for person / human-presence annotations."""

# Contact has no natural bounding-box representation at 32×24 resolution; its
# labels are stored in per-session contact_labels.csv files instead.
CONTACT_LABEL_FILENAME: Final[str] = "contact_labels.csv"
"""Default filename for per-session contact labels."""


# Activity-label columns shipped in labels.xlsx (Engineering Report § 4.3.1).
ACTIVITY_LABEL_COLUMNS = (
    "single human",
    "two humans",
    "three humans",
    "fire",
    "touch",
)


@dataclass(frozen=True)
class ActivityLabel:
    """One row of the multi-label activity workbook."""
    channel: int
    frame: int
    flags: dict[str, int]   # {col_name: 0/1}


# ---------------------------------------------------------------------------
# Class list (classes.txt)
# ---------------------------------------------------------------------------

def load_classes_file(path: str | Path) -> dict[int, str]:
    """Read `classes.txt` → {class_id: name}. One name per line, blanks skipped."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"classes.txt not found at {p}")
    names = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return {i: name for i, name in enumerate(names)}


# ---------------------------------------------------------------------------
# YOLO bounding-box labels
# ---------------------------------------------------------------------------

def load_yolo_labels(
    label_path: str | Path,
    *,
    frame_shape: tuple[int, int],
    camera_id: Optional[int] = None,
    class_filter: Optional[Iterable[int]] = None,
) -> list[Detection]:
    """Parse a single YOLO `.txt` label file into `Detection`s.

    Args:
        label_path: Path to `frame_XXXXX.txt`. If the file does not exist,
            returns `[]` (this is the dataset's convention for empty frames).
        frame_shape: `(height, width)` used to denormalize bbox coords.
        camera_id: Stamped on every returned Detection.
        class_filter: If given, only Detections whose class_id is in this set
            are returned. Useful for selecting humans (id=1) vs fire (id=0).

    Returns:
        list of Detection in pixel coordinates. bbox = (x_topleft, y_topleft,
        w, h). Boxes are clamped to the frame extents.
    """
    p = Path(label_path)
    if not p.is_file():
        return []

    h, w = frame_shape
    filter_set: Optional[set[int]] = set(class_filter) if class_filter is not None else None

    detections: list[Detection] = []
    for line_no, raw in enumerate(p.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{p}:{line_no}: expected 5 tokens (class cx cy w h), got {len(parts)}: {line!r}"
            )
        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"{p}:{line_no}: class index must be an integer (use classes.txt to "
                f"resolve names elsewhere); got {parts[0]!r}"
            ) from exc

        if filter_set is not None and class_id not in filter_set:
            continue

        try:
            cx_n, cy_n, w_n, h_n = (float(x) for x in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{p}:{line_no}: non-numeric bbox coord in {line!r}") from exc

        if not all(0.0 <= v <= 1.0 for v in (cx_n, cy_n, w_n, h_n)):
            # Many real-world YOLO exports allow slight overflow; warn-by-clamp
            cx_n = min(1.0, max(0.0, cx_n))
            cy_n = min(1.0, max(0.0, cy_n))
            w_n = min(1.0, max(0.0, w_n))
            h_n = min(1.0, max(0.0, h_n))

        bw = w_n * w
        bh = h_n * h
        bx = cx_n * w - bw / 2.0
        by = cy_n * h - bh / 2.0

        # Clamp to frame extents.
        bx = max(0.0, bx)
        by = max(0.0, by)
        bw = min(bw, w - bx)
        bh = min(bh, h - by)

        detections.append(
            Detection(
                bbox=(bx, by, bw, bh),
                score=1.0,
                class_id=class_id,
                camera_id=camera_id,
            )
        )
    return detections


def list_label_files(label_dir: str | Path) -> list[Path]:
    """Return all `frame_XXXXX.txt` files in a `ch{N}_frames/` directory,
    sorted by frame index. Excludes `classes.txt` and other auxiliaries."""
    return sorted(Path(label_dir).glob("frame_*.txt"))


def frame_index_from_label_path(path: str | Path) -> int:
    """Extract the integer frame index from a `frame_XXXXX.txt` filename."""
    p = Path(path)
    stem = p.stem
    if not stem.startswith("frame_"):
        raise ValueError(f"Expected 'frame_XXXXX' filename; got {p.name!r}")
    return int(stem.split("_", 1)[1])


# ---------------------------------------------------------------------------
# Activity labels (labels.xlsx)
# ---------------------------------------------------------------------------

def load_activity_labels(
    xlsx_path: str | Path,
    *,
    columns: Iterable[str] = ACTIVITY_LABEL_COLUMNS,
) -> dict[tuple[int, int], ActivityLabel]:
    """Parse `labels.xlsx` → {(channel, frame): ActivityLabel}.

    Requires pandas (and an xlsx engine like openpyxl). Returns an empty dict
    if the file does not exist.
    """
    p = Path(xlsx_path)
    if not p.is_file():
        return {}

    import pandas as pd  # local import: pandas isn't a hard requirement

    df = pd.read_excel(p)
    required = {"channel", "frame"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels.xlsx is missing required columns: {sorted(missing)}")

    out: dict[tuple[int, int], ActivityLabel] = {}
    for _, row in df.iterrows():
        ch = int(row["channel"])
        fr = int(row["frame"])
        flags = {col: int(row[col]) for col in columns if col in df.columns and not _is_nan(row[col])}
        out[(ch, fr)] = ActivityLabel(channel=ch, frame=fr, flags=flags)
    return out


def _is_nan(value) -> bool:
    try:
        return value != value  # NaN trick
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Contact labels  (contact_labels.csv)
# ---------------------------------------------------------------------------
#
# Format — one header row, then one data row per labeled frame:
#
#     frame_idx,contact
#     0,0
#     1,0
#     5,1
#     6,1
#
# Rules:
#   * frame_idx : 0-based integer frame index within the session.
#   * contact   : 1 = physical contact between patients; 0 = no contact.
#   * Lines beginning with '#' are ignored (comments).
#   * Frames absent from the file are excluded from ContactFrameDataset.
#     Annotate every frame you intend to use for training.
#
# The CSV lives at the session root (next to ch0_raw_data.npz etc.),
# one file per session.  It is session-level rather than per-channel because
# contact is a scene event that applies to all camera views simultaneously.

def load_contact_labels(path: str | Path) -> dict[int, int]:
    """Parse a contact_labels.csv file.

    Args:
        path: Path to ``contact_labels.csv``.  Returns empty dict if the file
            does not exist.

    Returns:
        ``{frame_idx: contact_binary}`` — only rows that appeared in the file.
        ``contact_binary`` is 0 or 1.

    Raises:
        ValueError: If any row has malformed data.
    """
    p = Path(path)
    if not p.is_file():
        return {}

    out: dict[int, int] = {}
    with p.open(newline="", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Skip header row containing non-numeric first field
            parts = [s.strip() for s in line.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"{p}:{line_no}: expected 2 columns (frame_idx, contact); "
                    f"got {len(parts)}: {line!r}"
                )
            try:
                frame_idx = int(parts[0])
                contact = int(parts[1])
            except ValueError:
                # Skip only when the row looks like a column header: both
                # fields are non-numeric strings (e.g. "frame_idx,contact").
                # A row like "abc,1" has a numeric second field → malformed data.
                first_numeric = True
                try:
                    int(parts[0])
                except ValueError:
                    first_numeric = False
                if not first_numeric:
                    second_numeric = True
                    try:
                        int(parts[1])
                    except ValueError:
                        second_numeric = False
                    if not second_numeric:
                        continue  # header row
                raise ValueError(
                    f"{p}:{line_no}: non-integer value in {line!r}"
                )
            if contact not in (0, 1):
                raise ValueError(
                    f"{p}:{line_no}: contact must be 0 or 1; got {contact!r}"
                )
            out[frame_idx] = contact
    return out


def save_contact_labels(labels: dict[int, int], path: str | Path) -> None:
    """Write contact labels to a contact_labels.csv file.

    Args:
        labels: ``{frame_idx: contact_binary}`` mapping to write.
        path: Destination path (created or overwritten).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_idx", "contact"])
        for frame_idx in sorted(labels):
            writer.writerow([frame_idx, int(labels[frame_idx])])


def has_contact_labels(session_root: str | Path) -> bool:
    """Return True if a contact_labels.csv exists in the given directory."""
    return (Path(session_root) / CONTACT_LABEL_FILENAME).is_file()
