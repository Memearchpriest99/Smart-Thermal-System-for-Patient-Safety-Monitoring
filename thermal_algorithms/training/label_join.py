"""Join room-level event-interval labels (a session's ``labels.csv``) to
per-frame fire/human/contact booleans, for the ``synth_room_*``/``room-1``
sources read via ``hdf5_source.py``.

``labels.csv`` schema (one row per event interval, not per frame)::

    Room_ID,Date,Start_Time,End_Time,Event_Class,Event_Class_ID,Cameras,Timestamp

Event_Class values observed across synth_room_1-5 + room-1 combine several
independent concepts into one string (e.g. ``Contact_2+Humans+Fire``), so
per-problem labels are derived via substring rules, not the class name
verbatim. Agreed mapping:

    fire_positive:    class contains "Fire"
    human_positive:   class is not "Empty" and not exactly "Fire" (bare fire,
                       no human present)
    contact_positive: class contains "Contact" OR "Violence" (Violence folds
                       into contact-positive per user decision)

The ``+fall`` classes (patient-fall scenario) are explicitly out of scope —
no dedicated flag; those frames simply count as human-positive like any
other non-Empty class, per user decision to ignore this for now.

There is no bounding-box ground truth in this source at all — only these
frame-level presence booleans. Bbox-dependent training/eval stays scoped to
``waveshare_work``.
"""

from __future__ import annotations

import bisect
import csv
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LabelInterval:
    """One row of ``labels.csv``: a room-wide event spanning a time range."""

    room_id: str
    date: str
    start: _dt.datetime
    end: _dt.datetime
    event_class: str
    event_class_id: int
    cameras: tuple[int, ...]


def _parse_time_of_day(date_str: str, time_str: str) -> _dt.datetime:
    """Combine a ``YYYY-MM-DD`` date with a ``HH:MM:SS.fff`` time-of-day."""
    y, mo, d = (int(x) for x in date_str.split("-"))
    h_str, mi_str, rest = time_str.split(":")
    s_str, _, ms_str = rest.partition(".")
    micros = int(ms_str.ljust(6, "0")[:6]) if ms_str else 0
    return _dt.datetime(y, mo, d, int(h_str), int(mi_str), int(s_str), micros)


def load_room_labels(
    csv_path: str | Path,
    *,
    room_id: Optional[str] = None,
    date: Optional[str] = None,
) -> list[LabelInterval]:
    """Parse ``labels.csv`` into a list of ``LabelInterval``, sorted by start.

    Args:
        csv_path: path to the session's (or room's) ``labels.csv``.
        room_id: if given, only rows for this ``Room_ID`` are kept.
        date: if given, only rows for this ``Date`` (``YYYY-MM-DD``) are kept.
    """
    path = Path(csv_path)
    intervals: list[LabelInterval] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if room_id is not None and row["Room_ID"] != room_id:
                continue
            if date is not None and row["Date"] != date:
                continue
            start = _parse_time_of_day(row["Date"], row["Start_Time"])
            end = _parse_time_of_day(row["Date"], row["End_Time"])
            cameras = tuple(
                int(c) for c in row["Cameras"].split("|") if c.strip() != ""
            )
            intervals.append(
                LabelInterval(
                    room_id=row["Room_ID"],
                    date=row["Date"],
                    start=start,
                    end=end,
                    event_class=row["Event_Class"],
                    event_class_id=int(row["Event_Class_ID"]),
                    cameras=cameras,
                )
            )
    intervals.sort(key=lambda iv: iv.start)
    return intervals


def classify_event(event_class: str) -> dict[str, bool]:
    """Map an ``Event_Class`` string to independent fire/human/contact flags.

    See module docstring for the agreed substring rules.
    """
    is_bare_fire = event_class == "Fire"
    is_empty = event_class == "Empty"
    return {
        "fire": "Fire" in event_class,
        "human": not is_empty and not is_bare_fire,
        "contact": ("Contact" in event_class) or ("Violence" in event_class),
    }


def detect_room_id(csv_path: str | Path, *, date: str) -> str:
    """Auto-detect the Room_ID a labels.csv actually uses for `date`.

    Confirmed 2026-08-07 (project owner): synth_room_3/4/5's labels.csv files
    were each copied from a differently-numbered room in the original
    generation set and never relabeled to match their containing folder's
    name -- synth_room_3's file says "synth_room_4" throughout,
    synth_room_4 says "synth_room_6", synth_room_5 says "synth_room_8",
    each internally consistent (not randomly corrupted, not a pixel/label
    mismatch -- just a rename-without-relabel). Filtering by the
    folder-derived room_id therefore finds zero rows and
    LabelJoiner.__init__ raises "no label intervals given". This scans the
    file's own Room_ID column for `date` and returns whatever it actually
    contains, so the caller filters by ground truth rather than an assumed
    naming convention.

    Raises ValueError if zero or more than one distinct Room_ID is found
    for this date -- multiple distinct values would be a different, more
    serious problem than the confirmed case above and needs human review,
    not an automatic pick.
    """
    path = Path(csv_path)
    found: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["Date"] == date:
                found.add(row["Room_ID"])
    if len(found) == 0:
        raise ValueError(f"{path}: no rows found for date={date!r} at all.")
    if len(found) > 1:
        raise ValueError(
            f"{path}: date={date!r} has multiple distinct Room_ID values "
            f"{sorted(found)} -- needs human review, not an automatic pick."
        )
    return next(iter(found))


class LabelJoiner:
    """Looks up the enclosing ``LabelInterval`` (and derived fire/human/
    contact booleans) for a given frame timestamp (Unix-epoch seconds, e.g.
    from ``Frame.timestamp``).

    Every session inspected so far has back-to-back intervals (the next
    row's Start_Time is at most a few tens of ms after the previous row's
    End_Time), so a frame almost always falls inside exactly one interval.
    ``labels_for`` returns ``None`` for a timestamp that lands in one of
    these small inter-interval gaps (or before the first / after the last
    interval) rather than silently guessing — callers should treat ``None``
    as "drop this frame", not as an error.
    """

    def __init__(self, intervals: list[LabelInterval]) -> None:
        if not intervals:
            raise ValueError("no label intervals given")
        self._intervals = intervals
        self._starts_epoch = [iv.start.timestamp() for iv in intervals]

    @classmethod
    def from_csv(
        cls, csv_path: str | Path, *, room_id: Optional[str] = None, date: Optional[str] = None
    ) -> "LabelJoiner":
        return cls(load_room_labels(csv_path, room_id=room_id, date=date))

    def interval_for(self, timestamp: float) -> Optional[LabelInterval]:
        i = bisect.bisect_right(self._starts_epoch, timestamp) - 1
        if i < 0:
            return None
        interval = self._intervals[i]
        if timestamp >= interval.end.timestamp():
            return None
        return interval

    def labels_for(self, timestamp: float) -> Optional[dict]:
        """Returns ``{"event_class": str, "fire": bool, "human": bool,
        "contact": bool}``, or ``None`` if ``timestamp`` falls outside every
        interval (a small inter-interval gap, or before/after the session)."""
        interval = self.interval_for(timestamp)
        if interval is None:
            return None
        return {"event_class": interval.event_class, **classify_event(interval.event_class)}
