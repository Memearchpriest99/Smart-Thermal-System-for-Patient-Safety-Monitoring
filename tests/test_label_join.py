"""Tests for the labels.csv interval -> per-frame fire/human/contact join."""

import datetime as dt

import pytest

from thermal_algorithms.training.label_join import (
    LabelInterval,
    LabelJoiner,
    classify_event,
    load_room_labels,
)

CSV_HEADER = "Room_ID,Date,Start_Time,End_Time,Event_Class,Event_Class_ID,Cameras,Timestamp\n"


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(CSV_HEADER)
        for r in rows:
            fh.write(",".join(r) + "\n")


class TestClassifyEvent:
    @pytest.mark.parametrize(
        "event_class,fire,human,contact",
        [
            ("Empty", False, False, False),
            ("Fire", True, False, False),
            ("Human_Movement_1Human", False, True, False),
            ("Human_Movement_2+Humans", False, True, False),
            ("Human_Movement_2+Fire", True, True, False),
            ("Human_Movement_1Human+Fire", True, True, False),
            ("Contact_2+Humans", False, True, True),
            ("Contact_2+Humans+Fire", True, True, True),
            ("Violence_2+Humans", False, True, True),
            ("Violence_2+Humans+Fire", True, True, True),
            # +fall is out of scope: no dedicated flag, still counts as human.
            ("Human_Movement_1Human+fall", False, True, False),
            ("Human_Movement_2+fall+Fire", True, True, False),
            ("Violence_2+Humans+fall", False, True, True),
        ],
    )
    def test_classification_rules(self, event_class, fire, human, contact):
        result = classify_event(event_class)
        assert result == {"fire": fire, "human": human, "contact": contact}


class TestLoadRoomLabels:
    def test_parses_and_sorts_rows(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [
            ("room-1", "2026-06-30", "12:26:24.000", "12:36:24.000", "Empty", "10", "0|1|2", "2026-07-01T07:31:46.310134+00:00"),
            ("room-1", "2026-06-30", "12:16:24.000", "12:26:24.000", "Human_Movement_1Human", "4", "0|1|2", "2026-07-01T07:31:46.310134+00:00"),
        ])
        intervals = load_room_labels(csv_path)
        assert len(intervals) == 2
        assert intervals[0].event_class == "Human_Movement_1Human"  # earlier start sorts first
        assert intervals[0].start == dt.datetime(2026, 6, 30, 12, 16, 24)
        assert intervals[0].end == dt.datetime(2026, 6, 30, 12, 26, 24)
        assert intervals[0].cameras == (0, 1, 2)

    def test_filters_by_room_and_date(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [
            ("room-1", "2026-06-30", "12:16:24.000", "12:26:24.000", "Empty", "10", "0|1|2", "x"),
            ("room-1", "2026-06-23", "09:00:00.000", "09:10:00.000", "Empty", "10", "0|1|2", "x"),
            ("synth_room_1", "2026-06-30", "12:16:24.000", "12:26:24.000", "Empty", "10", "0|1|2", "x"),
        ])
        intervals = load_room_labels(csv_path, room_id="room-1", date="2026-06-30")
        assert len(intervals) == 1
        assert intervals[0].room_id == "room-1"
        assert intervals[0].date == "2026-06-30"


class TestLabelJoiner:
    def _joiner(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [
            ("room-1", "2026-06-30", "12:16:24.000", "13:07:15.167", "Empty", "10", "0|1|2", "x"),
            ("room-1", "2026-06-30", "13:07:15.250", "13:08:15.667", "Human_Movement_2+Humans", "3", "0|1|2", "x"),
            ("room-1", "2026-06-30", "13:08:15.750", "13:08:26.083", "Contact_2+Humans", "2", "0|1|2", "x"),
        ])
        return LabelJoiner.from_csv(csv_path)

    def test_labels_for_within_first_interval(self, tmp_path):
        joiner = self._joiner(tmp_path)
        ts = dt.datetime(2026, 6, 30, 12, 30, 0).timestamp()
        result = joiner.labels_for(ts)
        assert result == {"event_class": "Empty", "fire": False, "human": False, "contact": False}

    def test_labels_for_within_contact_interval(self, tmp_path):
        joiner = self._joiner(tmp_path)
        ts = dt.datetime(2026, 6, 30, 13, 8, 20).timestamp()
        result = joiner.labels_for(ts)
        assert result["event_class"] == "Contact_2+Humans"
        assert result["contact"] is True
        assert result["human"] is True
        assert result["fire"] is False

    def test_labels_for_none_before_first_interval(self, tmp_path):
        joiner = self._joiner(tmp_path)
        ts = dt.datetime(2026, 6, 30, 12, 0, 0).timestamp()
        assert joiner.labels_for(ts) is None

    def test_labels_for_none_after_last_interval(self, tmp_path):
        joiner = self._joiner(tmp_path)
        ts = dt.datetime(2026, 6, 30, 14, 0, 0).timestamp()
        assert joiner.labels_for(ts) is None

    def test_labels_for_none_in_inter_interval_gap(self, tmp_path):
        joiner = self._joiner(tmp_path)
        # 13:07:15.167 -> 13:07:15.250 is an ~83ms gap between rows 1 and 2.
        ts = dt.datetime(2026, 6, 30, 13, 7, 15, 200000).timestamp()
        assert joiner.labels_for(ts) is None

    def test_exact_start_boundary_is_inclusive(self, tmp_path):
        joiner = self._joiner(tmp_path)
        ts = dt.datetime(2026, 6, 30, 13, 7, 15, 250000).timestamp()
        result = joiner.labels_for(ts)
        assert result["event_class"] == "Human_Movement_2+Humans"

    def test_exact_end_boundary_is_exclusive(self, tmp_path):
        joiner = self._joiner(tmp_path)
        # 13:07:15.167 is the Empty interval's declared End_Time exactly.
        ts = dt.datetime(2026, 6, 30, 13, 7, 15, 167000).timestamp()
        assert joiner.interval_for(ts) is None

    def test_raises_on_empty_intervals(self):
        with pytest.raises(ValueError, match="no label intervals"):
            LabelJoiner([])


class TestAgainstRealRoom1Labels:
    """Cross-check against the actual project data at data/Room_1/room-1,
    if present on this machine (skipped otherwise — this repo doesn't ship
    the data/ folder, it's a sibling directory)."""

    ROOM1_CSV = None  # set below if the real file exists

    @staticmethod
    def _real_csv_path():
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "data" / "Room_1" / "room-1" / "labels.csv"
        return p if p.is_file() else None

    def test_first_row_matches_known_values(self):
        path = self._real_csv_path()
        if path is None:
            pytest.skip("real data/Room_1/room-1/labels.csv not present on this machine")
        intervals = load_room_labels(path, room_id="room-1", date="2026-06-30")
        assert intervals[0].event_class == "Empty"
        assert intervals[0].start == dt.datetime(2026, 6, 30, 12, 16, 24)

    def test_known_contact_frame_classifies_correctly(self):
        path = self._real_csv_path()
        if path is None:
            pytest.skip("real data/Room_1/room-1/labels.csv not present on this machine")
        joiner = LabelJoiner.from_csv(path, room_id="room-1", date="2026-06-30")
        # From the raw CSV: 13:08:15.750-13:08:26.083 is Contact_2+Humans.
        ts = dt.datetime(2026, 6, 30, 13, 8, 20).timestamp()
        result = joiner.labels_for(ts)
        assert result["event_class"] == "Contact_2+Humans"
        assert result["contact"] is True
