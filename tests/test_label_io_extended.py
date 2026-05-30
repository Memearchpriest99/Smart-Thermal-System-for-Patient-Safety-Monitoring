"""Tests for the new label_io additions: class constants, contact label I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from thermal_algorithms.training.label_io import (
    CONTACT_LABEL_FILENAME,
    FIRE_CLASS_ID,
    PERSON_CLASS_ID,
    has_contact_labels,
    load_contact_labels,
    save_contact_labels,
)


class TestClassConstants:
    def test_fire_class_id_is_zero(self):
        assert FIRE_CLASS_ID == 0

    def test_person_class_id_is_one(self):
        assert PERSON_CLASS_ID == 1

    def test_fire_and_person_are_distinct(self):
        assert FIRE_CLASS_ID != PERSON_CLASS_ID

    def test_contact_label_filename_is_csv(self):
        assert CONTACT_LABEL_FILENAME.endswith(".csv")


class TestLoadContactLabels:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_contact_labels(tmp_path / "nope.csv")
        assert result == {}

    def test_basic_round_trip(self, tmp_path):
        labels = {0: 0, 1: 0, 5: 1, 6: 1, 10: 0}
        p = tmp_path / "contact_labels.csv"
        save_contact_labels(labels, p)
        loaded = load_contact_labels(p)
        assert loaded == labels

    def test_only_contact_frames(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text("frame_idx,contact\n3,1\n7,1\n")
        result = load_contact_labels(p)
        assert result == {3: 1, 7: 1}

    def test_only_non_contact_frames(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text("frame_idx,contact\n0,0\n1,0\n2,0\n")
        result = load_contact_labels(p)
        assert result == {0: 0, 1: 0, 2: 0}

    def test_comments_ignored(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text("# comment\nframe_idx,contact\n# another\n0,1\n1,0\n")
        result = load_contact_labels(p)
        assert result == {0: 1, 1: 0}

    def test_header_row_skipped(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text("frame_idx,contact\n0,1\n")
        result = load_contact_labels(p)
        assert "frame_idx" not in result
        assert 0 in result

    def test_invalid_contact_value_raises(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text("frame_idx,contact\n0,2\n")
        with pytest.raises(ValueError, match="0 or 1"):
            load_contact_labels(p)

    def test_wrong_column_count_raises(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text("frame_idx,contact,extra\n0,1,x\n")
        with pytest.raises(ValueError, match="2 columns"):
            load_contact_labels(p)

    def test_non_integer_frame_idx_raises(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text("frame_idx,contact\nabc,1\n")
        with pytest.raises(ValueError):
            load_contact_labels(p)

    def test_large_dataset(self, tmp_path):
        labels = {i: (i % 3 == 0) for i in range(500)}
        p = tmp_path / "c.csv"
        save_contact_labels(labels, p)
        loaded = load_contact_labels(p)
        assert loaded == labels

    def test_saved_file_is_sorted_by_frame_idx(self, tmp_path):
        labels = {10: 1, 0: 0, 5: 1, 3: 0}
        p = tmp_path / "c.csv"
        save_contact_labels(labels, p)
        lines = [l for l in p.read_text().splitlines() if l and not l.startswith("frame")]
        indices = [int(l.split(",")[0]) for l in lines]
        assert indices == sorted(indices)


class TestHasContactLabels:
    def test_returns_true_when_file_exists(self, tmp_path):
        (tmp_path / CONTACT_LABEL_FILENAME).write_text("frame_idx,contact\n")
        assert has_contact_labels(tmp_path) is True

    def test_returns_false_when_missing(self, tmp_path):
        assert has_contact_labels(tmp_path) is False
