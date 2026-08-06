"""Tests for session-level train/test splitting."""

from dataclasses import dataclass

import pytest

from thermal_algorithms.training.split import session_train_test_split


@dataclass(frozen=True)
class _FakeSession:
    scene: str


def _sessions(*names):
    return [_FakeSession(scene=n) for n in names]


class TestSessionTrainTestSplit:
    def test_empty_input(self):
        train, test = session_train_test_split([])
        assert train == []
        assert test == []

    def test_single_session_goes_entirely_to_train(self):
        sessions = _sessions("only_one")
        train, test = session_train_test_split(sessions)
        assert len(train) == 1
        assert test == []

    def test_splits_by_whole_session_not_frame(self):
        sessions = _sessions(*[f"scene_{i}" for i in range(10)])
        train, test = session_train_test_split(sessions, test_fraction=0.2, seed=0)
        assert len(train) + len(test) == 10
        assert len(test) == 2  # round(10 * 0.2)
        # No overlap, every session accounted for exactly once.
        assert set(s.scene for s in train) & set(s.scene for s in test) == set()
        assert set(s.scene for s in train) | set(s.scene for s in test) == {
            f"scene_{i}" for i in range(10)
        }

    def test_deterministic_given_same_seed(self):
        sessions = _sessions(*[f"scene_{i}" for i in range(17)])
        train1, test1 = session_train_test_split(sessions, seed=42)
        train2, test2 = session_train_test_split(sessions, seed=42)
        assert [s.scene for s in train1] == [s.scene for s in train2]
        assert [s.scene for s in test1] == [s.scene for s in test2]

    def test_independent_of_input_order(self):
        sessions_a = _sessions(*[f"scene_{i}" for i in range(10)])
        sessions_b = list(reversed(sessions_a))
        train_a, test_a = session_train_test_split(sessions_a, seed=7)
        train_b, test_b = session_train_test_split(sessions_b, seed=7)
        assert {s.scene for s in test_a} == {s.scene for s in test_b}

    def test_min_test_respected(self):
        sessions = _sessions(*[f"scene_{i}" for i in range(20)])
        train, test = session_train_test_split(sessions, test_fraction=0.01, min_test=3)
        assert len(test) == 3

    def test_never_empties_train(self):
        sessions = _sessions("a", "b")
        train, test = session_train_test_split(sessions, test_fraction=0.9, min_test=5)
        assert len(train) >= 1
        assert len(train) + len(test) == 2

    def test_two_sessions_split_one_one(self):
        sessions = _sessions("a", "b")
        train, test = session_train_test_split(sessions, test_fraction=0.5, seed=0)
        assert len(train) == 1
        assert len(test) == 1
