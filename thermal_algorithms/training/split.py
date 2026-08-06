"""Session-level train/test split for real data sources (room-1, waveshare).

Splitting whole sessions (not individual frames) into train/test avoids
leakage between near-duplicate consecutive frames within a session — frames
a few indices apart are visually almost identical, so a frame-level random
split would let the model effectively memorize near-copies of its own test
set. Synthetic data is used wholesale for training per task 3's instruction;
this module is only meant for the real sources.
"""

from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def session_train_test_split(
    sessions: Sequence[T],
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
    min_test: int = 1,
) -> tuple[list[T], list[T]]:
    """Split whole sessions into (train, test), deterministic given ``seed``.

    Args:
        sessions: session-like objects exposing ``.scene`` (``SessionMetadata``
            and ``HDF5Session`` both do) — sorted by that before shuffling, so
            the split doesn't depend on filesystem iteration order.
        test_fraction: fraction of sessions to hold out, rounded to the
            nearest whole session (at least ``min_test``, capped so at least
            one session remains for training).
        seed: shuffle seed, for a reproducible split across runs.
        min_test: minimum number of sessions to hold out when there are
            enough to do so.

    Returns:
        ``(train, test)``. If given 0 or 1 sessions, everything goes to
        ``train`` and ``test`` is empty — there's nothing to meaningfully
        hold out. Callers should check ``len(test) == 0`` and treat that as
        "this source currently can't validate generalization on its own",
        not silently proceed as if a split happened.
    """
    ordered = sorted(sessions, key=lambda s: s.scene)
    n = len(ordered)
    if n <= 1:
        return list(ordered), []

    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)

    n_test = max(min_test, round(n * test_fraction))
    n_test = min(n_test, n - 1)  # always keep at least one session for train
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, test
