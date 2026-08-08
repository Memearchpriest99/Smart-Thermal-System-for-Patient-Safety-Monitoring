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


# Curated, not random. Contact is a rare class overall (6.8% of waveshare_work
# frames) concentrated in only 4 of 17 scenarios; a random session split has
# no awareness of per-task label distribution and can -- confirmed: did, on
# the first attempt -- draw a 3-session held-out set with ZERO
# contact-positive frames, making contact evaluation on it structurally
# meaningless regardless of model quality (precision/recall are 0/0 when
# there is nothing to detect, not a measure of anything). These three each
# contain a genuine mix of contact and non-contact frames (42.7%, 16.8%,
# 22.1% positive respectively -- see data/DATASET_NOTES.md); `3pp_surprise`
# (95.7% positive, an extreme outlier) is deliberately left in training
# instead of test.
CONTACT_TEST_SCENES: frozenset[str] = frozenset({"2ppl_fight", "2ppl_hug", "3pplhedroncolider"})


def build_task_split(
    labeled_sessions: Sequence,
    *,
    task: str,
    fire_seed: int = 0,
    human_seed: int = 1,
) -> tuple[set[str], set[str]]:
    """Independent train/test split per task.

    Per the project owner's clarification: "the same test set" was meant
    PER GROUP of algorithms (fire / human / contact), not one shared split
    reused identically across all three -- fire and human each get their own
    random session-level split (different seeds, so they're genuinely
    decoupled rather than coincidentally identical), and contact gets the
    curated CONTACT_TEST_SCENES rather than a random draw, for the reason
    above.

    Args:
        labeled_sessions: session-like objects exposing `.scene` (as
            required by session_train_test_split).
        task: "fire", "human", or "contact".

    Returns:
        (train_scenes, test_scenes) as sets of scene name strings.
    """
    if task == "contact":
        all_scenes = {s.scene for s in labeled_sessions}
        test_scenes = set(CONTACT_TEST_SCENES) & all_scenes
        missing = set(CONTACT_TEST_SCENES) - all_scenes
        if missing:
            raise ValueError(
                f"CONTACT_TEST_SCENES references scenes not present in this "
                f"dataset: {sorted(missing)}. Check the dataset root or update "
                f"CONTACT_TEST_SCENES."
            )
        train_scenes = all_scenes - test_scenes
        return train_scenes, test_scenes

    if task not in ("fire", "human"):
        raise ValueError(f"task must be 'fire', 'human', or 'contact'; got {task!r}")
    seed = fire_seed if task == "fire" else human_seed
    train_sessions, test_sessions = session_train_test_split(labeled_sessions, seed=seed)
    return {s.scene for s in train_sessions}, {s.scene for s in test_sessions}
