"""Class-balanced (50/50) training-set construction.

The natural corpus is imbalanced in different directions per task (fire
~17% positive, contact ~2.4% positive, human ~90% positive -- see
data/DATASET_NOTES.md) and every full-corpus training run to date has
trained on that natural ratio. This module builds genuinely balanced
TRAINING inputs (sample-level resampling, not just sklearn's
``class_weight="balanced"`` loss-reweighting) for the six detectors chosen
for the balanced-training pass -- see the plan this was built against for
the full rationale.

Two distinct regimes, because one class of detector trains on independent
frames and the other trains on temporal sliding windows:

Group A -- frame-independent (OtsuFireDetector, AdaptiveThresholdDetector,
FireSVMDetector, HOGSVMDetector, MobileNetSSDDetector): each example is one
frame, so balancing is just "count labels, subsample the majority class to
match the minority". ``balance_examples`` + ``build_balanced_fire_pool`` /
``build_balanced_human_pool`` cover this.

Group B -- window-dependent (ThermoX3DDetector only): ``fit()`` builds
T-frame sliding windows over whatever contiguous ``examples`` list it's
given, labelling each window by its LAST frame
(``ThermoX3DDetector._build_training_windows``). Deleting individual
negative frames to rebalance would corrupt window contiguity. Because
contact-positive frames are rare and cluster into bursts, this module
curates whole contiguous RUNS instead: every positive-containing burst (with
leading context padding) up to a target volume, plus matching-volume
negative-only runs sampled from elsewhere in the corpus.

Four functions, deliberately kept separate rather than one combined
generator (as an earlier version of this module had), so that a train/val
split can happen at the PARENT-RUN level, before any run gets subdivided
into smaller sub-chunks -- splitting after subdivision risks two sub-chunks
of the same physical stretch landing on opposite sides of the split, the
same near-duplicate-across-split leakage failure mode ``training/split.py``
already documents having been burned by once elsewhere in this project:

1. ``build_balanced_contact_pools`` -- Pass 1: collect ``pos_items`` (whole
   padded positive-burst items -- never sub-chunked) and ``raw_neg_runs``
   (uncapped ``(source, start, stop, total_len)`` tuples -- deliberately
   NOT yet sub-chunked).
2. ``split_contact_pools_train_val`` -- split both pools, at this
   pre-capping granularity, into train/val.
3. ``size_and_cap_negative_runs`` -- sub-chunk ONE side's ``raw_neg_runs``
   (train or val) to a target volume, sized to roughly match the positive
   pool's own chunk granularity (matters for the training side only; the
   val side just wants a bounded total volume).
4. ``interleave_chunks`` -- order a train-side pos/neg item stream so no
   more than ``max_ratio`` consecutive items are the same class (while both
   pools still have items left).

For the synthetic corpus, the positive/negative run scan is done via a
CHEAP pre-pass (``_session_contact_label_array``): synthetic ``.h5`` chunk
headers carry per-frame timestamps that can be read without touching the
(large) ``frames`` dataset (``hdf5_source._peek_chunk_header``), and
``LabelJoiner.labels_for`` is an in-memory bisect lookup -- so an entire
session's contact-positive/negative structure can be mapped out without
decoding a single pixel. Only the frames actually selected into a run are
decoded (via ``HDF5CameraSession.load_frame``).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from thermal_algorithms.core.types import ContactEvent, FireLevel
from thermal_algorithms.training.datasets import (
    ContactFrameDataset,
    DatasetIndex,
    FireFrameDataset,
    FrameLevelDataset,
)
from thermal_algorithms.training.full_corpus import (
    iter_synth_sessions,
    stream_fire_examples_subsampled,
)
from thermal_algorithms.training.hdf5_source import _peek_chunk_header, list_chunks
from thermal_algorithms.training.label_io import PERSON_CLASS_ID

# ---------------------------------------------------------------------------
# Group A -- frame-independent balancing
# ---------------------------------------------------------------------------


def balance_examples(
    examples: list,
    is_positive: Callable[[object], bool],
    *,
    max_total: Optional[int] = None,
    seed: int = 0,
) -> tuple[list, dict]:
    """Undersample the majority class to match the minority, optionally
    capped at ``max_total`` (split evenly). Returns (balanced_list, counts)
    -- counts always report what was actually AVAILABLE and USED, since the
    cap doesn't always land exactly even and callers should be able to tell
    the difference between "balanced at the requested size" and "balanced,
    but smaller than requested because one class ran out"."""
    rng = random.Random(seed)
    pos = [e for e in examples if is_positive(e)]
    neg = [e for e in examples if not is_positive(e)]
    n_pos_avail, n_neg_avail = len(pos), len(neg)
    n_each = min(n_pos_avail, n_neg_avail)
    if max_total is not None:
        n_each = min(n_each, max_total // 2)
    if n_each == 0:
        raise ValueError(
            f"cannot build a balanced pool: {n_pos_avail} positive / {n_neg_avail} "
            f"negative examples available (need >=1 of each)."
        )
    pos_sample = rng.sample(pos, n_each)
    neg_sample = rng.sample(neg, n_each)
    combined = pos_sample + neg_sample
    rng.shuffle(combined)
    counts = {
        "n_pos_available": n_pos_avail,
        "n_neg_available": n_neg_avail,
        "n_used_each": n_each,
        "n_total": len(combined),
    }
    return combined, counts


def build_balanced_fire_pool(
    data_root: str | Path,
    waveshare_index: DatasetIndex,
    train_scenes: set[str],
    *,
    synth_per_camera_samples: int = 1500,
    max_total: Optional[int] = None,
    seed: int = 0,
) -> tuple[list, dict]:
    """Balanced (Frame, FireAlert) pool: waveshare fire-train scenes in full
    + a bounded synth draw, mirroring exactly what FireSVMDetector's natural
    -ratio full-corpus training already samples
    (``stream_fire_examples_subsampled`` yields waveshare in full, then the
    bounded per-camera synth draw) -- then balanced on top."""
    examples = list(stream_fire_examples_subsampled(
        data_root, waveshare_index, per_camera_samples=synth_per_camera_samples,
        seed=seed, scenes=train_scenes,
    ))
    return balance_examples(
        examples, lambda ex: ex[1].level != FireLevel.SAFE, max_total=max_total, seed=seed,
    )


def build_balanced_human_pool(
    waveshare_index: DatasetIndex,
    train_scenes: set[str],
    *,
    exclude_scenes: Optional[set[str]] = None,
    max_total: Optional[int] = None,
    seed: int = 0,
) -> tuple[list, dict]:
    """Balanced (Frame, list[Detection]) pool from waveshare_work only (synth
    has no bounding boxes, ever). Human presence is the MAJORITY class here
    (waveshare is ~90% person-positive, only 838 negative frames total across
    all 17 scenes) -- so this pool is capped by the scarce negative class and
    will be much smaller than the natural-ratio training set. That's a real,
    disclosed trade-off (proper balance, less data), not a bug.

    exclude_scenes: pass any scene ALSO used as an extra held-out negative
    test scene (e.g. 'empty_room' via --extra-human-test-scenes elsewhere in
    this project) -- otherwise those frames end up in both the balanced
    training pool and the "corrected" held-out eval, which is train/test
    leakage. 'empty_room' is nominally in human_train_scenes (per
    build_task_split) but train_human() in train_full_corpus.py never
    actually trains on it (include_negative_frames=False there) -- THIS
    function defaults to include_negative_frames=True, so unlike the
    natural-ratio pipeline, an excluded scene here really would otherwise
    leak."""
    scenes = train_scenes - exclude_scenes if exclude_scenes else train_scenes
    examples = list(FrameLevelDataset(
        waveshare_index, scenes=scenes, class_filter=[PERSON_CLASS_ID],
        include_negative_frames=True,
    ))
    return balance_examples(
        examples, lambda ex: bool(ex[1]), max_total=max_total, seed=seed,
    )


# ---------------------------------------------------------------------------
# Group B -- window-dependent (contact) run curation
# ---------------------------------------------------------------------------


def _find_runs(is_true: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True-runs in a boolean array, as half-open (start, stop)
    index pairs."""
    runs = []
    n = len(is_true)
    i = 0
    while i < n:
        if is_true[i]:
            j = i
            while j < n and is_true[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _session_contact_label_array(session) -> tuple[np.ndarray, np.ndarray]:
    """Cheap (zero pixel decode) per-frame contact-positive boolean array
    for one synth HDF5Session's cam_0, built purely from header-peeked
    timestamps (``hdf5_source._peek_chunk_header`` reads only the tiny
    ``timestamps`` dataset, never the large ``frames`` one) joined against
    the session's own ``LabelJoiner`` (an in-memory bisect lookup -- no
    disk I/O per call). Returns (timestamps, is_contact)."""
    cam0_dir = session.ref.cam_dir(0)
    ts_parts = []
    for p in list_chunks(cam0_dir):
        _n, ts = _peek_chunk_header(p)
        ts_parts.append(ts)
    timestamps = np.concatenate(ts_parts) if ts_parts else np.zeros(0, dtype=np.float64)
    is_contact = np.zeros(len(timestamps), dtype=bool)
    for i, t in enumerate(timestamps):
        labels = session._joiner.labels_for(float(t))
        if labels is not None and labels["contact"]:
            is_contact[i] = True
    return timestamps, is_contact


def _cap_run_length(start: int, stop: int, max_len: int) -> Iterator[tuple[int, int]]:
    """Split a long run into <=max_len sub-runs (bounds per-chunk memory)."""
    i = start
    while i < stop:
        yield i, min(i + max_len, stop)
        i += max_len


def _pad_run_to_min_length(lo: int, hi: int, min_len: int, total_len: int) -> Optional[tuple[int, int]]:
    """Widen [lo, hi) to at least min_len frames -- ThermoX3DDetector.fit()
    raises if a chunk has fewer than T frames (needs >=1 full sliding
    window), which a positive burst or a capped negative sub-run near a
    session boundary can otherwise produce. Extends forward first (doesn't
    shift which frame ends up labelling any window it wasn't already going
    to label), then backward, staying in [0, total_len). Returns None if
    total_len itself can't satisfy min_len (should not happen for any real
    session, but skip rather than crash if it ever does)."""
    if total_len < min_len:
        return None
    length = hi - lo
    if length >= min_len:
        return lo, hi
    deficit = min_len - length
    extend_hi = min(deficit, total_len - hi)
    hi += extend_hi
    deficit -= extend_hi
    if deficit > 0:
        lo = max(0, lo - deficit)
    return lo, hi


def _decode_synth_run(session, lo: int, hi: int) -> tuple[list, list]:
    """Decode one [lo, hi) frame range from a synth session's 3 cameras --
    module-level (not a per-session closure) so it can't fall into the
    late-binding-in-a-loop hazard, and to keep it independently testable."""
    frames_chunk, events_chunk = [], []
    for i in range(lo, hi):
        frames = tuple(session._cams[c].load_frame(i) for c in (0, 1, 2))
        labels = session._joiner.labels_for(frames[0].timestamp)
        event = ContactEvent(
            actors=(),
            pairs_in_contact=((0, 1),) if (labels and labels["contact"]) else (),
            timestamp=frames[0].timestamp, confidence=1.0,
        )
        frames_chunk.append(frames)
        events_chunk.append(event)
    return frames_chunk, events_chunk


def build_balanced_contact_pools(
    data_root: str | Path,
    waveshare_index: DatasetIndex,
    train_scenes: set[str],
    *,
    target_positive_frames: int,
    target_negative_frames: int,
    T: int = 5,
    context_pad: Optional[int] = None,
    max_negative_frames_per_session: Optional[int] = None,
    seed: int = 0,
) -> tuple[list[tuple[int, Callable[[], tuple]]], list[tuple]]:
    """Pass 1: collect candidate runs up to the requested targets, via the
    cheap header-peek scan for synth (``_session_contact_label_array`` --
    zero pixel decode). Synth runs are stored as un-decoded (session, lo, hi)
    references; waveshare runs are stored decoded (that source is small
    enough to already be fully materialized).

    Returns ``(pos_items, raw_neg_runs)``:

    - ``pos_items``: ``(n_frames, thunk)`` pairs, one per positive-containing
      burst (+ leading context padding) -- already parent-run-level, never
      sub-chunked, so it's safe to split directly at this granularity.
    - ``raw_neg_runs``: ``(source, start, stop, total_len)`` tuples --
      deliberately left UNCAPPED (no ``_cap_run_length`` applied yet). Each
      synth session contributes at most ``max_negative_frames_per_session``
      frames (default: 1/4 of the target) so negatives aren't drawn almost
      entirely from whichever single session happens to be scanned first --
      real synth sessions run ~1.4M frames each, so one session's negative
      supply alone can trivially satisfy a 100k-frame target. ``source`` is
      either the materialized waveshare example list, or a synth
      ``HDF5Session`` -- dispatch on ``isinstance(source, list)``.

    Call ``split_contact_pools_train_val`` on the result BEFORE calling
    ``size_and_cap_negative_runs`` -- splitting after sub-chunking risks two
    sub-chunks of the same physical stretch landing on opposite sides.

    Stops once BOTH targets are met, or the corpus (waveshare train scenes +
    every synth session) is exhausted -- whichever comes first.
    """
    context_pad = context_pad if context_pad is not None else T - 1
    max_neg_per_session = (
        max_negative_frames_per_session if max_negative_frames_per_session is not None
        else max(5000, target_negative_frames // 4)
    )
    rng = random.Random(seed)
    pos_yielded = 0
    # Each item: (n_frames, thunk) where thunk() -> (source_label, frames_chunk, events_chunk).
    pos_items: list[tuple[int, Callable[[], tuple]]] = []
    # Entries: (source, start, stop, total_len) where source is the
    # materialized waveshare example list, or a synth HDF5Session.
    raw_neg_runs: list[tuple] = []
    raw_neg_volume = 0

    def _synth_thunk(session, lo: int, hi: int):
        return lambda: (session.scene,) + _decode_synth_run(session, lo, hi)

    # Waveshare: small enough to materialize directly.
    ws_examples = list(ContactFrameDataset(waveshare_index, scenes=train_scenes))
    n_ws = len(ws_examples)
    if ws_examples:
        ws_is_pos = np.array([e[1].any_contact for e in ws_examples])
        for start, stop in _find_runs(ws_is_pos):
            if pos_yielded >= target_positive_frames:
                break
            lo = max(0, start - context_pad)
            padded = _pad_run_to_min_length(lo, stop, T, n_ws)
            if padded is None:
                continue
            lo, hi = padded
            run = ws_examples[lo:hi]
            pos_items.append((stop - start, _ws_thunk(run)))
            pos_yielded += (stop - start)

        ws_neg_bound = 0
        for start, stop in _find_runs(~ws_is_pos):
            if ws_neg_bound >= max_neg_per_session:
                break
            if stop - start < T:
                # Too short to ever produce a same-run T-frame window (see
                # size_and_cap_negative_runs' docstring) -- skip rather than
                # let it occupy a train/val split slot that can only ever
                # resolve to zero usable windows.
                continue
            take = min(stop - start, max_neg_per_session - ws_neg_bound)
            if take < T:
                continue
            raw_neg_runs.append((ws_examples, start, start + take, n_ws))
            ws_neg_bound += take
            raw_neg_volume += take

    # Synth: cheap header-peek scan per session, pixel decode deferred to build time.
    if pos_yielded < target_positive_frames or raw_neg_volume < target_negative_frames:
        for session in iter_synth_sessions(data_root):
            if pos_yielded >= target_positive_frames and raw_neg_volume >= target_negative_frames:
                break
            if not all(c in session._cams for c in (0, 1, 2)):
                continue
            timestamps, is_pos = _session_contact_label_array(session)
            if len(timestamps) == 0:
                continue
            # Use the SHORTEST camera's frame count, not just cam_0's -- a
            # truncated trailing chunk on one camera (the kind of acquisition
            # artifact documented elsewhere in this corpus) would otherwise
            # let a run's decode range run past that camera's real length.
            n_session = min(len(timestamps), *(session._cams[c].n_frames for c in (0, 1, 2)))

            if pos_yielded < target_positive_frames:
                for start, stop in _find_runs(is_pos):
                    if pos_yielded >= target_positive_frames:
                        break
                    lo = max(0, start - context_pad)
                    padded = _pad_run_to_min_length(lo, stop, T, n_session)
                    if padded is None:
                        continue
                    lo, hi = padded
                    pos_items.append((stop - start, _synth_thunk(session, lo, hi)))
                    pos_yielded += (stop - start)

            if raw_neg_volume < target_negative_frames:
                session_neg_bound = 0
                neg_runs = _find_runs(~is_pos)
                rng.shuffle(neg_runs)
                for start, stop in neg_runs:
                    if session_neg_bound >= max_neg_per_session:
                        break
                    if stop - start < T:
                        continue
                    take = min(stop - start, max_neg_per_session - session_neg_bound)
                    if take < T:
                        continue
                    raw_neg_runs.append((session, start, start + take, n_session))
                    session_neg_bound += take
                    raw_neg_volume += take

    return pos_items, raw_neg_runs


def _ws_thunk(run: list):
    return lambda: ("waveshare_train", [e[0] for e in run], [e[1] for e in run])


def split_contact_pools_train_val(
    pos_items: list[tuple[int, Callable[[], tuple]]],
    raw_neg_runs: list[tuple],
    *,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[list, list, list, list]:
    """Splits both pools independently, at the PARENT-RUN granularity
    ``build_balanced_contact_pools`` returns (before any sub-chunking) --
    returns ``(train_pos, val_pos, train_neg_runs, val_neg_runs)``. Splitting
    here, rather than after ``size_and_cap_negative_runs`` sub-divides a run,
    is what prevents two sub-chunks of the same physical stretch from landing
    on opposite sides of the split."""
    rng = random.Random(seed)
    pos = list(pos_items)
    rng.shuffle(pos)
    n_val_pos = max(1, round(len(pos) * val_fraction)) if pos else 0
    val_pos, train_pos = pos[:n_val_pos], pos[n_val_pos:]

    neg = list(raw_neg_runs)
    rng.shuffle(neg)
    n_val_neg = max(1, round(len(neg) * val_fraction)) if neg else 0
    val_neg_runs, train_neg_runs = neg[:n_val_neg], neg[n_val_neg:]

    return train_pos, val_pos, train_neg_runs, val_neg_runs


def size_and_cap_negative_runs(
    raw_neg_runs: list[tuple],
    pos_items: list[tuple[int, Callable[[], tuple]]],
    *,
    T: int,
    target_negative_frames: int,
    max_run_frames: int = 5000,
) -> list[tuple[int, Callable[[], tuple]]]:
    """Sub-chunks ``raw_neg_runs`` (one side of a train/val split -- call
    this separately for the train-side and val-side lists) to a target
    volume, sized to roughly match ``pos_items``'s own chunk granularity --
    this is what keeps a downstream interleave from degenerating into
    "alternate for a while, then one giant same-class tail" once the smaller
    pool runs out. Returns ``(n_frames, thunk)`` items, same shape as
    ``pos_items``.

    Padding a short sub-chunk up to T frames is bounded by the PARENT run's
    own ``(start, stop)``, never by the whole session's ``total_len`` --
    unlike a positive run (where reaching into real preceding timeline
    frames for context is the whole point), a negative run's neighbours are
    unknown territory: ``raw_neg_runs`` can legitimately contain a
    razor-thin gap (e.g. a single negative frame sandwiched between two
    contact bursts), and padding that outward using the session bound would
    silently pull in mostly-positive frames -- producing an item this
    function reports as "negative" whose windows are actually labelled
    positive by ``_build_training_windows`` (last-frame labelling). Runs
    that can't reach T frames within their OWN bounds are dropped instead
    (``_pad_run_to_min_length`` returns None when its total_len argument --
    here the parent run's own length -- is under T)."""
    total_pos_frames = sum(p[0] for p in pos_items)
    avg_pos_chunk = max(T, total_pos_frames // len(pos_items)) if pos_items else max_run_frames
    neg_chunk_cap = min(max_run_frames, avg_pos_chunk)

    neg_items: list[tuple[int, Callable[[], tuple]]] = []
    neg_yielded = 0
    for source, start, stop, _total_len in raw_neg_runs:
        if neg_yielded >= target_negative_frames:
            break
        for sub_start, sub_stop in _cap_run_length(start, stop, neg_chunk_cap):
            if neg_yielded >= target_negative_frames:
                break
            padded = _pad_run_to_min_length(sub_start - start, sub_stop - start, T, stop - start)
            if padded is None:
                continue
            lo, hi = padded[0] + start, padded[1] + start
            if isinstance(source, list):
                run = source[lo:hi]
                neg_items.append((sub_stop - sub_start, _ws_thunk(run)))
            else:
                neg_items.append((sub_stop - sub_start, lambda s=source, lo=lo, hi=hi:
                                   (s.scene,) + _decode_synth_run(s, lo, hi)))
            neg_yielded += (sub_stop - sub_start)
    return neg_items


def interleave_chunks(
    pos_items: list[tuple[int, Callable[[], tuple]]],
    neg_items: list[tuple[int, Callable[[], tuple]]],
    *,
    max_ratio: int = 3,
    seed: int = 0,
) -> list[tuple[str, tuple[int, Callable[[], tuple]]]]:
    """Orders a train-side pos/neg item stream as ``(class_label, item)``
    pairs so no more than ``max_ratio`` consecutive items are the same class
    WHILE BOTH pools still have items left -- once one pool is exhausted,
    the remaining tail is unavoidably one class (nothing left to alternate
    with); that's an honest limitation, not silently hidden. This matters
    because ThermoX3DDetector's fit() persists its optimizer across calls
    (see thermo_x3d.py's _get_optimizer) but the model still shouldn't see
    only one class for too long a stretch -- feeding it all positive runs
    then all negative runs in sequence (the original, buggy design) caused
    catastrophic forgetting even before the persistent-optimizer fix."""
    rng = random.Random(seed)
    pos = list(pos_items)
    rng.shuffle(pos)
    neg = list(neg_items)
    rng.shuffle(neg)

    out: list[tuple[str, tuple]] = []
    i = j = 0
    streak_cls, streak_n = None, 0
    while i < len(pos) or j < len(neg):
        can_pos = i < len(pos) and not (streak_cls == "pos" and streak_n >= max_ratio and j < len(neg))
        can_neg = j < len(neg) and not (streak_cls == "neg" and streak_n >= max_ratio and i < len(pos))
        if can_pos and (not can_neg or i <= j):
            out.append(("pos", pos[i]))
            streak_n = streak_n + 1 if streak_cls == "pos" else 1
            streak_cls = "pos"
            i += 1
        else:
            out.append(("neg", neg[j]))
            streak_n = streak_n + 1 if streak_cls == "neg" else 1
            streak_cls = "neg"
            j += 1
    return out
