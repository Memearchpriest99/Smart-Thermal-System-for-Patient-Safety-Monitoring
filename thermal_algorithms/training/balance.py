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
contact-positive frames are rare and cluster into bursts,
``iter_balanced_contact_runs`` instead curates whole contiguous RUNS: every
positive-containing burst (with leading context padding) up to a target
volume, plus matching-volume negative-only runs sampled from elsewhere in
the corpus. Each run is yielded as its own chunk, in the same
``(source_label, frames_chunk, events_chunk)`` shape
``contact_chunks()``/``iter_contact_training_chunks()`` already use, so it
drops straight into the existing "one model instance, one fit() call per
chunk, weights persist across calls" pattern.

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
    max_total: Optional[int] = None,
    seed: int = 0,
) -> tuple[list, dict]:
    """Balanced (Frame, list[Detection]) pool from waveshare_work only (synth
    has no bounding boxes, ever). Human presence is the MAJORITY class here
    (waveshare is ~90% person-positive, only 838 negative frames total across
    all 17 scenes) -- so this pool is capped by the scarce negative class and
    will be much smaller than the natural-ratio training set. That's a real,
    disclosed trade-off (proper balance, less data), not a bug."""
    examples = list(FrameLevelDataset(
        waveshare_index, scenes=train_scenes, class_filter=[PERSON_CLASS_ID],
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


def iter_balanced_contact_runs(
    data_root: str | Path,
    waveshare_index: DatasetIndex,
    train_scenes: set[str],
    *,
    target_positive_frames: int,
    target_negative_frames: int,
    T: int = 16,
    context_pad: Optional[int] = None,
    max_run_frames: int = 5000,
    seed: int = 0,
) -> Iterator[tuple[str, list, list]]:
    """Yields (source_label, frames_chunk, events_chunk) -- same contract as
    ``contact_chunks()``/``iter_contact_training_chunks()`` -- but each chunk
    is a whole contiguous run (a contact-positive burst + leading context, or
    a negative-only stretch), curated so the resulting sliding windows
    (labelled by their LAST frame, see module docstring) land close to 50/50,
    rather than a frame-level resample that would break window contiguity.

    Stops once BOTH targets are met, or the corpus (waveshare train scenes +
    every synth session) is exhausted -- whichever comes first. Positive
    frames are collected from waveshare first (real, but a tiny pool -- only
    ~150-190 positive frames total), then synth (via the cheap header-peek
    scan) until the target is reached. Negative runs are drawn from the same
    per-session scan, so scanning a session once yields candidates for both.
    """
    context_pad = context_pad if context_pad is not None else T - 1
    rng = random.Random(seed)
    pos_yielded = 0
    neg_yielded = 0

    # ---- Waveshare: small enough to materialize directly ----
    ws_examples = list(ContactFrameDataset(waveshare_index, scenes=train_scenes))
    if ws_examples:
        ws_is_pos = np.array([e[1].any_contact for e in ws_examples])
        for start, stop in _find_runs(ws_is_pos):
            if pos_yielded >= target_positive_frames:
                break
            lo = max(0, start - context_pad)
            run = ws_examples[lo:stop]
            yield "waveshare_train", [e[0] for e in run], [e[1] for e in run]
            pos_yielded += (stop - start)
        for start, stop in _find_runs(~ws_is_pos):
            if neg_yielded >= target_negative_frames:
                break
            for sub_start, sub_stop in _cap_run_length(start, stop, max_run_frames):
                if neg_yielded >= target_negative_frames:
                    break
                run = ws_examples[sub_start:sub_stop]
                yield "waveshare_train", [e[0] for e in run], [e[1] for e in run]
                neg_yielded += (sub_stop - sub_start)

    # ---- Synth: cheap header-peek scan per session, decode only chosen runs ----
    if pos_yielded < target_positive_frames or neg_yielded < target_negative_frames:
        for session in iter_synth_sessions(data_root):
            if pos_yielded >= target_positive_frames and neg_yielded >= target_negative_frames:
                break
            if not all(c in session._cams for c in (0, 1, 2)):
                continue
            timestamps, is_pos = _session_contact_label_array(session)
            if len(timestamps) == 0:
                continue

            def _decode_run(lo: int, hi: int) -> tuple[list, list]:
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

            if pos_yielded < target_positive_frames:
                for start, stop in _find_runs(is_pos):
                    if pos_yielded >= target_positive_frames:
                        break
                    lo = max(0, start - context_pad)
                    frames_chunk, events_chunk = _decode_run(lo, stop)
                    yield session.scene, frames_chunk, events_chunk
                    pos_yielded += (stop - start)

            if neg_yielded < target_negative_frames:
                neg_runs = _find_runs(~is_pos)
                rng.shuffle(neg_runs)
                for start, stop in neg_runs:
                    if neg_yielded >= target_negative_frames:
                        break
                    for sub_start, sub_stop in _cap_run_length(start, stop, max_run_frames):
                        if neg_yielded >= target_negative_frames:
                            break
                        frames_chunk, events_chunk = _decode_run(sub_start, sub_stop)
                        yield session.scene, frames_chunk, events_chunk
                        neg_yielded += (sub_stop - sub_start)
