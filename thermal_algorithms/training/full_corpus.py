"""Stream the FULL waveshare + synthetic corpus for Phase 3/4 training scripts.

The synthetic corpus is ~31M frame-instances (10.36M unique frames x 3 cams,
across 5 rooms x 1-2 dates each) — every function here yields, never
materializes a list of frames, since even a fraction of that would exceed
available memory (a single 62x80 float32 frame is ~20KB; 31M of them is
~600 GB).

Scope note (per `data/DATASET_NOTES.md`): synthetic data has NO bounding-box
ground truth, ever — only frame-level presence labels. It can feed
`FireSVMDetector` (frame-level fire presence) and the contact detectors
(frame-level contact presence), but NOT `HOGSVMDetector`/
`MobileNetSSDDetector` (need real bboxes, which only `waveshare_work` has).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

from thermal_algorithms.core.types import ContactEvent, Frame, FireAlert
from thermal_algorithms.training.datasets import ContactFrameDataset, DatasetIndex, FireFrameDataset
from thermal_algorithms.training.multi_source import HDF5Session, discover_hdf5_sessions

SYNTH_ROOMS: tuple[str, ...] = (
    "synth_room_1", "synth_room_2", "synth_room_3", "synth_room_4", "synth_room_5",
)


def iter_synth_sessions(data_root: str | Path) -> Iterator[HDF5Session]:
    """One `HDF5Session` per (room, date) — lightweight refs, not loaded data."""
    data_root = Path(data_root)
    for room in SYNTH_ROOMS:
        room_root = data_root / room
        labels_csv = room_root / "labels.csv"
        if not labels_csv.is_file():
            continue
        for ref in discover_hdf5_sessions(room_root, room):
            yield HDF5Session(ref, labels_csv)


def stream_fire_examples(
    data_root: str | Path, waveshare_index: DatasetIndex,
) -> Iterator[tuple[Frame, FireAlert]]:
    """(Frame, FireAlert) pairs: real-bbox waveshare examples first, then every
    synthetic session's presence-only examples. A single flat generator —
    `FireSVMDetector.fit()` streams its `X`/`y` args without materializing
    them (only the small derived feature vectors accumulate), so this can be
    passed straight through."""
    yield from FireFrameDataset(waveshare_index)
    for session in iter_synth_sessions(data_root):
        yield from session.fire_examples()


def stream_fire_examples_subsampled(
    data_root: str | Path,
    waveshare_index: DatasetIndex,
    *,
    per_camera_samples: int,
    seed: int = 0,
    scenes: "set[str] | None" = None,
) -> Iterator[tuple[Frame, FireAlert]]:
    """Like stream_fire_examples, but random-samples at most
    `per_camera_samples` frames per camera per synthetic session (via
    HDF5Session.sample_fire_examples_random -- direct index access, not a
    full scan) instead of visiting the entire ~31M-frame corpus.

    All of waveshare_work's real-bbox examples are kept in full (it's a few
    thousand frames, not the problem, and it's real annotated data).
    FireSVMDetector's sklearn SVC is more-than-quadratic in sample count and
    explicitly unsuited to datasets much past "a couple of 10,000" samples
    (sklearn's own docs) -- training it on the full corpus is not "slow", it
    is algorithmically infeasible (confirmed: 11.6 CPU-hours into one fit()
    call with no path to completion and 22-28GB of paged memory against 16GB
    physical RAM). This is the fix, not a workaround: same detector, same
    RBF kernel, same class_weight="balanced" handling of the natural class
    ratio, just a tractable training-set size.
    """
    yield from FireFrameDataset(waveshare_index, scenes=scenes)
    rng = random.Random(seed)
    for session in iter_synth_sessions(data_root):
        yield from session.sample_fire_examples_random(per_camera_samples, rng)


def stream_contact_examples(
    data_root: str | Path, waveshare_index: DatasetIndex,
) -> Iterator[tuple[tuple[Frame, Frame, Frame], ContactEvent]]:
    """(frames_triplet, ContactEvent) pairs: real waveshare examples first,
    then every synthetic session's examples. Unlike fire, do NOT pass this
    directly to `MVSTGCNDetector`/`ThermoX3DDetector.fit()` — both
    materialize their `X`/`y` into a list internally, and both build T-frame
    sliding windows naively across whatever list they're given with no
    session-boundary awareness. Use `iter_contact_training_chunks` instead,
    which chunks per-session so windows never span two different sessions."""
    yield from ContactFrameDataset(waveshare_index)
    for session in iter_synth_sessions(data_root):
        yield from session.contact_examples()


def iter_contact_training_chunks(
    data_root: str | Path,
    waveshare_index: DatasetIndex,
    *,
    chunk_size: int = 5000,
) -> Iterator[tuple[str, list, list]]:
    """Yields ``(source_label, frames_chunk, events_chunk)`` — bounded-size,
    temporally-contiguous chunks, one session at a time, safe to feed
    individually to `MVSTGCNDetector.fit()`/`ThermoX3DDetector.fit()`.

    Chunking within a session (rather than only between sessions) is
    necessary because even one synthetic session can be up to ~1.45M
    timesteps — still far too large to materialize as one list. A small
    number of potential T-frame windows are lost at each chunk boundary
    (at most T-1, negligible against `chunk_size` in the thousands) —
    an accepted, documented trade-off, not a correctness bug.
    """

    def _chunks(iterable, source_label: str):
        buf_frames: list = []
        buf_events: list = []
        for frames, event in iterable:
            buf_frames.append(frames)
            buf_events.append(event)
            if len(buf_frames) >= chunk_size:
                yield source_label, buf_frames, buf_events
                buf_frames, buf_events = [], []
        if buf_frames:
            yield source_label, buf_frames, buf_events

    yield from _chunks(ContactFrameDataset(waveshare_index), "waveshare")
    for session in iter_synth_sessions(data_root):
        yield from _chunks(session.contact_examples(), session.scene)
