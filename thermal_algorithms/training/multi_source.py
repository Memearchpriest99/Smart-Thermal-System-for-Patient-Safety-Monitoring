"""Unify the three dataset sources behind one interface per problem.

* ``waveshare_work`` — real YOLO bboxes (currently only for `1_man_run`'s
  ~195 annotated frames) + complete per-frame contact labels, read via the
  existing ``DatasetIndex``/``FrameLevelDataset``/``FireFrameDataset``/
  ``ContactFrameDataset``.
* ``synth_room_*`` / ``room-1`` — HDF5-backed raw frames (``hdf5_source.py``)
  joined against room-level event-interval labels (``label_join.py``). There
  is no bounding-box ground truth in this source at all — only frame-level
  fire/human/contact presence booleans.

Because of that asymmetry, this module deliberately does NOT pretend synth/
room-1 data has bounding boxes. Contact is a clean unification (both sources
are frame-level binary already); fire is unified as presence-only (still
useful — ``FireAlert`` itself carries no bbox, just a level/confidence, so
losing bbox-level detail doesn't break anything downstream that only reads
``FireAlert.level``); human detection is NOT unified into the existing
``FrameLevelDataset``/``list[Detection]`` shape — ``HDF5Session.
human_presence_examples()`` yields plain ``(Frame, bool)`` pairs instead, and
callers must be written against that shape explicitly rather than assuming
IoU/bbox-based evaluation works across all three sources.

``Trainer.evaluate_contact_detection``/``evaluate_fire_detection`` only need
``dataset.by_session()`` to yield ``(session, examples)`` where ``session``
has a ``.scene`` attribute — that's duck-typed, not an ``isinstance`` check —
so ``MultiSourceContactDataset``/``MultiSourceFireDataset`` work as direct
drop-ins there. ``Trainer.fit_and_evaluate`` DOES ``isinstance``-dispatch on
the dataset type, so it does not recognize these merged datasets; assemble
training examples via plain iteration instead (see each class's docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from thermal_algorithms.core.types import ContactEvent, Frame, FireAlert, FireLevel
from thermal_algorithms.training.datasets import ContactFrameDataset, FireFrameDataset
from thermal_algorithms.training.hdf5_source import HDF5CameraSession
from thermal_algorithms.training.label_join import LabelJoiner, detect_room_id


@dataclass(frozen=True)
class HDF5SessionRef:
    """Pointer to one ``<room_root>/<date>/cam_{0,1,2}`` session, before any
    file is opened."""

    room_root: Path
    room_id: str
    date: str
    date_dir: Path

    def cam_dir(self, cam: int) -> Path:
        return self.date_dir / f"cam_{cam}"


def discover_hdf5_sessions(room_root: str | Path, room_id: str) -> list[HDF5SessionRef]:
    """Scan a room's root directory (e.g. ``data/synth_room_1``,
    ``data/Room_1/room-1``) for ``<date>/cam_N`` session folders.

    Args:
        room_root: the room's directory, containing one subfolder per
            recording date.
        room_id: the ``Room_ID`` string this room's ``labels.csv`` uses
            (must match exactly — e.g. ``"synth_room_1"``, ``"room-1"``).
    """
    room_root = Path(room_root)
    refs: list[HDF5SessionRef] = []
    if not room_root.is_dir():
        return refs
    for date_dir in sorted(room_root.iterdir()):
        if not date_dir.is_dir():
            continue
        if not any((date_dir / f"cam_{c}").is_dir() for c in (0, 1, 2)):
            continue
        refs.append(
            HDF5SessionRef(room_root=room_root, room_id=room_id, date=date_dir.name, date_dir=date_dir)
        )
    return refs


class HDF5Session:
    """One full multi-camera session (one room, one date): HDF5 frame access
    joined against that room's ``labels.csv``.

    No bounding-box ground truth exists in this source — fire/human labels
    are frame-level presence only (see module docstring).
    """

    def __init__(
        self,
        ref: HDF5SessionRef,
        labels_csv: str | Path,
        *,
        cache_chunks: int = 2,
    ) -> None:
        self.ref = ref
        self._cams: dict[int, HDF5CameraSession] = {}
        for c in (0, 1, 2):
            cam_dir = ref.cam_dir(c)
            if cam_dir.is_dir():
                self._cams[c] = HDF5CameraSession(
                    cam_dir, cam_id=c, session_date=ref.date, cache_chunks=cache_chunks
                )
        if not self._cams:
            raise ValueError(f"no cam_N directories under {ref.date_dir}")
        # Filter by the labels.csv's OWN Room_ID for this date, not the
        # folder-derived ref.room_id -- some rooms' label files were copied
        # from a differently-numbered room in the original generation set
        # and never relabeled to match their folder (see detect_room_id's
        # docstring). ref.room_id is still used for .scene/display naming
        # below; only label filtering needs the CSV's ground truth.
        actual_room_id = detect_room_id(labels_csv, date=ref.date)
        self._joiner = LabelJoiner.from_csv(labels_csv, room_id=actual_room_id, date=ref.date)

    @property
    def scene(self) -> str:
        """Duck-types ``SessionMetadata.scene`` for ``Trainer.evaluate_*``
        per-session reporting."""
        return f"{self.ref.room_id}/{self.ref.date}"

    @property
    def cameras(self) -> tuple[int, ...]:
        return tuple(sorted(self._cams))

    def contact_examples(self) -> Iterator[tuple[tuple[Frame, ...], ContactEvent]]:
        """Yields ``(frames, ContactEvent)`` — requires cams 0, 1 and 2 all
        present; yields nothing otherwise."""
        if not all(c in self._cams for c in (0, 1, 2)):
            return
        n = min(self._cams[c].n_frames for c in (0, 1, 2))
        for i in range(n):
            frames = tuple(self._cams[c].load_frame(i) for c in (0, 1, 2))
            labels = self._joiner.labels_for(frames[0].timestamp)
            if labels is None:
                continue
            event = ContactEvent(
                actors=(),
                pairs_in_contact=((0, 1),) if labels["contact"] else (),
                timestamp=frames[0].timestamp,
                confidence=1.0,
            )
            yield frames, event

    def fire_examples(self) -> Iterator[tuple[Frame, FireAlert]]:
        """Yields ``(Frame, FireAlert)`` across every available camera —
        presence only; ``FireAlert.blob_features`` is always empty here (no
        bbox exists to put in it)."""
        for cam_id in self.cameras:
            for frame in self._cams[cam_id].iter_frames():
                labels = self._joiner.labels_for(frame.timestamp)
                if labels is None:
                    continue
                level = FireLevel.ACTIVE_COMBUSTION if labels["fire"] else FireLevel.SAFE
                yield frame, FireAlert(level=level, timestamp=frame.timestamp, confidence=1.0)

    def sample_fire_examples_random(
        self, per_camera_samples: int, rng: "random.Random", *, chunks_per_camera: int = 8,
    ) -> Iterator[tuple[Frame, FireAlert]]:
        """Like fire_examples(), but visits only a bounded number of chunks
        per camera instead of the whole session, sampling `per_camera_samples`
        frames from within those chunks.

        Exists because sklearn's SVC is more-than-quadratic in sample count
        and explicitly unsuited to datasets much past "a couple of 10,000"
        samples (its own docs) -- the full synthetic corpus is ~31M
        frame-instances, so FireSVMDetector's full-corpus training needs a
        subsample.

        The decode cost here is per-CHUNK, not per-frame (each chunk is a
        compressed ~7200-frame block that must be fully decompressed before
        any single frame in it is readable) -- sampling `per_camera_samples`
        individual frame INDICES uniformly at random over the whole session
        does NOT save time at any sample size that matters: with ~150-200
        chunks per session, a few thousand random frame indices will, by the
        pigeonhole/coupon-collector effect, land in nearly every chunk
        anyway, degenerating to a full scan (confirmed: an earlier version of
        this function did exactly that and was not meaningfully faster).
        Sampling a small, FIXED number of chunks first (chunks_per_camera),
        then drawing many frames from each chosen chunk (near-free once that
        chunk is already decoded), is what actually bounds decode cost.
        """
        for cam_id in self.cameras:
            cam = self._cams[cam_id]
            n_chunks = cam.n_chunks
            chosen_chunks = sorted(rng.sample(range(n_chunks), min(chunks_per_camera, n_chunks)))
            per_chunk_budget = max(1, -(-per_camera_samples // len(chosen_chunks)))  # ceil div
            for chunk_i in chosen_chunks:
                start, stop = cam.chunk_frame_range(chunk_i)
                k = min(per_chunk_budget, stop - start)
                for idx in sorted(rng.sample(range(start, stop), k)):
                    frame = cam.load_frame(idx)
                    labels = self._joiner.labels_for(frame.timestamp)
                    if labels is None:
                        continue
                    level = FireLevel.ACTIVE_COMBUSTION if labels["fire"] else FireLevel.SAFE
                    yield frame, FireAlert(level=level, timestamp=frame.timestamp, confidence=1.0)

    def human_presence_examples(self) -> Iterator[tuple[Frame, bool]]:
        """Yields ``(Frame, is_human_present)`` across every available
        camera. Explicitly presence-only — NOT a ``(Frame, list[Detection])``
        pair, since this source has no bounding boxes. Not a drop-in for
        ``FrameLevelDataset``; only use with code written against this plain
        boolean shape."""
        for cam_id in self.cameras:
            for frame in self._cams[cam_id].iter_frames():
                labels = self._joiner.labels_for(frame.timestamp)
                if labels is None:
                    continue
                yield frame, labels["human"]


class MultiSourceContactDataset:
    """Combines ``waveshare_work``'s real ``ContactFrameDataset`` with
    ``HDF5Session`` contact examples from ``synth_room_*``/``room-1`` behind
    one iterable surface (``by_session``, ``label_counts``, plain
    iteration).

    Drop-in for ``Trainer.evaluate_contact_detection`` (only needs
    ``by_session()`` yielding sessions with ``.scene``). NOT recognized by
    ``Trainer.fit_and_evaluate``'s ``isinstance`` dispatch — for training,
    assemble examples yourself::

        frame_seqs = [triplet for triplet, _ in merged]
        events = [e for _, e in merged]
        detector.fit(frame_seqs, events)
    """

    def __init__(
        self,
        waveshare_dataset: Optional[ContactFrameDataset] = None,
        hdf5_sessions: Iterable[HDF5Session] = (),
    ) -> None:
        self._waveshare = waveshare_dataset
        self._hdf5_sessions = tuple(hdf5_sessions)

    def __iter__(self):
        if self._waveshare is not None:
            yield from self._waveshare
        for sess in self._hdf5_sessions:
            yield from sess.contact_examples()

    def by_session(self):
        if self._waveshare is not None:
            yield from self._waveshare.by_session()
        for sess in self._hdf5_sessions:
            yield sess, list(sess.contact_examples())

    def label_counts(self) -> dict[str, int]:
        counts = {"contact": 0, "no_contact": 0}
        if self._waveshare is not None:
            wc = self._waveshare.label_counts()
            counts["contact"] += wc.get("contact", 0)
            counts["no_contact"] += wc.get("no_contact", 0)
        for sess in self._hdf5_sessions:
            for _frames, event in sess.contact_examples():
                if event.any_contact:
                    counts["contact"] += 1
                else:
                    counts["no_contact"] += 1
        return counts


class MultiSourceFireDataset:
    """Combines ``waveshare_work``'s real ``FireFrameDataset`` (genuine
    bboxes, but currently only for the ~195 annotated ``1_man_run`` frames)
    with ``HDF5Session`` fire examples (presence-only, all scenes, no bbox)
    from ``synth_room_*``/``room-1``.

    Same drop-in/assembly caveats as ``MultiSourceContactDataset`` — see its
    docstring.
    """

    def __init__(
        self,
        waveshare_dataset: Optional[FireFrameDataset] = None,
        hdf5_sessions: Iterable[HDF5Session] = (),
    ) -> None:
        self._waveshare = waveshare_dataset
        self._hdf5_sessions = tuple(hdf5_sessions)

    def __iter__(self):
        if self._waveshare is not None:
            yield from self._waveshare
        for sess in self._hdf5_sessions:
            yield from sess.fire_examples()

    def by_session(self):
        if self._waveshare is not None:
            yield from self._waveshare.by_session()
        for sess in self._hdf5_sessions:
            yield sess, list(sess.fire_examples())
