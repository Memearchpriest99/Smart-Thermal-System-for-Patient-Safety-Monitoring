"""Dataset classes that turn the recorded thermal sessions on disk into
algorithm-ready training data.

Five levels / classes:

    1. SessionMetadata      - a single recording session's catalog.
    2. DatasetIndex         - scans a dataset root and catalogs all sessions,
                              handling both folder layouts:
                                  scene/timestamp_session/ch{N}_*   (layout A)
                                  scene/ch{N}_*                      (layout B)
    3. FrameLevelDataset    - (Frame, list[Detection]) pairs for HumanDetector
                              training.  Use class_filter=[PERSON_CLASS_ID].
    4. FireFrameDataset     - (Frame, FireAlert) pairs for FireSVMDetector
                              training.  Wraps FrameLevelDataset with
                              class_filter=[FIRE_CLASS_ID] and converts
                              Detection lists to FireAlert objects.
    5. ContactFrameDataset  - (ThreeViewFrames, ContactEvent) pairs for
                              MVSTGCNDetector / ThermoX3DDetector training.
                              Reads per-session contact_labels.csv files.

Class convention (user-confirmed):
    YOLO class 0 = fire / ignition source   (FIRE_CLASS_ID)
    YOLO class 1 = person                   (PERSON_CLASS_ID)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile, MLX90640
from thermal_algorithms.core.types import ContactEvent, Detection, FireAlert, FireLevel, Frame
from thermal_algorithms.training.label_io import (
    CONTACT_LABEL_FILENAME,
    FIRE_CLASS_ID,
    PERSON_CLASS_ID,
    frame_index_from_label_path,
    has_contact_labels,
    list_label_files,
    load_classes_file,
    load_contact_labels,
    load_yolo_labels,
)


@dataclass(frozen=True)
class SessionMetadata:
    """Catalog of a single recording session."""

    scene: str
    session: Optional[str]
    root: Path
    sensor_profile: SensorProfile
    fps: float
    n_frames: int
    channels_with_data: tuple[int, ...]
    channels_with_labels: tuple[int, ...]
    class_names: dict[int, str] = field(default_factory=dict)

    @property
    def has_labels(self) -> bool:
        return len(self.channels_with_labels) > 0

    @property
    def session_id(self) -> str:
        return f"{self.scene}/{self.session}" if self.session else self.scene

    def npz_path(self, channel: int) -> Path:
        return self.root / f"ch{channel}_raw_data.npz"

    def frames_dir(self, channel: int) -> Path:
        return self.root / f"ch{channel}_frames"

    def load_frame(self, channel: int, frame_idx: int) -> Frame:
        arr = np.load(self.npz_path(channel))["frames"][frame_idx]
        return Frame(
            data=arr.astype(np.float32),
            timestamp=frame_idx / self.fps,
            camera_id=channel,
            metadata={"session_id": self.session_id},
        )

    def load_frames(self, channel: int) -> np.ndarray:
        return np.load(self.npz_path(channel))["frames"].astype(np.float32)


class DatasetIndex:
    """Catalog of every session under a dataset root.

    Handles both folder layouts in one pass:
        Layout A:  root/scene/timestamp/ch{N}_*
        Layout B:  root/scene/ch{N}_*

    Sessions are sorted deterministically by (scene, session).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        sensor_profile: SensorProfile = MLX90640,
        fps: float = 8.0,
    ) -> None:
        self._root = Path(root)
        self._sensor_profile = sensor_profile
        self._fps = float(fps)
        self._sessions: tuple[SessionMetadata, ...] = self._scan()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def sessions(self) -> tuple[SessionMetadata, ...]:
        return self._sessions

    def labeled_sessions(self) -> tuple[SessionMetadata, ...]:
        return tuple(s for s in self._sessions if s.has_labels)

    def empty_room_sessions(
        self, name_hints: Iterable[str] = ("empty",),
    ) -> tuple[SessionMetadata, ...]:
        hints = tuple(h.lower() for h in name_hints)
        return tuple(
            s for s in self._sessions
            if any(h in s.scene.lower() for h in hints)
        )

    def find(self, scene: str, session: Optional[str] = None) -> SessionMetadata:
        for s in self._sessions:
            if s.scene == scene and (session is None or s.session == session):
                return s
        raise KeyError(f"No session matching scene={scene!r}, session={session!r}")

    def _scan(self) -> tuple[SessionMetadata, ...]:
        sessions: list[SessionMetadata] = []
        if not self._root.is_dir():
            return ()

        for scene_dir in sorted(self._root.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("_"):
                continue

            # Layout B (direct): ch0_raw_data.npz sits at the scene level.
            if (scene_dir / "ch0_raw_data.npz").is_file():
                meta = self._build_metadata(scene_dir.name, None, scene_dir)
                if meta is not None:
                    sessions.append(meta)
                continue

            # Layout A: timestamped subfolders, each a session.
            for sub in sorted(scene_dir.iterdir()):
                if not sub.is_dir() or sub.name.startswith("_") \
                        or sub.name in {"ch0_frames", "ch1_frames", "ch2_frames"}:
                    continue
                if (sub / "ch0_raw_data.npz").is_file():
                    meta = self._build_metadata(scene_dir.name, sub.name, sub)
                    if meta is not None:
                        sessions.append(meta)

        return tuple(sessions)

    def _build_metadata(
        self, scene: str, session: Optional[str], root: Path,
    ) -> Optional[SessionMetadata]:
        chans_data: list[int] = []
        chans_labels: list[int] = []
        n_frames = 0
        for ch in (0, 1, 2):
            npz = root / f"ch{ch}_raw_data.npz"
            if npz.is_file():
                chans_data.append(ch)
                if n_frames == 0:
                    try:
                        with np.load(npz) as data:
                            n_frames = int(data["frames"].shape[0])
                    except Exception:
                        n_frames = 0
            frames_dir = root / f"ch{ch}_frames"
            if frames_dir.is_dir() and list_label_files(frames_dir):
                chans_labels.append(ch)

        if not chans_data:
            return None

        class_names: dict[int, str] = {}
        classes_txt = root / "classes.txt"
        if classes_txt.is_file():
            try:
                class_names = load_classes_file(classes_txt)
            except Exception:
                class_names = {}

        return SessionMetadata(
            scene=scene,
            session=session,
            root=root,
            sensor_profile=self._sensor_profile,
            fps=self._fps,
            n_frames=n_frames,
            channels_with_data=tuple(chans_data),
            channels_with_labels=tuple(chans_labels),
            class_names=class_names,
        )

    def __repr__(self) -> str:
        n_lab = sum(1 for s in self._sessions if s.has_labels)
        return (
            f"DatasetIndex(root={str(self._root)!r}, sessions={len(self._sessions)}, "
            f"labeled={n_lab}, sensor={self._sensor_profile.name})"
        )


@dataclass(frozen=True)
class _FrameKey:
    session_idx: int
    channel: int
    frame_idx: int


class FrameLevelDataset:
    """Yields (Frame, list[Detection]) pairs for HumanDetector training.

    Iterates over every (session, channel, frame_idx) triple where the
    session has bbox labels in that channel and there's a frame_XXXXX.txt
    matching the frame index. Frames with no .txt file (empty/negative
    frames) are excluded by default; set include_negative_frames=True to
    yield them with an empty Detection list.
    """

    def __init__(
        self,
        index: DatasetIndex,
        *,
        channels: Iterable[int] = (0, 1, 2),
        scenes: Optional[Iterable[str]] = None,
        class_filter: Optional[Iterable[int]] = None,
        include_negative_frames: bool = False,
    ) -> None:
        self._index = index
        self._channels = tuple(channels)
        self._scenes = set(scenes) if scenes is not None else None
        self._class_filter = (
            tuple(class_filter) if class_filter is not None else None
        )
        self._include_negative_frames = bool(include_negative_frames)

        sessions = index.labeled_sessions()
        if self._scenes is not None:
            sessions = tuple(s for s in sessions if s.scene in self._scenes)
        self._sessions = sessions

        self._examples: tuple[_FrameKey, ...] = tuple(self._enumerate_examples())

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> tuple[Frame, list[Detection]]:
        if not 0 <= i < len(self._examples):
            raise IndexError(i)
        return self._load_example(self._examples[i])

    def __iter__(self) -> Iterator[tuple[Frame, list[Detection]]]:
        for key in self._examples:
            yield self._load_example(key)

    @property
    def sessions(self) -> tuple[SessionMetadata, ...]:
        return self._sessions

    def _enumerate_examples(self) -> Iterable[_FrameKey]:
        for sidx, session in enumerate(self._sessions):
            for ch in self._channels:
                if ch not in session.channels_with_labels:
                    continue
                label_files = list_label_files(session.frames_dir(ch))
                labeled_indices = {frame_index_from_label_path(p) for p in label_files}
                if self._include_negative_frames:
                    iterable = range(session.n_frames)
                else:
                    iterable = sorted(labeled_indices)
                for f in iterable:
                    if f < 0 or f >= session.n_frames:
                        continue
                    yield _FrameKey(session_idx=sidx, channel=ch, frame_idx=f)

    def _load_example(self, key: _FrameKey) -> tuple[Frame, list[Detection]]:
        session = self._sessions[key.session_idx]
        frame = session.load_frame(key.channel, key.frame_idx)

        label_path = session.frames_dir(key.channel) / f"frame_{key.frame_idx:05d}.txt"
        detections = load_yolo_labels(
            label_path,
            frame_shape=frame.shape,
            camera_id=key.channel,
            class_filter=self._class_filter,
        )
        return frame, detections

    def by_session(
        self,
    ) -> "Iterator[tuple[SessionMetadata, list[tuple[Frame, list[Detection]]]]]":
        """Yield ``(session, examples)`` groups in session order.

        Callers use this to reset stateful detectors at session boundaries and
        to produce per-scene ``ScenarioResult`` objects for the report tables.
        """
        if not self._examples:
            return
        current_sidx = self._examples[0].session_idx
        batch: list[tuple[Frame, list[Detection]]] = []
        for key in self._examples:
            if key.session_idx != current_sidx:
                yield self._sessions[current_sidx], batch
                current_sidx = key.session_idx
                batch = []
            batch.append(self._load_example(key))
        if batch:
            yield self._sessions[current_sidx], batch


def sample_background_patches(
    index: DatasetIndex,
    *,
    patch_h: int,
    patch_w: int,
    n_patches: int,
    name_hints: Iterable[str] = ("empty",),
    channels: Iterable[int] = (0, 1, 2),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample n_patches background patches from empty-room recordings.

    Useful as the negative class for HOG+SVM training (the labeled dataset
    has no explicit 'background' annotations).
    """
    rng = rng if rng is not None else np.random.default_rng()
    sessions = index.empty_room_sessions(name_hints=name_hints)
    if not sessions:
        raise ValueError(
            f"No empty-room sessions found in {index.root} matching {tuple(name_hints)!r}."
        )

    sources: list[tuple[np.ndarray, int]] = []
    for s in sessions:
        for ch in channels:
            if ch in s.channels_with_data:
                arr = s.load_frames(ch)
                sources.append((arr, ch))
    if not sources:
        raise ValueError("No channels with data in the matched empty-room sessions.")

    patches = np.zeros((n_patches, patch_h, patch_w), dtype=np.float32)
    for i in range(n_patches):
        arr, _ch = sources[rng.integers(0, len(sources))]
        k = int(rng.integers(0, arr.shape[0]))
        h, w = arr.shape[1], arr.shape[2]
        if h < patch_h or w < patch_w:
            raise ValueError(
                f"Frame ({h}, {w}) is smaller than requested patch ({patch_h}, {patch_w})."
            )
        y = int(rng.integers(0, h - patch_h + 1))
        x = int(rng.integers(0, w - patch_w + 1))
        patches[i] = arr[k, y:y + patch_h, x:x + patch_w]
    return patches


# ---------------------------------------------------------------------------
# FireFrameDataset
# ---------------------------------------------------------------------------

class FireFrameDataset:
    """Yields ``(Frame, FireAlert)`` pairs for ``FireSVMDetector`` training.

    Wraps ``FrameLevelDataset`` with ``class_filter=[FIRE_CLASS_ID]`` and
    converts each frame's ``list[Detection]`` into a ``FireAlert``:

    * Frame with ≥ 1 fire bbox  →  ``FireAlert(level=ACTIVE_COMBUSTION, ...)``
    * Frame with 0 fire bboxes  →  ``FireAlert(level=SAFE, ...)``

    The fire bounding boxes are stored in ``blob_features`` so downstream
    code can use them for IoU evaluation without re-running detection.

    Args:
        index: Dataset index to draw sessions from.
        channels: Camera channels to include (default all three).
        scenes: If given, only sessions from these scene names are included.
        include_negative_frames: Whether to yield frames that have no fire
            annotations (i.e. ground-truth SAFE frames).  Default ``True``
            because the SVM needs balanced negative examples.
    """

    def __init__(
        self,
        index: DatasetIndex,
        *,
        channels: Iterable[int] = (0, 1, 2),
        scenes: Optional[Iterable[str]] = None,
        include_negative_frames: bool = True,
    ) -> None:
        self._include_negatives = bool(include_negative_frames)
        # Always request all annotated frames (including person-only ones) so
        # that frames with person bboxes but no fire bbox can appear as SAFE
        # examples.  We apply our own positive/negative filter afterwards.
        self._inner = FrameLevelDataset(
            index,
            channels=channels,
            scenes=scenes,
            class_filter=[FIRE_CLASS_ID],
            include_negative_frames=True,
        )
        # Pre-filter once if negatives are not wanted (avoids per-call overhead)
        if not include_negative_frames:
            self._examples: list[int] = [
                i for i in range(len(self._inner))
                if self._inner[i][1]  # fire_dets non-empty
            ]
        else:
            self._examples = list(range(len(self._inner)))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> tuple[Frame, FireAlert]:
        frame, fire_dets = self._inner[self._examples[i]]
        return frame, _dets_to_fire_alert(frame.timestamp, fire_dets)

    def __iter__(self) -> Iterator[tuple[Frame, FireAlert]]:
        for idx in self._examples:
            frame, fire_dets = self._inner[idx]
            yield frame, _dets_to_fire_alert(frame.timestamp, fire_dets)

    @property
    def sessions(self) -> tuple[SessionMetadata, ...]:
        return self._inner.sessions

    def by_session(
        self,
    ) -> "Iterator[tuple[SessionMetadata, list[tuple[Frame, FireAlert]]]]":
        """Yield ``(session, examples)`` groups for session-aware evaluation."""
        if not self._examples:
            return
        inner_keys = self._inner._examples   # tuple[_FrameKey]
        current_sidx = inner_keys[self._examples[0]].session_idx
        batch: list[tuple[Frame, FireAlert]] = []
        for outer_idx in self._examples:
            sidx = inner_keys[outer_idx].session_idx
            if sidx != current_sidx:
                yield self._inner.sessions[current_sidx], batch
                current_sidx = sidx
                batch = []
            frame, fire_dets = self._inner[outer_idx]
            batch.append((frame, _dets_to_fire_alert(frame.timestamp, fire_dets)))
        if batch:
            yield self._inner.sessions[current_sidx], batch


def _dets_to_fire_alert(timestamp: float, fire_dets: list[Detection]) -> FireAlert:
    """Convert a list of fire-class detections to a ``FireAlert`` label."""
    if fire_dets:
        # Expose fire bboxes in blob_features for IoU evaluation
        blob_features: dict = {
            "n_fire_boxes": float(len(fire_dets)),
            "bboxes": [d.bbox for d in fire_dets],
            "max_score": float(max(d.score for d in fire_dets)),
        }
        return FireAlert(
            level=FireLevel.ACTIVE_COMBUSTION,
            timestamp=timestamp,
            blob_features=blob_features,
            confidence=1.0,
        )
    return FireAlert(level=FireLevel.SAFE, timestamp=timestamp, confidence=1.0)


# ---------------------------------------------------------------------------
# ContactFrameDataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ContactKey:
    session_idx: int
    frame_idx: int


class ContactFrameDataset:
    """Yields ``(ThreeViewFrames, ContactEvent)`` pairs for contact detector training.

    Reads binary contact labels from per-session ``contact_labels.csv`` files.
    Only sessions that have both a ``contact_labels.csv`` and data on all
    ``required_channels`` are included.

    Label format (``contact_labels.csv``)::

        frame_idx,contact
        0,0
        1,0
        5,1
        6,1

    Frames absent from the CSV are excluded from the dataset.  Annotate every
    frame you intend to use.

    Args:
        index: Dataset index to draw sessions from.
        scenes: If given, only sessions from these scene names are included.
        required_channels: Channels that must all have data.  Default (0,1,2)
            for three-view contact detection.
        contact_label_filename: Override the default CSV filename.
    """

    def __init__(
        self,
        index: DatasetIndex,
        *,
        scenes: Optional[Iterable[str]] = None,
        required_channels: Iterable[int] = (0, 1, 2),
        contact_label_filename: str = CONTACT_LABEL_FILENAME,
    ) -> None:
        self._index = index
        self._required_channels = tuple(required_channels)
        self._label_filename = contact_label_filename
        self._scene_filter: Optional[set[str]] = (
            set(scenes) if scenes is not None else None
        )

        self._sessions, self._labels_per_session = self._scan()
        self._examples: tuple[_ContactKey, ...] = tuple(
            self._enumerate_examples()
        )

    # ---- Public API --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(
        self, i: int
    ) -> tuple[tuple[Frame, Frame, Frame], ContactEvent]:
        if not 0 <= i < len(self._examples):
            raise IndexError(i)
        return self._load(self._examples[i])

    def __iter__(
        self,
    ) -> Iterator[tuple[tuple[Frame, Frame, Frame], ContactEvent]]:
        for key in self._examples:
            yield self._load(key)

    @property
    def sessions(self) -> tuple[SessionMetadata, ...]:
        return self._sessions

    def by_session(self) -> "Iterator[tuple[SessionMetadata, list]]":
        """Yield ``(session, examples)`` groups for session-aware evaluation."""
        if not self._examples:
            return
        current_sidx = self._examples[0].session_idx
        batch: list = []
        for key in self._examples:
            if key.session_idx != current_sidx:
                yield self._sessions[current_sidx], batch
                current_sidx = key.session_idx
                batch = []
            batch.append(self._load(key))
        if batch:
            yield self._sessions[current_sidx], batch

    def label_counts(self) -> dict[str, int]:
        """Return ``{'contact': N, 'no_contact': M}`` across all examples."""
        n_contact = sum(
            1
            for key in self._examples
            if self._labels_per_session[key.session_idx][key.frame_idx] == 1
        )
        return {"contact": n_contact, "no_contact": len(self._examples) - n_contact}

    # ---- Internal ----------------------------------------------------------

    def _scan(
        self,
    ) -> tuple[tuple[SessionMetadata, ...], list[dict[int, int]]]:
        sessions: list[SessionMetadata] = []
        labels: list[dict[int, int]] = []

        for s in self._index.sessions:
            if self._scene_filter is not None and s.scene not in self._scene_filter:
                continue
            # Must have data on every required channel
            if not all(ch in s.channels_with_data for ch in self._required_channels):
                continue
            csv_path = s.root / self._label_filename
            contact_labels = load_contact_labels(csv_path)
            if not contact_labels:
                continue
            sessions.append(s)
            labels.append(contact_labels)

        return tuple(sessions), labels

    def _enumerate_examples(self) -> Iterable[_ContactKey]:
        for sidx, (session, label_map) in enumerate(
            zip(self._sessions, self._labels_per_session)
        ):
            for frame_idx in sorted(label_map):
                if frame_idx < 0 or frame_idx >= session.n_frames:
                    continue
                yield _ContactKey(session_idx=sidx, frame_idx=frame_idx)

    def _load(
        self, key: _ContactKey
    ) -> tuple[tuple[Frame, Frame, Frame], ContactEvent]:
        session = self._sessions[key.session_idx]
        label_map = self._labels_per_session[key.session_idx]
        contact = bool(label_map[key.frame_idx])
        timestamp = key.frame_idx / session.fps

        frames = tuple(
            session.load_frame(ch, key.frame_idx)
            for ch in self._required_channels
        )

        event = ContactEvent(
            actors=(),
            pairs_in_contact=((0, 1),) if contact else (),
            timestamp=timestamp,
            confidence=1.0,
        )
        return frames, event  # type: ignore[return-value]
