"""Generate and consume pseudo (silver) person-detection labels for
``waveshare_work`` scenarios that don't have real YOLO annotations yet.

Only 1 of 17 waveshare scenarios (``1_man_run``) currently has real bbox
annotations — the rest have real pixel data but no ground-truth boxes.
The imported baseline `MobileNetSSDDetector` (`scripts/import_baseline_weights.py`,
trained on the full dataset before those annotations were lost) can regenerate
approximate boxes by inference — the same idea as `image_annotator`'s
detection-assisted recommender, just applied dataset-wide instead of
per-frame-on-demand.

These are silver labels, not ground truth. Every function/class here is named
"pseudo" for a reason — don't let them get silently treated as real
annotations downstream (e.g. in the final report).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from thermal_algorithms.core.types import Detection, Frame
from thermal_algorithms.training.datasets import DatasetIndex, SessionMetadata
from thermal_algorithms.training.label_io import (
    PERSON_CLASS_ID,
    frame_index_from_label_path,
    list_label_files,
)

PseudoBoxes = dict[str, dict[int, dict[int, list[tuple[float, float, float, float, float]]]]]
"""``{scene: {channel: {frame_idx: [(x, y, w, h, score), ...]}}}``."""


def generate_pseudo_person_labels(
    index: DatasetIndex,
    detector,
    *,
    channels: Iterable[int] = (0, 1, 2),
    skip_annotated: bool = True,
) -> PseudoBoxes:
    """Run ``detector`` (a fitted person detector, e.g. the imported baseline
    ``MobileNetSSDDetector``) over every frame in ``index`` and collect its
    predictions as pseudo-labels.

    Args:
        skip_annotated: if True (default), frames that already have a real
            YOLO ``.txt`` annotation are skipped — real labels always take
            priority over silver ones, and this keeps pseudo-labels strictly
            additive (no overlap with `FrameLevelDataset`'s real examples).
    """
    channels = tuple(channels)
    result: PseudoBoxes = {}

    for session in index.sessions:
        scene_boxes: dict[int, dict[int, list]] = {}
        for ch in channels:
            if ch not in session.channels_with_data:
                continue
            annotated_indices: set[int] = set()
            if skip_annotated and ch in session.channels_with_labels:
                annotated_indices = {
                    frame_index_from_label_path(p)
                    for p in list_label_files(session.frames_dir(ch))
                }

            frames_arr = session.load_frames(ch)
            ch_boxes: dict[int, list] = {}
            for frame_idx in range(session.n_frames):
                if frame_idx in annotated_indices:
                    continue
                frame = Frame(
                    data=frames_arr[frame_idx],
                    timestamp=frame_idx / session.fps,
                    camera_id=ch,
                )
                dets = detector.predict(frame)
                ch_boxes[frame_idx] = [
                    (d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.score) for d in dets
                ]
            if ch_boxes:
                scene_boxes[ch] = ch_boxes
        if scene_boxes:
            result[session.session_id] = scene_boxes

    return result


def save_pseudo_labels(labels: PseudoBoxes, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        scene: {str(ch): {str(fi): boxes for fi, boxes in ch_boxes.items()} for ch, ch_boxes in scenes.items()}
        for scene, scenes in labels.items()
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh)


def load_pseudo_labels(path: str | Path) -> PseudoBoxes:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {
        scene: {
            int(ch): {int(fi): [tuple(b) for b in boxes] for fi, boxes in ch_boxes.items()}
            for ch, ch_boxes in scenes.items()
        }
        for scene, scenes in raw.items()
    }


@dataclass(frozen=True)
class _PseudoKey:
    session_idx: int
    channel: int
    frame_idx: int


class PseudoLabeledFrameDataset:
    """Yields ``(Frame, list[Detection])`` pairs using real frames from a
    ``DatasetIndex`` but silver bboxes from a ``PseudoBoxes`` mapping.

    Scoped strictly to frames present in ``pseudo_labels`` (i.e. whatever
    ``generate_pseudo_person_labels`` produced, normally excluding any
    already-annotated frames) — combine with a real `FrameLevelDataset` for
    the annotated portion, e.g. via ``itertools.chain`` over both datasets'
    examples, or ``merge_by_session`` for a session-grouped view.
    """

    def __init__(
        self,
        index: DatasetIndex,
        pseudo_labels: PseudoBoxes,
        *,
        min_score: float = 0.0,
    ) -> None:
        self._index = index
        self._pseudo_labels = pseudo_labels
        self._min_score = min_score
        self._sessions: list[SessionMetadata] = []
        self._keys: list[_PseudoKey] = []

        for sidx, session in enumerate(index.sessions):
            scene_boxes = pseudo_labels.get(session.session_id)
            if not scene_boxes:
                continue
            self._sessions.append(session)
            local_sidx = len(self._sessions) - 1
            for ch, ch_boxes in scene_boxes.items():
                for frame_idx in sorted(ch_boxes):
                    self._keys.append(_PseudoKey(local_sidx, ch, frame_idx))

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[tuple[Frame, list[Detection]]]:
        for key in self._keys:
            yield self._load(key)

    def _load(self, key: _PseudoKey) -> tuple[Frame, list[Detection]]:
        session = self._sessions[key.session_idx]
        frame = session.load_frame(key.channel, key.frame_idx)
        boxes = self._pseudo_labels[session.session_id][key.channel][key.frame_idx]
        dets = [
            Detection(bbox=(x, y, w, h), score=score, class_id=PERSON_CLASS_ID, camera_id=key.channel)
            for (x, y, w, h, score) in boxes
            if score >= self._min_score
        ]
        return frame, dets

    def by_session(self) -> Iterator[tuple[SessionMetadata, list[tuple[Frame, list[Detection]]]]]:
        if not self._keys:
            return
        current = self._keys[0].session_idx
        batch: list[tuple[Frame, list[Detection]]] = []
        for key in self._keys:
            if key.session_idx != current:
                yield self._sessions[current], batch
                current = key.session_idx
                batch = []
            batch.append(self._load(key))
        if batch:
            yield self._sessions[current], batch


def merge_by_session(*datasets) -> Iterator[tuple[SessionMetadata, list]]:
    """Chain multiple datasets' ``by_session()`` groups together — e.g. a
    real `FrameLevelDataset` (annotated scenes) + a `PseudoLabeledFrameDataset`
    (silver-labeled scenes), for `Trainer.evaluate_human_detection`."""
    for ds in datasets:
        yield from ds.by_session()
