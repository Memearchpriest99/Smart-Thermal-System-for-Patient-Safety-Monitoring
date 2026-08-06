#!/usr/bin/env python3
"""Compare TatenoPipeline vs. GlobalNormPreprocessor (Task 2).

Generalized runner (per the plan — `ablate_preprocessing.py` is hardcoded to
one contact-detector config and pulls in several `eval_waveshare_contact_v*`
helper scripts, not a reusable template). All of waveshare_work now has real
per-frame ground truth (fire + person bboxes, contact labels) across all 17
scenarios, so every comparison below runs on the full real dataset — no
synthetic/room-1 fallback needed (room-1 is excluded from the project
entirely; see data/DATASET_NOTES.md).

Two levels of comparison:

  1. SBR (`Trainer.evaluate_preprocessing`) — needs real ground-truth bboxes
     to define the signal region; now meaningful across the whole dataset.
  2. Full detector-level metrics (`evaluate_human_detection`/
     `evaluate_fire_detection`/`evaluate_contact_detection`) with the
     imported `checkpoints_baseline_waveshare/` detectors, feeding each
     preprocessor's output in — "better results" judged on F1/IoU/precision/
     recall, not just SBR.

Usage::

    python scripts/eval_preprocessing_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector  # noqa: E402
from thermal_algorithms.core.checkpoints import CheckpointRegistry  # noqa: E402
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984  # noqa: E402
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector  # noqa: E402
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector  # noqa: E402
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector  # noqa: E402
from thermal_algorithms.preprocessing import GlobalNormPreprocessor, TatenoPipeline  # noqa: E402
from thermal_algorithms.training import (  # noqa: E402
    ContactFrameDataset,
    DatasetIndex,
    FireFrameDataset,
    FrameLevelDataset,
    Trainer,
)
from thermal_algorithms.training.label_io import PERSON_CLASS_ID  # noqa: E402

DATA_ROOT = _REPO_ROOT.parent / "data"
CHECKPOINTS = _REPO_ROOT / "checkpoints_baseline_waveshare"
OUT_JSON = _REPO_ROOT / "reports" / "preprocessing_comparison_results.json"


def build_preprocessors(waveshare_index: DatasetIndex) -> dict:
    """One TatenoPipeline (calibrated on real empty/calibrate-room frames)
    and one GlobalNormPreprocessor (no calibration needed) per Waveshare_26984.

    empty_room/calibrate_room have zero YOLO annotations, so they're excluded
    from `DatasetIndex.labeled_sessions()` and `FrameLevelDataset` can't see
    them at all — load frames directly from their npz files instead, the
    same way `sample_background_patches` does.
    """
    calib_sessions = waveshare_index.empty_room_sessions(name_hints=("empty", "calibrate"))
    calib_frames = [
        session.load_frame(ch, i)
        for session in calib_sessions
        for ch in session.channels_with_data
        for i in range(session.n_frames)
    ]
    print(f"Calibrating TatenoPipeline on {len(calib_frames)} empty/calibrate-room frames "
          f"from {[s.scene for s in calib_sessions]} ...")
    tateno = TatenoPipeline(sensor_profile=WAVESHARE_26984).fit(calib_frames)
    global_norm = GlobalNormPreprocessor(sensor_profile=WAVESHARE_26984).fit()
    return {"tateno": tateno, "global_norm": global_norm}


def run_sbr_comparison(waveshare_index: DatasetIndex, preprocessors: dict) -> dict:
    print("\n=== SBR comparison (full waveshare dataset, real bboxes) ===")
    ds = FrameLevelDataset(
        waveshare_index, class_filter=[PERSON_CLASS_ID], include_negative_frames=False,
    )
    print(f"  {len(ds)} real person-annotated frames")
    results = {}
    for name, prep in preprocessors.items():
        raw_sbr, proc_sbr = Trainer.evaluate_preprocessing(ds, prep, verbose=False)
        factor = proc_sbr / raw_sbr if raw_sbr > 0 else float("nan")
        print(f"  {name:<12} raw={raw_sbr:.3f}  processed={proc_sbr:.3f}  improvement={factor:.2f}x")
        results[name] = {"raw_sbr": raw_sbr, "processed_sbr": proc_sbr, "improvement": factor}
    return results


class _StridedBySessionDataset:
    """Wraps a dataset's `by_session()` to keep every Nth example per scene.

    Only used for HOG-SVM: profiled at ~4.4s/frame (sliding-window multi-scale
    search) — the full 8259-frame dataset x 2 preprocessors would take ~20
    hours. MobileNet-SSD (~7ms/frame) runs on the full dataset; this keeps
    HOG-SVM's numbers real (still every scene, just sparser) rather than
    skipping it.
    """

    def __init__(self, inner, stride: int):
        self._inner = inner
        self._stride = stride

    def by_session(self):
        for session, examples in self._inner.by_session():
            yield session, examples[:: self._stride]


def run_human_detection_comparison(waveshare_index: DatasetIndex, preprocessors: dict) -> dict:
    print("\n=== Human detection comparison (real bboxes) ===")
    ds = FrameLevelDataset(
        waveshare_index, class_filter=[PERSON_CLASS_ID], include_negative_frames=True,
    )
    registry = CheckpointRegistry(root=CHECKPOINTS)
    detector_loaders = {
        # ~4.4s/frame -> stride to keep every scene represented without a
        # multi-hour run; see _StridedBySessionDataset.
        "hog_svm": (lambda: registry.load(HOGSVMDetector, profile_name="Waveshare_26984"), 50),
        "mobilenet_ssd": (lambda: registry.load(MobileNetSSDDetector, profile_name="Waveshare_26984"), 1),
    }
    results = {}
    for det_name, (load_det, stride) in detector_loaders.items():
        eval_ds = ds if stride == 1 else _StridedBySessionDataset(ds, stride)
        print(f"  {det_name}: stride={stride} ({'full dataset' if stride == 1 else f'every {stride}th frame/scene'})")
        results[det_name] = {}
        for prep_name, prep in preprocessors.items():
            det = load_det()
            scenario_results = Trainer.evaluate_human_detection(
                det, eval_ds, preprocessor=prep, mode=prep_name, verbose=False,
            )
            total_correct = sum(r.confusion.correct for r in scenario_results)
            total_n = sum(r.confusion.total for r in scenario_results)
            print(f"  {det_name:<14} {prep_name:<12} {len(scenario_results)} scenes, "
                  f"aggregate correct={total_correct}/{total_n}")
            results[det_name][prep_name] = [
                {"scene": r.name, **r.confusion.__dict__} for r in scenario_results
            ]
    return results


def run_fire_detection_comparison(waveshare_index: DatasetIndex, preprocessors: dict) -> dict:
    print("\n=== Fire detection comparison (full waveshare dataset, real bboxes, real IoU) ===")
    ds = FireFrameDataset(waveshare_index)
    registry = CheckpointRegistry(root=CHECKPOINTS)
    results = {}

    print("  [baseline checkpoint, trained on raw-frame features -- NOT retrained per "
          "preprocessor; expected to be a poor/invalid comparison, kept for the record]")
    for prep_name, prep in preprocessors.items():
        det = registry.load(FireSVMDetector)
        scenario_results = Trainer.evaluate_fire_detection(
            det, ds, preprocessor=prep, mode=f"{prep_name}_baseline_ckpt", verbose=False,
        )
        total_correct = sum(r.confusion.correct for r in scenario_results)
        total_n = sum(r.confusion.total for r in scenario_results)
        mean_iou = sum(r.mean_iou for r in scenario_results) / len(scenario_results) if scenario_results else 0.0
        print(f"    {prep_name:<12} {len(scenario_results)} scenes, "
              f"aggregate correct={total_correct}/{total_n}, mean IoU={mean_iou:.1%}")
        results[f"{prep_name}_baseline_ckpt"] = [
            {"scene": r.name, "mean_iou": r.mean_iou, **r.confusion.__dict__} for r in scenario_results
        ]

    print("  [refit fresh per preprocessor -- the methodologically valid comparison "
          "for 'does this preprocessing help a fire classifier'; trained and evaluated "
          "on the same set, so treat as directional, not held-out generalization]")
    for prep_name, prep in preprocessors.items():
        proc_frames = [prep.predict(f) for f, _alert in ds]
        alerts = [alert for _f, alert in ds]
        det = FireSVMDetector()
        det.fit(proc_frames, alerts)
        scenario_results = Trainer.evaluate_fire_detection(
            det, ds, preprocessor=prep, mode=f"{prep_name}_refit", verbose=False,
        )
        total_correct = sum(r.confusion.correct for r in scenario_results)
        total_n = sum(r.confusion.total for r in scenario_results)
        mean_iou = sum(r.mean_iou for r in scenario_results) / len(scenario_results) if scenario_results else 0.0
        print(f"    {prep_name:<12} {len(scenario_results)} scenes, "
              f"aggregate correct={total_correct}/{total_n}, mean IoU={mean_iou:.1%}")
        results[f"{prep_name}_refit"] = [
            {"scene": r.name, "mean_iou": r.mean_iou, **r.confusion.__dict__} for r in scenario_results
        ]
    return results


def run_contact_detection_comparison(waveshare_index: DatasetIndex, preprocessors: dict) -> dict:
    print("\n=== Contact detection comparison (full waveshare dataset, real contact labels) ===")
    ds = ContactFrameDataset(waveshare_index)
    registry = CheckpointRegistry(root=CHECKPOINTS)
    results = {}
    for prep_name, prep in preprocessors.items():
        # ThermoX3DDetector.predict() doesn't take a preprocessor arg directly;
        # apply it per-frame via a tiny wrapper matching evaluate_contact_detection's
        # expectations (it calls detector.predict(frames) with no preprocessing hook
        # of its own for this dataset type, so wrap the detector instead).
        det = registry.load(ThermoX3DDetector, profile_name="Waveshare_26984")

        class _PreprocessedDetector:
            def __init__(self, inner, pre):
                self._inner = inner
                self._pre = pre

            def predict(self, frames):
                return self._inner.predict(tuple(self._pre.predict(f) for f in frames))

            def reset(self):
                if hasattr(self._inner, "reset"):
                    self._inner.reset()

        wrapped = _PreprocessedDetector(det, prep)
        scenario_results = Trainer.evaluate_contact_detection(wrapped, ds, mode=prep_name, verbose=False)
        total_correct = sum(r.confusion.correct for r in scenario_results)
        total_n = sum(r.confusion.total for r in scenario_results)
        print(f"  {prep_name:<12} {len(scenario_results)} scenes, aggregate correct={total_correct}/{total_n}")
        results[prep_name] = [
            {"scene": r.name, **r.confusion.__dict__} for r in scenario_results
        ]
    return results


def main() -> int:
    waveshare_index = DatasetIndex(DATA_ROOT / "waveshare_work", sensor_profile=WAVESHARE_26984, fps=8.0)
    preprocessors = build_preprocessors(waveshare_index)

    all_results = {
        "sbr": run_sbr_comparison(waveshare_index, preprocessors),
        "human_detection": run_human_detection_comparison(waveshare_index, preprocessors),
        "fire_detection": run_fire_detection_comparison(waveshare_index, preprocessors),
        "contact_detection": run_contact_detection_comparison(waveshare_index, preprocessors),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\nSaved to {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
