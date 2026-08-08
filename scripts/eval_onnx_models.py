#!/usr/bin/env python3
"""Evaluate the ONNX-exported models on the exact same held-out test split
as scripts/eval_all_detectors.py, by swapping only each detector's learned
sub-component for an ONNX Runtime shim (thermal_algorithms.training.onnx_shims)
-- every other line of pre/post-processing runs completely unmodified, so any
metric or latency delta vs the fp32 baseline is attributable to the ONNX
conversion itself, not to a different code path.

Requires scripts/export_onnx_models.py to have been run first (reads
reports/onnx_export_manifest.json for the .onnx paths).

Usage::

    python scripts/export_onnx_models.py --checkpoints checkpoints_full_corpus
    python scripts/eval_onnx_models.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from thermal_algorithms.core.checkpoints import CheckpointRegistry  # noqa: E402
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984  # noqa: E402
from thermal_algorithms.preprocessing.global_norm_pipeline import GlobalNormPreprocessor  # noqa: E402
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector  # noqa: E402
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector  # noqa: E402
from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector  # noqa: E402
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector  # noqa: E402
from thermal_algorithms.training import (  # noqa: E402
    ContactFrameDataset,
    DatasetIndex,
    FireFrameDataset,
    FrameLevelDataset,
    PERSON_CLASS_ID,
    build_task_split,
    evaluate_contact_timed,
    evaluate_fire_timed,
    evaluate_human_timed,
)
from thermal_algorithms.training.onnx_shims import (  # noqa: E402
    OnnxFireSvmShim,
    OnnxLinearSvcShim,
    OnnxTorchShim,
    _Identity,
)

try:
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

DATA_ROOT = _REPO_ROOT.parent / "data"


def build_waveshare_test_split(waveshare_index: DatasetIndex, task: str) -> tuple[set[str], set[str]]:
    """Must exactly mirror scripts/train_full_corpus.py:build_waveshare_split
    -- independent per-task split, not one split shared across fire/human/
    contact. See thermal_algorithms.training.split.build_task_split."""
    return build_task_split(waveshare_index.labeled_sessions(), task=task)


def attach_mvstgcn_inference_deps(det: MVSTGCNDetector, registry: CheckpointRegistry) -> None:
    """See scripts/eval_all_detectors.py:attach_mvstgcn_inference_deps -- same
    synthetic-homography + embedded-human-detector caveat applies here."""
    from examples.utils import make_synthetic_homographies

    det._homography = make_synthetic_homographies(det.sensor_profile)
    try:
        det._human_detector = registry.load(MobileNetSSDDetector, profile_name="Waveshare_26984")
    except Exception:
        det._human_detector = registry.load(HOGSVMDetector, profile_name="Waveshare_26984")


class _PreprocessedContactDetector:
    def __init__(self, inner, pre) -> None:
        self._inner = inner
        self._pre = pre

    def predict(self, frames):
        return self._inner.predict(tuple(self._pre.predict(f) for f in frames))

    def reset(self) -> None:
        if hasattr(self._inner, "reset"):
            self._inner.reset()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--checkpoints", default=str(_REPO_ROOT / "checkpoints_full_corpus"))
    ap.add_argument("--manifest", default=str(_REPO_ROOT / "reports" / "onnx_export_manifest.json"))
    ap.add_argument("--out", default=str(_REPO_ROOT / "reports" / "full_corpus_eval_onnx.json"))
    ap.add_argument("--no-gpu", action="store_true", help="Force ONNX Runtime CPUExecutionProvider")
    ap.add_argument(
        "--only", nargs="*", default=None,
        help="Restrict to these detector class names (e.g. --only HOGSVMDetector), "
             "to run the slow ones as separate parallel jobs.",
    )
    ap.add_argument(
        "--extra-human-test-scenes", nargs="*", default=[],
        help="Additional waveshare_work scenes to union into the human-detection test "
             "set on top of build_task_split's normal draw -- e.g. 'empty_room', to "
             "include real negative frames. Mirrors eval_all_detectors.py's flag of the "
             "same name -- keep both in sync if this is used for the fp32 baseline.",
    )
    args = ap.parse_args()
    prefer_gpu = not args.no_gpu
    only = set(args.only) if args.only else None

    def wanted(name: str) -> bool:
        return only is None or name in only

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    data_root = Path(args.data_root)
    waveshare_index = DatasetIndex(data_root / "waveshare_work", sensor_profile=WAVESHARE_26984, fps=8.0)
    _, fire_test_scenes = build_waveshare_test_split(waveshare_index, "fire")
    _, human_test_scenes = build_waveshare_test_split(waveshare_index, "human")
    _, contact_test_scenes = build_waveshare_test_split(waveshare_index, "contact")
    if args.extra_human_test_scenes:
        human_test_scenes = human_test_scenes | set(args.extra_human_test_scenes)
    print(f"Fire test scenes: {sorted(fire_test_scenes)}")
    print(f"Human test scenes: {sorted(human_test_scenes)}")
    print(f"Contact test scenes: {sorted(contact_test_scenes)}")

    preprocessor = GlobalNormPreprocessor(WAVESHARE_26984)
    preprocessor.fit()

    registry = CheckpointRegistry(root=args.checkpoints)
    results = []
    providers_used: dict[str, list[str]] = {}

    def report_providers(name: str, shim) -> None:
        providers_used[name] = list(shim.providers)
        print(f"  [{name}] ONNX Runtime providers: {shim.providers}")

    # --- Fire ---
    if wanted("FireSVMDetector") and "FireSVMDetector" in manifest:
        det = registry.load(FireSVMDetector, profile_name=None)
        shim = OnnxFireSvmShim(manifest["FireSVMDetector"]["onnx_path"], prefer_gpu=prefer_gpu)
        det._scaler = _Identity()
        det._svm = shim
        report_providers("FireSVMDetector", shim)
        fire_ds = FireFrameDataset(waveshare_index, scenes=fire_test_scenes)
        r = evaluate_fire_timed(det, fire_ds, preprocessor=preprocessor, variant="onnx_fp32")
        results.append(r)
        print(r.to_dict())

    human_ds = FrameLevelDataset(
        waveshare_index, scenes=human_test_scenes, class_filter=[PERSON_CLASS_ID], include_negative_frames=True,
    )

    # --- Human: HOG-SVM ---
    if wanted("HOGSVMDetector") and "HOGSVMDetector" in manifest:
        det = registry.load(HOGSVMDetector, profile_name="Waveshare_26984")
        shim = OnnxLinearSvcShim(manifest["HOGSVMDetector"]["onnx_path"], prefer_gpu=prefer_gpu)
        det._svm = shim
        report_providers("HOGSVMDetector", shim)
        r = evaluate_human_timed(det, human_ds, preprocessor=preprocessor, variant="onnx_fp32")
        results.append(r)
        print(r.to_dict())

    # --- Human: MobileNet-SSD ---
    if wanted("MobileNetSSDDetector") and _TORCH_OK and "MobileNetSSDDetector" in manifest:
        det = registry.load(MobileNetSSDDetector, profile_name="Waveshare_26984")
        anchors = det._model.anchors
        shim = OnnxTorchShim(
            manifest["MobileNetSSDDetector"]["onnx_path"], ["frame"], prefer_gpu=prefer_gpu, anchors=anchors,
        )
        det._model = shim
        report_providers("MobileNetSSDDetector", shim)
        r = evaluate_human_timed(det, human_ds, preprocessor=preprocessor, variant="onnx_fp32")
        results.append(r)
        print(r.to_dict())

    # --- Contact ---
    contact_ds = ContactFrameDataset(waveshare_index, scenes=contact_test_scenes)
    if wanted("MVSTGCNDetector") and _TORCH_OK and "MVSTGCNDetector" in manifest:
        det = registry.load(MVSTGCNDetector, profile_name="Waveshare_26984")
        attach_mvstgcn_inference_deps(det, registry)
        shim = OnnxTorchShim(
            manifest["MVSTGCNDetector"]["onnx_path"],
            ["node_features", "positions", "masks"],
            prefer_gpu=prefer_gpu,
        )
        det._model = shim
        report_providers("MVSTGCNDetector", shim)
        wrapped = _PreprocessedContactDetector(det, preprocessor)
        r = evaluate_contact_timed(wrapped, contact_ds, variant="onnx_fp32", detector_name="MVSTGCNDetector")
        results.append(r)
        print(r.to_dict())

    if wanted("ThermoX3DDetector") and _TORCH_OK and "ThermoX3DDetector" in manifest:
        det = registry.load(ThermoX3DDetector, profile_name="Waveshare_26984")
        shim = OnnxTorchShim(manifest["ThermoX3DDetector"]["onnx_path"], ["volume"], prefer_gpu=prefer_gpu)
        det._model = shim
        report_providers("ThermoX3DDetector", shim)
        wrapped = _PreprocessedContactDetector(det, preprocessor)
        r = evaluate_contact_timed(wrapped, contact_ds, variant="onnx_fp32", detector_name="ThermoX3DDetector")
        results.append(r)
        print(r.to_dict())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fire_test_scenes": sorted(fire_test_scenes),
        "human_test_scenes": sorted(human_test_scenes),
        "contact_test_scenes": sorted(contact_test_scenes),
        "onnxruntime_providers": providers_used,
        "results": [r.to_dict() for r in results],
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
