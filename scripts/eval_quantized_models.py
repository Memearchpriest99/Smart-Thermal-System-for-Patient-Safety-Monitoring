#!/usr/bin/env python3
"""Evaluate fp16/bf16 (torch, GPU) and int8 (ONNX Runtime dynamic
quantization) variants of every detector on the exact same held-out test
split as eval_all_detectors.py / eval_onnx_models.py.

Applicability, stated up front because it shapes what this script actually
runs (see thermal_algorithms.training.precision_shims module docstring for
the full reasoning):

  - MobileNetSSDDetector, MVSTGCNDetector, ThermoX3DDetector (torch, GPU):
    fp16, bf16, AND int8 (via their ONNX export).
  - FireSVMDetector, HOGSVMDetector (scikit-learn, CPU only): fp16/bf16 are
    NOT tested -- libsvm/liblinear's C implementation always computes in
    float64 on CPU, so there is no mixed-precision code path to engage and
    no GPU tensor-core benefit to gain; casting inputs would just add a
    no-op cast. int8 dynamic quantization of their ONNX-exported classifier
    graph IS tested, since that's a real ONNX Runtime optimization path.

Requires scripts/export_onnx_models.py to have been run first (int8 is
produced by quantizing those .onnx files).

Usage::

    python scripts/eval_quantized_models.py
    python scripts/eval_quantized_models.py --only HOGSVMDetector --formats int8
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
    describe_device,
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
from thermal_algorithms.training.precision_shims import CastModelShim  # noqa: E402

try:
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

DATA_ROOT = _REPO_ROOT.parent / "data"
ONNX_DIR = _REPO_ROOT.parent / "Weights" / "onnx"
INT8_DIR = _REPO_ROOT.parent / "Weights" / "onnx" / "int8"


def build_waveshare_test_split(waveshare_index: DatasetIndex, task: str) -> tuple[set[str], set[str]]:
    """Independent per-task split -- see thermal_algorithms.training.split.build_task_split."""
    return build_task_split(waveshare_index.labeled_sessions(), task=task)


def quantize_to_int8(onnx_path: str) -> str:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    INT8_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(onnx_path)
    dst = INT8_DIR / f"{src.stem}_int8.onnx"
    # DefaultTensorType works around a shape-inference gap the legacy
    # TorchScript ONNX exporter leaves on some Gemm/MatMul nodes (seen on
    # thermo_x3d_detector.onnx's CBAM-attention MLP) -- ORT's quantizer
    # can't otherwise determine that tensor's dtype and aborts. This is the
    # workaround its own error message documents.
    quantize_dynamic(
        model_input=str(src), model_output=str(dst), weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    return str(dst)


def attach_mvstgcn_inference_deps(det: MVSTGCNDetector, registry: CheckpointRegistry) -> None:
    """See scripts/eval_all_detectors.py -- same synthetic-homography +
    embedded-human-detector caveat applies here."""
    from examples.utils import make_synthetic_homographies

    det._homography = make_synthetic_homographies(det.sensor_profile)
    if _TORCH_OK:
        det._human_detector = registry.load(MobileNetSSDDetector, profile_name="Waveshare_26984")
    else:
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
    ap.add_argument("--out", default=str(_REPO_ROOT / "reports" / "full_corpus_eval_quantized.json"))
    ap.add_argument("--formats", nargs="*", default=["fp16", "bf16", "int8"])
    ap.add_argument(
        "--only", nargs="*", default=None,
        help="Restrict to these detector class names, to run the slow ones "
             "(HOGSVMDetector) as separate parallel jobs.",
    )
    ap.add_argument(
        "--extra-human-test-scenes", nargs="*", default=[],
        help="Additional waveshare_work scenes to union into the human-detection test "
             "set on top of build_task_split's normal draw -- e.g. 'empty_room', to "
             "include real negative frames. Mirrors eval_all_detectors.py's flag of the "
             "same name -- keep both in sync if this is used for the fp32 baseline.",
    )
    args = ap.parse_args()
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

    human_ds = FrameLevelDataset(
        waveshare_index, scenes=human_test_scenes, class_filter=[PERSON_CLASS_ID], include_negative_frames=True,
    )
    fire_ds = FireFrameDataset(waveshare_index, scenes=fire_test_scenes)
    contact_ds = ContactFrameDataset(waveshare_index, scenes=contact_test_scenes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save_partial() -> None:
        payload = {
            "fire_test_scenes": sorted(fire_test_scenes),
            "human_test_scenes": sorted(human_test_scenes),
            "contact_test_scenes": sorted(contact_test_scenes),
            "results": [r.to_dict() for r in results],
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def run_block(label: str, fn) -> None:
        """Run one (detector, format) block; a failure here (e.g. ORT's
        quantizer choking on one model's graph) must not take down every
        other block that already succeeded or still has to run -- and
        whatever succeeded so far is saved immediately, so a later crash
        (or an external kill, which has happened to long-running jobs in
        this session before) doesn't lose completed work."""
        try:
            r = fn()
        except Exception as e:
            print(f"  FAILED {label}: {e}")
            return
        results.append(r)
        print(r.to_dict())
        save_partial()

    # ---- int8 (all 5 detectors, via their ONNX export) ----
    if "int8" in args.formats:
        if wanted("FireSVMDetector") and "FireSVMDetector" in manifest:
            def _run():
                det = registry.load(FireSVMDetector, profile_name=None)
                int8_path = quantize_to_int8(manifest["FireSVMDetector"]["onnx_path"])
                det._scaler = _Identity()
                det._svm = OnnxFireSvmShim(int8_path, prefer_gpu=False)
                return evaluate_fire_timed(det, fire_ds, preprocessor=preprocessor, variant="int8",
                                           device=describe_device(force="ONNX Runtime / CPUExecutionProvider"))
            run_block("FireSVMDetector int8", _run)

        if wanted("HOGSVMDetector") and "HOGSVMDetector" in manifest:
            def _run():
                det = registry.load(HOGSVMDetector, profile_name="Waveshare_26984")
                int8_path = quantize_to_int8(manifest["HOGSVMDetector"]["onnx_path"])
                det._svm = OnnxLinearSvcShim(int8_path, prefer_gpu=False)
                return evaluate_human_timed(det, human_ds, preprocessor=preprocessor, variant="int8",
                                            device=describe_device(force="ONNX Runtime / CPUExecutionProvider"))
            run_block("HOGSVMDetector int8", _run)

        if _TORCH_OK and wanted("MobileNetSSDDetector") and "MobileNetSSDDetector" in manifest:
            def _run():
                det = registry.load(MobileNetSSDDetector, profile_name="Waveshare_26984")
                anchors = det._model.anchors
                int8_path = quantize_to_int8(manifest["MobileNetSSDDetector"]["onnx_path"])
                det._model = OnnxTorchShim(int8_path, ["frame"], prefer_gpu=False, anchors=anchors)
                return evaluate_human_timed(det, human_ds, preprocessor=preprocessor, variant="int8",
                                            device=describe_device(force="ONNX Runtime / CPUExecutionProvider"))
            run_block("MobileNetSSDDetector int8", _run)

        if _TORCH_OK and wanted("MVSTGCNDetector") and "MVSTGCNDetector" in manifest:
            def _run():
                det = registry.load(MVSTGCNDetector, profile_name="Waveshare_26984")
                attach_mvstgcn_inference_deps(det, registry)
                int8_path = quantize_to_int8(manifest["MVSTGCNDetector"]["onnx_path"])
                det._model = OnnxTorchShim(int8_path, ["node_features", "positions", "masks"], prefer_gpu=False)
                wrapped = _PreprocessedContactDetector(det, preprocessor)
                return evaluate_contact_timed(wrapped, contact_ds, variant="int8", detector_name="MVSTGCNDetector",
                                              device=describe_device(force="ONNX Runtime / CPUExecutionProvider"))
            run_block("MVSTGCNDetector int8", _run)

        if _TORCH_OK and wanted("ThermoX3DDetector") and "ThermoX3DDetector" in manifest:
            def _run():
                det = registry.load(ThermoX3DDetector, profile_name="Waveshare_26984")
                int8_path = quantize_to_int8(manifest["ThermoX3DDetector"]["onnx_path"])
                det._model = OnnxTorchShim(int8_path, ["volume"], prefer_gpu=False)
                wrapped = _PreprocessedContactDetector(det, preprocessor)
                return evaluate_contact_timed(wrapped, contact_ds, variant="int8", detector_name="ThermoX3DDetector",
                                              device=describe_device(force="ONNX Runtime / CPUExecutionProvider"))
            run_block("ThermoX3DDetector int8", _run)

    # ---- fp16 / bf16 (torch detectors only, GPU) ----
    if _TORCH_OK:
        import torch

        for fmt, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            if fmt not in args.formats:
                continue

            if wanted("MobileNetSSDDetector"):
                def _run():
                    det = registry.load(MobileNetSSDDetector, profile_name="Waveshare_26984")
                    orig_model = det._model
                    det._model = CastModelShim(orig_model, dtype, anchors=orig_model.anchors)
                    return evaluate_human_timed(det, human_ds, preprocessor=preprocessor, variant=fmt,
                                                device=describe_device(det))
                run_block(f"MobileNetSSDDetector {fmt}", _run)

            if wanted("MVSTGCNDetector"):
                def _run():
                    det = registry.load(MVSTGCNDetector, profile_name="Waveshare_26984")
                    attach_mvstgcn_inference_deps(det, registry)
                    det._model = CastModelShim(det._model, dtype)
                    wrapped = _PreprocessedContactDetector(det, preprocessor)
                    return evaluate_contact_timed(wrapped, contact_ds, variant=fmt, detector_name="MVSTGCNDetector",
                                                  device=describe_device(det))
                run_block(f"MVSTGCNDetector {fmt}", _run)

            if wanted("ThermoX3DDetector"):
                def _run():
                    det = registry.load(ThermoX3DDetector, profile_name="Waveshare_26984")
                    det._model = CastModelShim(det._model, dtype)
                    wrapped = _PreprocessedContactDetector(det, preprocessor)
                    return evaluate_contact_timed(wrapped, contact_ds, variant=fmt, detector_name="ThermoX3DDetector",
                                                  device=describe_device(det))
                run_block(f"ThermoX3DDetector {fmt}", _run)
    else:
        print("  SKIP fp16/bf16: torch not available")

    save_partial()
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
