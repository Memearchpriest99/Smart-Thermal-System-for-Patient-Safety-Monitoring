#!/usr/bin/env python3
"""Export every trained detector to ONNX, saved under Weights/onnx/.

Scope note (documented for the report, not a bug): only the *learned*
sub-component of each detector is traced into ONNX. Classical-CV
pre/post-processing that surrounds it -- Otsu segmentation + blob extraction
(FireSVMDetector), the HOG descriptor + sliding-window pyramid (HOGSVMDetector),
NMS/box decoding (MobileNetSSDDetector), and the detection+Kalman-tracking+
homography+graph-construction front-end (MVSTGCNDetector) -- is deterministic
NumPy/OpenCV code, not a neural-net or sklearn estimator, and isn't part of
any framework's ONNX graph by construction. Tracing it would require manually
re-implementing each op as a custom ONNX node for zero benefit (it's already
fast, exact, and framework-independent):

    FireSVMDetector   -> StandardScaler + SVC(probability=True), via skl2onnx
    HOGSVMDetector     -> LinearSVC decision function, via skl2onnx
    MobileNetSSDDetector -> full MicroMobileNetSSD backbone+heads, torch.onnx.export
    MVSTGCNDetector    -> _ThermalSTGCN graph-conv body only, at a FIXED
                          representative actor count (self._max_actors slots,
                          mask-padded exactly as real inference does) --
                          torch.onnx.export
    ThermoX3DDetector  -> full 3D-CNN body, torch.onnx.export

Usage::

    python scripts/export_onnx_models.py
    python scripts/export_onnx_models.py --checkpoints checkpoints_baseline_waveshare
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

import numpy as np  # noqa: E402

from thermal_algorithms.core.checkpoints import CheckpointRegistry  # noqa: E402
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector  # noqa: E402
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector  # noqa: E402

DEFAULT_CHECKPOINTS = _REPO_ROOT / "checkpoints_full_corpus"
ONNX_OUT = _REPO_ROOT.parent / "Weights" / "onnx"


def _onnx_out_dir(checkpoints_dir) -> Path:
    """Where this export's .onnx files go.

    Exports from the DEFAULT checkpoint root keep the flat historical paths
    (Weights/onnx/<detector>.onnx) so existing manifests stay valid. Any other
    root gets its own subdirectory, because the filenames are otherwise
    identical: exporting checkpoints_balanced would silently overwrite the
    checkpoints_full_corpus .onnx that the section-2 manifest still points at,
    and the next section-2 ONNX evaluation would quietly measure the wrong
    weights while reporting them as natural-ratio results.
    """
    name = Path(checkpoints_dir).name
    if name == DEFAULT_CHECKPOINTS.name:
        return ONNX_OUT
    return ONNX_OUT / name


def export_fire_svm(registry: CheckpointRegistry, out_dir: Path) -> dict | None:
    det = registry.load(FireSVMDetector, profile_name=None)
    if det._svm is None or det._scaler is None:
        print("  SKIP FireSVMDetector: not fitted")
        return None

    from sklearn.pipeline import Pipeline
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    pipeline = Pipeline([("scaler", det._scaler), ("svm", det._svm)])
    n_features = det._scaler.mean_.shape[0]
    dummy = np.zeros((1, n_features), dtype=np.float32)

    onnx_model = to_onnx(
        pipeline, dummy,
        options={id(det._svm): {"zipmap": False}},
        target_opset=17,
    )
    out_path = out_dir / "fire_svm_detector.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())
    print(f"  FireSVMDetector -> {out_path}")
    return {
        "onnx_path": str(out_path),
        "input_shape": [1, n_features],
        "input_dtype": "float32",
        "feature_order": ["max_temp", "mean_temp", "std_temp", "area", "skewness", "kurtosis"],
        "scope": "StandardScaler + SVC(probability=True) only -- Otsu segmentation "
                 "and blob feature extraction are classical CV code, not exported.",
    }


def export_hog_svm(registry: CheckpointRegistry, out_dir: Path) -> dict | None:
    det = registry.load(HOGSVMDetector, profile_name="Waveshare_26984")
    if det._svm is None:
        print("  SKIP HOGSVMDetector: not fitted")
        return None

    from skl2onnx import to_onnx

    n_features = det._n_features
    dummy = np.zeros((1, n_features), dtype=np.float32)
    onnx_model = to_onnx(det._svm, dummy, target_opset=17)
    out_path = out_dir / "hog_svm_detector.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())
    print(f"  HOGSVMDetector -> {out_path}")
    return {
        "onnx_path": str(out_path),
        "input_shape": [1, n_features],
        "input_dtype": "float32",
        "scope": "LinearSVC decision function only -- the HOG descriptor and "
                 "multi-scale sliding-window pyramid are classical CV code, not exported.",
    }


def export_mobilenet_ssd(registry: CheckpointRegistry, out_dir: Path) -> dict | None:
    import torch
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector

    det = registry.load(MobileNetSSDDetector, profile_name="Waveshare_26984")
    if det._model is None:
        print("  SKIP MobileNetSSDDetector: not fitted")
        return None

    model = det._model.to("cpu").eval()
    h, w = det.sensor_profile.resolution[1], det.sensor_profile.resolution[0]
    dummy = torch.zeros(1, 1, h, w, dtype=torch.float32)

    out_path = out_dir / "mobilenet_ssd_detector.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["frame"], output_names=["box_preds", "cls_preds"],
        opset_version=17, dynamo=False,
    )
    print(f"  MobileNetSSDDetector -> {out_path}")
    return {
        "onnx_path": str(out_path),
        "input_shape": [1, 1, h, w],
        "input_dtype": "float32",
        "outputs": ["box_preds", "cls_preds"],
        "scope": "Full backbone + SSD heads. Anchor decoding + NMS stay in "
                 "Python (thermal_algorithms.human_detection.mobilenet_ssd_anchors) "
                 "-- apply identically to torch and ONNX raw outputs for a fair comparison.",
    }


def export_mv_stgcn(registry: CheckpointRegistry, out_dir: Path) -> dict | None:
    import torch
    from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector

    det = registry.load(MVSTGCNDetector, profile_name="Waveshare_26984")
    if det._model is None:
        print("  SKIP MVSTGCNDetector: not fitted")
        return None

    model = det._model.to("cpu").eval()
    T, N, D = det._T, det._max_actors, det._node_feat_dim
    feat = torch.zeros(1, T, N, D, dtype=torch.float32)
    pos = torch.zeros(1, T, N, 2, dtype=torch.float32)
    mask = torch.ones(1, T, N, dtype=torch.float32)

    out_path = out_dir / "mv_stgcn_detector.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (feat, pos, mask), str(out_path),
        input_names=["node_features", "positions", "masks"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )
    print(f"  MVSTGCNDetector -> {out_path}")
    return {
        "onnx_path": str(out_path),
        "input_shapes": {"node_features": [1, T, N, D], "positions": [1, T, N, 2], "masks": [1, T, N]},
        "input_dtype": "float32",
        "outputs": ["logits (2-class, softmax for probability)"],
        "scope": f"Graph-conv body only, at a FIXED representative actor count "
                 f"(max_actors={N}, mask-padded exactly as real inference does). "
                 f"Detection + Kalman tracking + homography projection + graph "
                 f"construction (thermal_algorithms.contact_detection.mv_stgcn / "
                 f"multi_view/*) stay in Python -- not neural-net ops.",
    }


def export_thermo_x3d(registry: CheckpointRegistry, out_dir: Path) -> dict | None:
    import torch
    from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector

    det = registry.load(ThermoX3DDetector, profile_name="Waveshare_26984")
    if det._model is None:
        print("  SKIP ThermoX3DDetector: not fitted")
        return None

    model = det._model.to("cpu").eval()
    T = det._T
    h, w = det.sensor_profile.resolution[1], det.sensor_profile.resolution[0]
    dummy = torch.zeros(1, 3, T, h, w, dtype=torch.float32)

    out_path = out_dir / "thermo_x3d_detector.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["volume"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )
    print(f"  ThermoX3DDetector -> {out_path}")
    return {
        "onnx_path": str(out_path),
        "input_shape": [1, 3, T, h, w],
        "input_dtype": "float32",
        "outputs": ["logits (2-class, softmax for probability)"],
        "scope": "Full 3D-CNN body. Per-camera frame buffering + global "
                 "normalization stay in Python.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default=str(DEFAULT_CHECKPOINTS))
    ap.add_argument("--out-manifest", default=str(_REPO_ROOT / "reports" / "onnx_export_manifest.json"))
    args = ap.parse_args()

    registry = CheckpointRegistry(root=args.checkpoints)
    out_dir = _onnx_out_dir(args.checkpoints)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ONNX output dir: {out_dir}")
    manifest: dict[str, dict] = {}
    # Export failures are recorded, not just printed: a detector that cannot be
    # exported needs to show up in the report as "not convertible, because X"
    # rather than silently having no ONNX/quantized rows.
    failures: dict[str, str] = {}

    print("=== Exporting FireSVMDetector ===")
    try:
        r = export_fire_svm(registry, out_dir)
        if r:
            manifest["FireSVMDetector"] = r
    except Exception as e:
        print(f"  FAILED FireSVMDetector: {type(e).__name__}: {e}")
        failures["FireSVMDetector"] = f"{type(e).__name__}: {e}"

    print("=== Exporting HOGSVMDetector ===")
    try:
        r = export_hog_svm(registry, out_dir)
        if r:
            manifest["HOGSVMDetector"] = r
    except Exception as e:
        print(f"  FAILED HOGSVMDetector: {type(e).__name__}: {e}")
        failures["HOGSVMDetector"] = f"{type(e).__name__}: {e}"

    print("=== Exporting MobileNetSSDDetector ===")
    try:
        r = export_mobilenet_ssd(registry, out_dir)
        if r:
            manifest["MobileNetSSDDetector"] = r
    except Exception as e:
        print(f"  FAILED MobileNetSSDDetector: {type(e).__name__}: {e}")
        failures["MobileNetSSDDetector"] = f"{type(e).__name__}: {e}"

    print("=== Exporting MVSTGCNDetector ===")
    try:
        r = export_mv_stgcn(registry, out_dir)
        if r:
            manifest["MVSTGCNDetector"] = r
    except Exception as e:
        print(f"  FAILED MVSTGCNDetector: {type(e).__name__}: {e}")
        failures["MVSTGCNDetector"] = f"{type(e).__name__}: {e}"

    print("=== Exporting ThermoX3DDetector ===")
    try:
        r = export_thermo_x3d(registry, out_dir)
        if r:
            manifest["ThermoX3DDetector"] = r
    except Exception as e:
        # Print the full traceback, not just str(e). A bare one-line message
        # here is how ThermoX3DDetector silently vanished from the manifest
        # for weeks: every downstream `if "ThermoX3DDetector" in manifest`
        # guard then skipped it, so the report showed no ONNX/int8 rows for it
        # and looked like an oversight rather than a failure.
        import traceback
        print(f"  FAILED ThermoX3DDetector: {type(e).__name__}: {e}")
        traceback.print_exc()
        failures["ThermoX3DDetector"] = f"{type(e).__name__}: {e}"

    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if failures:
        manifest["_export_failures"] = failures
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nManifest -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
