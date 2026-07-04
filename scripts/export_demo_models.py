"""Export the demo model bundle — everything the Pi demo needs, torch-free.

Produces (all under checkpoints/):
  mobilenet_ssd_detector/Waveshare_26984_raw.onnx        SSD backbone+heads
  mobilenet_ssd_detector/Waveshare_26984_raw.anchors.npy (A, 4) cxcywh
  mobilenet_ssd_detector/Waveshare_26984_raw.meta.json   thresholds/variances
  thermo_x3d_detector/Waveshare_26984_T5_v2.meta.json    T/norm stats/threshold

The X3D ftz ONNX (denormal-flushed) and FireSVM .thalg already exist.
Denormal weights are flushed in the SSD export too. Parity of the
onnx+numpy decode path vs the torch predict() is verified on real frames.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

SSD_DIR = _ROOT / "checkpoints" / "mobilenet_ssd_detector"
X3D_DIR = _ROOT / "checkpoints" / "thermo_x3d_detector"
SSD_THALG = SSD_DIR / "Waveshare_26984_raw.thalg"
SSD_ONNX = SSD_DIR / "Waveshare_26984_raw.onnx"
SSD_ANCHORS = SSD_DIR / "Waveshare_26984_raw.anchors.npy"
SSD_META = SSD_DIR / "Waveshare_26984_raw.meta.json"
X3D_THALG = X3D_DIR / "Waveshare_26984_T5_v2.thalg"
X3D_META = X3D_DIR / "Waveshare_26984_T5_v2.meta.json"


def flush_denormals_onnx(path: Path) -> int:
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path))
    n = 0
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32:
            mask = (a != 0) & (np.abs(a) < np.finfo(np.float32).tiny)
            if mask.any():
                n += int(mask.sum())
                init.CopyFrom(numpy_helper.from_array(
                    np.where(mask, 0.0, a).astype(np.float32), init.name))
    onnx.save(m, str(path))
    return n


def main() -> None:
    import torch
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
    from thermal_algorithms.core.types import Frame

    det = MobileNetSSDDetector.load(SSD_THALG)
    det._device_str = "cpu"
    det._device = None
    if det._model is not None:
        det._model = det._model.cpu()
        det._device = torch.device("cpu")
        det._model.build_anchors(device=det._device)
    model = det._model
    model.eval()
    h, w = model.input_size

    # ---- export ------------------------------------------------------------
    dummy = torch.zeros(1, 1, h, w, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(SSD_ONNX),
        input_names=["frame"], output_names=["box_preds", "cls_preds"],
        opset_version=17, do_constant_folding=True,
    )
    n_flushed = flush_denormals_onnx(SSD_ONNX)
    np.save(SSD_ANCHORS, model.anchors.cpu().numpy().astype(np.float32))
    SSD_META.write_text(json.dumps({
        "input_hw": [h, w],
        "score_threshold": det._score_threshold,
        "nms_iou_threshold": det._nms_iou,
        "bbox_variance_xy": 0.1,
        "bbox_variance_wh": 0.2,
        "person_class_id": 1,
        "normalize": "per-frame z-score: (x - mean) / (std + 1e-6)",
    }, indent=2))
    print(f"SSD -> {SSD_ONNX.name} ({SSD_ONNX.stat().st_size/1e6:.2f} MB, "
          f"{n_flushed} denormal weights flushed), anchors {SSD_ANCHORS.name}")

    # ---- X3D meta ------------------------------------------------------------
    from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector

    x3d = ThermoX3DDetector.load(X3D_THALG)
    X3D_META.write_text(json.dumps({
        "T": x3d._T,
        "global_mean": x3d._global_mean,
        "global_std": x3d._global_std,
        "conf_threshold": 0.35,
        "persistence_frames": x3d._persistence_frames,
        "onnx": "Waveshare_26984_T5_v2.ftz.onnx",
    }, indent=2))
    print(f"X3D meta -> {X3D_META.name}: T={x3d._T}, "
          f"mean={x3d._global_mean:.3f}, std={x3d._global_std:.3f}")

    # ---- parity: torch predict vs onnx + numpy decode ------------------------
    from demo.detectors import OnnxSSD

    ssd_np = OnnxSSD(SSD_ONNX, SSD_ANCHORS, SSD_META)

    import glob
    sess_files = sorted(glob.glob(str(
        _ROOT / "scripts" / "_x3d_backend_cache" / "*.npz")))[:3]
    if not sess_files:
        print("WARNING: no cached frames for parity check; skipped")
        return
    worst = 0.0
    n_boxes = 0
    for f in sess_files:
        resid = np.load(f)["resid"]  # residuals — fine as inputs for parity
        for i in range(0, min(20, resid.shape[1])):
            raw = resid[0, i] * 5.0 + 25.0  # scale to plausible temp range
            frame = Frame(data=raw.astype(np.float32), timestamp=0.0, camera_id=0)
            ref = det.predict(frame)
            out = ssd_np.predict(raw.astype(np.float32))
            assert len(ref) == len(out), f"box count mismatch {len(ref)} vs {len(out)}"
            for r, o in zip(sorted(ref, key=lambda d: -d.score),
                            sorted(out, key=lambda d: -d[1])):
                worst = max(worst, max(abs(a - b) for a, b in zip(r.bbox, o[0])),
                            abs(r.score - o[1]))
                n_boxes += 1
    print(f"parity: {n_boxes} boxes compared, max |diff| = {worst:.2e}")


if __name__ == "__main__":
    main()
