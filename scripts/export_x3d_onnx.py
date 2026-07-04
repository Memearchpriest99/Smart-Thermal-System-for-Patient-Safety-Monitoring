"""Export Thermo-X3D T5v2 to ONNX, quantize to int8, benchmark all backends.

Steps:
  1. Load checkpoints/thermo_x3d_detector/Waveshare_26984_T5_v2.thalg (CPU).
  2. Export to ONNX  -> Waveshare_26984_T5_v2.onnx
  3. Parity check torch vs onnxruntime fp32 on real p25-background residual
     volumes (max |dprob|).
  4. Static int8 quantization calibrated on real volumes
     -> Waveshare_26984_T5_v2.int8.onnx
  5. CPU runtime benchmark: torch fp32 / ORT fp32 / ORT int8.

Outputs reports/x3d_onnx_runtime.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_waveshare_contact as geo
from thermal_algorithms.core.types import Frame
from thermal_algorithms.preprocessing import TatenoPipeline
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.training import DatasetIndex

CKPT_DIR = _ROOT / "checkpoints" / "thermo_x3d_detector"
TORCH_CKPT = CKPT_DIR / "Waveshare_26984_T5_v2.thalg"
ONNX_FP32 = CKPT_DIR / "Waveshare_26984_T5_v2.onnx"
ONNX_INT8 = CKPT_DIR / "Waveshare_26984_T5_v2.int8.onnx"
OUT_JSON = _ROOT / "reports" / "x3d_onnx_runtime.json"

BG_PCT = 25.0
CALIB_SCENES = ["2ppl_fight", "2ppl_hug", "3pplhedroncolider", "the_more_the_merrier"]
N_CALIB = 120       # calibration volumes for int8
N_PARITY = 50       # volumes for torch-vs-onnx parity
N_BENCH = 50        # timed forwards per backend


def build_volumes(det, n_max: int) -> np.ndarray:
    """Real normalized (3, T, H, W) volumes via the T5_v2 eval protocol."""
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)
    T = det._T
    vols = []
    for scene in CALIB_SCENES:
        try:
            s = idx.find(scene)
        except Exception:
            continue
        fis = sorted(fi for fi in labels.get(scene, {}) if 0 <= fi < s.n_frames)
        if len(fis) < T:
            continue
        pre = {}
        for c, ch in enumerate(geo.CHANNELS):
            stack = np.stack([geo.get_frame(s, ch, fi).data for fi in fis], 0)
            bg = np.percentile(stack, BG_PCT, axis=0)
            pre[c] = TatenoPipeline(geo.PROFILE).fit(
                [Frame(data=bg, timestamp=0.0, camera_id=ch)]
            )
        resid = [
            [pre[c].predict(geo.get_frame(s, ch, fi)).data.astype(np.float32)
             for fi in fis]
            for c, ch in enumerate(geo.CHANNELS)
        ]
        for start in range(0, len(fis) - T + 1):
            vol = np.stack(
                [np.stack([det._normalise(resid[c][start + t]) for t in range(T)], 0)
                 for c in range(3)], 0)
            vols.append(vol.astype(np.float32))
            if len(vols) >= n_max:
                return np.stack(vols, 0)
    return np.stack(vols, 0)


def bench(fn, vols, n: int) -> dict:
    for v in vols[:5]:
        fn(v)
    times = []
    for i in range(n):
        v = vols[i % len(vols)]
        t0 = time.perf_counter()
        fn(v)
        times.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(times)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95))}


def main() -> None:
    import torch

    det = ThermoX3DDetector.load(TORCH_CKPT)
    det._device_str = "cpu"
    det._device = None
    if det._model is not None:
        det._model = det._model.cpu()
        det._device = torch.device("cpu")
    model, device = det._get_model()
    model.eval()
    T, H, W = det._T, det._input_h, det._input_w
    print(f"loaded T5_v2: T={T}, input {H}x{W}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # ---- 1. Export ONNX ----------------------------------------------------
    dummy = torch.zeros(1, 3, T, H, W, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(ONNX_FP32),
        input_names=["volume"], output_names=["logits"],
        opset_version=17, do_constant_folding=True,
    )
    print(f"exported -> {ONNX_FP32.relative_to(_ROOT)} "
          f"({ONNX_FP32.stat().st_size / 1e6:.2f} MB)")

    # ---- 2. Real volumes -----------------------------------------------------
    print("building real calibration/parity volumes (p25 backgrounds)...")
    vols = build_volumes(det, max(N_CALIB, N_PARITY + N_BENCH))
    print(f"  {len(vols)} volumes of shape {vols.shape[1:]}")

    # ---- 3. Parity torch vs ORT fp32 ---------------------------------------
    import onnxruntime as ort

    so = ort.SessionOptions()
    sess_fp32 = ort.InferenceSession(str(ONNX_FP32), so,
                                     providers=["CPUExecutionProvider"])

    def torch_prob(v):
        with torch.no_grad():
            logits = model(torch.from_numpy(v).unsqueeze(0))
            return float(torch.softmax(logits, dim=1)[0, 1])

    def ort_prob(sess):
        def run(v):
            logits = sess.run(None, {"volume": v[None]})[0][0]
            e = np.exp(logits - logits.max())
            return float((e / e.sum())[1])
        return run

    fp32_run = ort_prob(sess_fp32)
    diffs = [abs(torch_prob(v) - fp32_run(v)) for v in vols[:N_PARITY]]
    print(f"parity torch vs ORT fp32: max |dprob| = {max(diffs):.2e}")

    # ---- 4. Int8 quantization ------------------------------------------------
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre_path = ONNX_FP32.with_suffix(".preproc.onnx")
    quant_pre_process(str(ONNX_FP32), str(pre_path))

    class VolReader(CalibrationDataReader):
        def __init__(self, data):
            self._it = iter(data)

        def get_next(self):
            v = next(self._it, None)
            return None if v is None else {"volume": v[None]}

    quantize_static(
        str(pre_path), str(ONNX_INT8), VolReader(vols[:N_CALIB]),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    pre_path.unlink(missing_ok=True)
    print(f"quantized -> {ONNX_INT8.relative_to(_ROOT)} "
          f"({ONNX_INT8.stat().st_size / 1e6:.2f} MB)")

    sess_int8 = ort.InferenceSession(str(ONNX_INT8), so,
                                     providers=["CPUExecutionProvider"])
    int8_run = ort_prob(sess_int8)
    diffs8 = [abs(torch_prob(v) - int8_run(v)) for v in vols[:N_PARITY]]
    print(f"parity torch vs ORT int8: max |dprob| = {max(diffs8):.3f}, "
          f"mean = {np.mean(diffs8):.4f}")

    # ---- 5. Runtime benchmark -------------------------------------------------
    results = {}
    bench_vols = vols[-N_BENCH:]
    for name, fn in [("torch_fp32", torch_prob), ("onnx_fp32", fp32_run),
                     ("onnx_int8", int8_run)]:
        results[name] = bench(fn, bench_vols, N_BENCH)
        r = results[name]
        print(f"{name:<12} mean {r['mean_ms']:7.2f}  p50 {r['p50_ms']:7.2f}  "
              f"p95 {r['p95_ms']:7.2f} ms")

    OUT_JSON.write_text(json.dumps({
        "T": T, "input_hw": [H, W],
        "torch_threads": torch.get_num_threads(),
        "parity_max_dprob_fp32": float(max(diffs)),
        "parity_max_dprob_int8": float(max(diffs8)),
        "parity_mean_dprob_int8": float(np.mean(diffs8)),
        "runtime": results,
        "onnx_fp32_mb": ONNX_FP32.stat().st_size / 1e6,
        "onnx_int8_mb": ONNX_INT8.stat().st_size / 1e6,
    }, indent=2))
    print(f"saved -> {OUT_JSON.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
