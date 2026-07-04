"""Profile FireSVMDetector.predict — where do the ~8 ms/frame go?

Trains the eval-grade model (same 70/30 stratified protocol as
eval_waveshare_fire.py), saves it as a checkpoint, then times each stage of
predict() separately over the test split:

    astype -> otsu_segment (histogram/threshold | morphology)
           -> extract_blobs (findContours | per-blob loop | skew+kurtosis)
           -> scaler.transform -> svm.predict -> svm.predict_proba

Outputs reports/fire_svm_profile.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_waveshare_fire as fire_eval
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.fire_detection.fire_svm import (
    FireSVMDetector,
    _FEATURES,
)
from thermal_algorithms.fire_detection.otsu_utils import extract_blobs, otsu_segment
from thermal_algorithms.training import DatasetIndex, FireFrameDataset

CKPT_DIR = _ROOT / "checkpoints" / "fire_svm_detector"
CKPT = CKPT_DIR / "_default.thalg"
OUT_JSON = _ROOT / "reports" / "fire_svm_profile.json"
N_FRAMES = 300


class StageTimer:
    def __init__(self):
        self.acc: dict[str, list[float]] = {}

    def add(self, key: str, dt_s: float) -> None:
        self.acc.setdefault(key, []).append(dt_s * 1e3)

    def stats(self) -> dict:
        return {
            k: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95)),
                "total_share": None}
            for k, v in self.acc.items()
        }


def instrumented_predict(det: FireSVMDetector, frame, timer: StageTimer):
    """Replicates FireSVMDetector.predict step by step with timers."""
    t0 = time.perf_counter()
    data = frame.data.astype(np.float32)
    t1 = time.perf_counter(); timer.add("astype", t1 - t0)

    # --- otsu_segment internals -------------------------------------------
    d_min, d_max = float(data.min()), float(data.max())
    span = d_max - d_min + 1e-6
    norm = ((data - d_min) / span * (det._n_bins - 1)).astype(np.float32)
    hist, _ = np.histogram(norm.ravel(), bins=det._n_bins,
                           range=(0.0, det._n_bins - 1.0))
    hist = hist.astype(np.float64)
    p = hist / hist.sum()
    idx = np.arange(det._n_bins, dtype=np.float64)
    w0 = np.cumsum(p)
    mu_k = np.cumsum(idx * p)
    mu_t = mu_k[-1]
    w1 = 1.0 - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_b2 = np.where((w0 > 0) & (w1 > 0),
                            (mu_t * w0 - mu_k) ** 2 / (w0 * w1), 0.0)
    k_star = int(np.argmax(sigma_b2))
    threshold = d_min + (k_star / (det._n_bins - 1)) * (d_max - d_min)
    raw_mask = (data >= threshold).astype(np.uint8) * 255
    t2 = time.perf_counter(); timer.add("otsu:hist+threshold", t2 - t1)

    eroded = cv2.erode(raw_mask, det._morph_kernel, iterations=1)
    mask = cv2.dilate(eroded, det._morph_kernel, iterations=2)
    t3 = time.perf_counter(); timer.add("otsu:morphology", t3 - t2)

    # --- extract_blobs internals -------------------------------------------
    from scipy.stats import skew, kurtosis as kurt

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    t4 = time.perf_counter(); timer.add("blobs:findContours", t4 - t3)

    blobs = []
    t_loop = 0.0
    t_stats = 0.0
    for contour in contours:
        ta = time.perf_counter()
        area = float(cv2.contourArea(contour))
        if area < 1.0:
            t_loop += time.perf_counter() - ta
            continue
        blob_mask = np.zeros_like(mask)
        cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
        pixels = data[blob_mask > 0].astype(np.float64)
        if pixels.size == 0:
            t_loop += time.perf_counter() - ta
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        m = cv2.moments(contour)
        cx = m["m10"] / m["m00"] if m["m00"] > 0 else bx + bw / 2.0
        cy = m["m01"] / m["m00"] if m["m00"] > 0 else by + bh / 2.0
        tb = time.perf_counter(); t_loop += tb - ta
        blobs.append({
            "area": area,
            "max_temp": float(pixels.max()),
            "mean_temp": float(pixels.mean()),
            "std_temp": float(pixels.std()),
            "skewness": float(skew(pixels)),
            "kurtosis": float(kurt(pixels)),
        })
        t_stats += time.perf_counter() - tb
    timer.add("blobs:mask+moments loop", t_loop)
    timer.add("blobs:numpy/scipy stats", t_stats)
    timer.add("blobs:n_contours", len(contours) / 1e3)  # count, not ms

    t5 = time.perf_counter()
    if blobs:
        hottest = max(blobs, key=lambda b: b["max_temp"])
        fv = np.array([hottest[k] for k in _FEATURES], dtype=np.float64)
    else:
        fv = np.zeros(len(_FEATURES), dtype=np.float64)
    t6 = time.perf_counter(); timer.add("feature-vector assembly", t6 - t5)

    fv_scaled = det._scaler.transform(fv.reshape(1, -1))
    t7 = time.perf_counter(); timer.add("sklearn:scaler.transform", t7 - t6)

    pred = int(det._svm.predict(fv_scaled)[0])
    t8 = time.perf_counter(); timer.add("sklearn:svm.predict", t8 - t7)

    proba = float(det._svm.predict_proba(fv_scaled)[0, 1])
    t9 = time.perf_counter(); timer.add("sklearn:svm.predict_proba", t9 - t8)
    return pred, proba


def main() -> None:
    idx = DatasetIndex(fire_eval.DATASET_ROOT, sensor_profile=WAVESHARE_26984)
    ds = FireFrameDataset(idx, channels=fire_eval.CHANNELS)
    train, test = fire_eval.split_examples(ds)
    print(f"train {len(train)} / test {len(test)} examples")

    det = FireSVMDetector(WAVESHARE_26984, kernel="rbf")
    t0 = time.time()
    det.fit([f for f, _ in train], [g for _, g in train])
    print(f"fitted in {time.time() - t0:.1f}s — n_SV = {det._svm.n_support_.tolist()}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    det.save(CKPT)
    print(f"checkpoint -> {CKPT.relative_to(_ROOT)}")

    frames = [f for f, _ in test[:N_FRAMES]]
    timer = StageTimer()

    # Warmup + full-predict reference timing
    for f in frames[:5]:
        det.predict(f)
    t0 = time.perf_counter()
    for f in frames:
        det.predict(f)
    full_ms = (time.perf_counter() - t0) / len(frames) * 1e3
    print(f"\nfull predict(): {full_ms:.2f} ms/frame over {len(frames)} frames")

    # Instrumented pass
    for f in frames:
        instrumented_predict(det, f, timer)

    stats = timer.stats()
    stage_keys = [k for k in stats if not k.startswith("blobs:n_")]
    total = sum(stats[k]["mean_ms"] for k in stage_keys)
    print(f"instrumented total: {total:.2f} ms/frame\n")
    print(f"{'stage':<28} {'mean ms':>9} {'p95 ms':>9} {'share':>7}")
    for k in stage_keys:
        s = stats[k]
        s["total_share"] = s["mean_ms"] / total
        print(f"{k:<28} {s['mean_ms']:>9.3f} {s['p95_ms']:>9.3f} {s['total_share']:>6.1%}")
    n_blobs = np.mean(timer.acc["blobs:n_contours"]) * 1e3
    print(f"\nmean contours/frame: {n_blobs:.1f}")

    OUT_JSON.write_text(json.dumps({
        "full_predict_ms": full_ms,
        "n_frames": len(frames),
        "n_support_vectors": det._svm.n_support_.tolist(),
        "mean_contours_per_frame": float(n_blobs),
        "stages": stats,
    }, indent=2))
    print(f"saved -> {OUT_JSON.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
