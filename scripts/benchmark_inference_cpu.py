"""CPU inference-time benchmark for every detector (fire / person / contact).

Times per-frame ``predict()`` cost on real Waveshare frames
(datasets/waveshare_work), forcing CPU for the torch models, and compares
against the runtime budget: 8 Hz => 125 ms per pipeline tick.

Per camera each tick runs preprocessing + fire + human detection; contact
runs once on the 3-view bundle. Numbers here are from the dev machine —
expect roughly 3-5x slower single-core on the Pi 5 (Cortex-A76).

Usage::

    python scripts/benchmark_inference_cpu.py [--n 100] [--threads N]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.core.types import Frame
from thermal_algorithms.preprocessing import TatenoPipeline
from thermal_algorithms.fire_detection.otsu_pipeline import OtsuFireDetector
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector
from thermal_algorithms.human_detection.adaptive_threshold import (
    AdaptiveThresholdDetector,
)
from thermal_algorithms.training import (
    DatasetIndex,
    FireFrameDataset,
    FrameLevelDataset,
    PERSON_CLASS_ID,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "datasets" / "waveshare_work"
CKPT = ROOT / "checkpoints"
PROFILE = WAVESHARE_26984
BUDGET_MS = 1000.0 / PROFILE.sample_rate_hz  # 125 ms tick at 8 Hz
EMPTY_SCENE = "empty_room"
OUT_JSON = ROOT / "reports" / "cpu_inference_benchmark.json"


def timeit(fn, inputs, n_warmup=3, label=""):
    """Per-call wall time (ms) of fn over inputs, cycling if needed."""
    seq = list(inputs)
    for x in seq[:n_warmup]:
        fn(x)
    times = []
    for i, x in enumerate(seq[n_warmup:]):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(times)
    return {
        "n": len(t),
        "mean_ms": float(t.mean()),
        "p50_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)),
    }


def force_cpu(det):
    det._device_str = "cpu"
    det._device = None
    if getattr(det, "_model", None) is not None:
        import torch

        det._model = det._model.cpu()
        det._device = torch.device("cpu")
        if hasattr(det._model, "build_anchors"):
            det._model.build_anchors(device=det._device)
    return det


def fmt(name, stats, per_camera=True):
    mult = 3 if per_camera else 1
    worst = stats["p95_ms"] * mult
    pct = 100.0 * worst / BUDGET_MS
    scope = "x3 cams" if per_camera else "1x tick"
    return (f"{name:<38} mean {stats['mean_ms']:8.2f}  p50 {stats['p50_ms']:8.2f}  "
            f"p95 {stats['p95_ms']:8.2f} ms   [{scope}: {pct:5.1f}% of budget]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="timed frames per model")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (default: torch default)")
    args = ap.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)

    print(f"CPU: {platform.processor()}")
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    print(f"Budget: {BUDGET_MS:.0f} ms/tick @ {PROFILE.sample_rate_hz:.0f} Hz\n")

    idx = DatasetIndex(DATASET_ROOT, sensor_profile=PROFILE)
    sessions = {s.scene: s for s in idx.sessions}
    empty = sessions[EMPTY_SCENE]

    # Timing scene: 3 channels, enough frames, people present.
    scene = next(
        s for s in idx.sessions
        if s.scene != EMPTY_SCENE
        and len(s.channels_with_data) == 3
        and s.n_frames >= args.n + 20
    )
    n_load = min(scene.n_frames, args.n + 20)
    raw = {ch: scene.load_frames(ch)[:n_load] for ch in (0, 1, 2)}
    frames = {
        ch: [Frame(data=raw[ch][i], timestamp=i / 8.0, camera_id=ch)
             for i in range(n_load)]
        for ch in (0, 1, 2)
    }
    print(f"Timing on scene '{scene.scene}' ({n_load} frames/ch)\n")

    results = {}

    # ---- Preprocessing ---------------------------------------------------
    pre = TatenoPipeline(PROFILE).fit(
        [empty.load_frame(0, i) for i in range(0, empty.n_frames, 2)]
    )
    results["TatenoPipeline (preprocessing)"] = timeit(pre.predict, frames[0])
    residual = [pre.predict(f) for f in frames[0]]

    # ---- Fire ------------------------------------------------------------
    otsu = OtsuFireDetector(PROFILE).fit([])
    otsu.reset()
    results["OtsuFireDetector (fire)"] = timeit(otsu.predict, frames[0])

    fire_scenes = [s.scene for s in idx.sessions
                   if any(k in s.scene.lower() for k in ("cig", "heater", "fire"))]
    fire_ds = FireFrameDataset(idx, channels=(0, 1, 2),
                               scenes=fire_scenes + [scene.scene])
    pos, neg = [], []
    for f, alert in fire_ds:
        (pos if alert.level.name != "SAFE" else neg).append((f, alert))
        if len(pos) >= 150 and len(neg) >= 150:
            break
    train = pos[:150] + neg[:150]
    print(f"FireSVM training set: {len(pos[:150])} fire / {len(neg[:150])} safe")
    fire_svm = FireSVMDetector(PROFILE, kernel="rbf")
    fire_svm.fit([f for f, _ in train], [a for _, a in train])
    results["FireSVMDetector (fire)"] = timeit(fire_svm.predict, frames[0])

    # ---- Person ------------------------------------------------------------
    at = AdaptiveThresholdDetector(PROFILE)
    try:
        at.fit([])
    except Exception:
        pass
    results["AdaptiveThreshold (person)"] = timeit(at.predict, residual)

    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector

    ssd = force_cpu(MobileNetSSDDetector.load(
        CKPT / "mobilenet_ssd_detector" / "Waveshare_26984_raw.thalg"))
    results["MobileNet-SSD (person)"] = timeit(ssd.predict, frames[0])

    from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector

    person_ds = FrameLevelDataset(idx, channels=(0,),
                                  class_filter=[PERSON_CLASS_ID])
    hog_train = []
    for f, dets in person_ds:
        if dets:
            hog_train.append((pre.predict(f), dets))
        if len(hog_train) >= 20:
            break
    hog = HOGSVMDetector(PROFILE).fit(hog_train)
    results["HOG-SVM (person)"] = timeit(hog.predict, residual[:5], n_warmup=1,
                                         label="hog")

    # ---- Contact -----------------------------------------------------------
    from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector

    x3d = force_cpu(ThermoX3DDetector.load(
        CKPT / "thermo_x3d_detector" / "Waveshare_26984_T5_v2.thalg"))
    bundles = [tuple(frames[ch][i] for ch in (0, 1, 2)) for i in range(n_load)]
    for b in bundles[:x3d._T]:  # fill rolling buffers (untimed)
        x3d.predict(b)
    n_x3d = min(40 + 3, len(bundles) - x3d._T)
    results["Thermo-X3D T5v2 (contact, 3-view)"] = timeit(
        x3d.predict, bundles[x3d._T:x3d._T + n_x3d])

    # ---- Report -------------------------------------------------------------
    print()
    per_camera = {"TatenoPipeline (preprocessing)", "OtsuFireDetector (fire)",
                  "FireSVMDetector (fire)", "AdaptiveThreshold (person)",
                  "MobileNet-SSD (person)", "HOG-SVM (person)"}
    for name, stats in results.items():
        print(fmt(name, stats, per_camera=name in per_camera))

    # Representative full-tick estimate: Tateno + best fire + best person
    # per camera (x3) + contact once.
    tick = 3 * (results["TatenoPipeline (preprocessing)"]["p50_ms"]
                + results["FireSVMDetector (fire)"]["p50_ms"]
                + results["MobileNet-SSD (person)"]["p50_ms"]) \
        + results["Thermo-X3D T5v2 (contact, 3-view)"]["p50_ms"]
    print(f"\nFull-tick estimate (3x[Tateno+FireSVM+SSD] + X3D): "
          f"{tick:.1f} ms  ({100 * tick / BUDGET_MS:.0f}% of {BUDGET_MS:.0f} ms budget)")

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "cpu": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "budget_ms": BUDGET_MS,
        "scene": scene.scene,
        "results": results,
        "full_tick_estimate_ms": tick,
    }, indent=2))
    print(f"Saved -> {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
