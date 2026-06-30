"""Contact detection — image-plane bounding-box voting (homography-free).

A deliberately simple alternative to the geometric multi-view detector: run
MobileNet-SSD per camera, decide "touch" in each *image* purely from the 2-D
person boxes (do two boxes overlap / nearly touch?), then combine the three
cameras' per-image votes. No homography, no world-plane projection.

Per-camera touch
----------------
A camera "votes touch" iff it sees >=2 person boxes whose minimum edge-to-edge
gap is below tau pixels (gap = 0 means the boxes overlap). For >2 people the
minimum gap over all pairs is used.

Three combination rules (each reported as a confusion matrix)
-------------------------------------------------------------
  V1  Majority   : >=2 cameras vote touch.
  V2  Unanimous  : all 3 cameras vote touch.
  V3  Relaxed    : >=2 cameras vote touch AND the third camera corroborates --
                   it is near-touch (gap < tau2) or shows a single merged blob
                   (1 box = two people fused). A third camera that clearly sees
                   two well-separated people (>=2 boxes, gap >= tau2) vetoes.

Per-camera state given (tau, tau2 >= tau):
    TOUCH  : n>=2 and gap < tau
    NEAR   : n>=2 and gap < tau2          (TOUCH is a subset of NEAR)
    MERGED : n == 1
    CLEAR  : otherwise (n>=2 & gap>=tau2, or n==0)
V3 fires iff (#TOUCH >= 2) and (no camera is CLEAR).

Protocol
--------
Same per-session timeline split as the deep models (train .60 / val .15 /
test .25, min length T+2) so the 695-frame / 41-positive test set is identical
to the geometric / Thermo-X3D / MV-STGCN evaluation. Rule-based, so no training;
tau (and tau2 for V3) are tuned on the val split (max F1), reported on test.
Per-camera box stats are cached so the threshold sweeps need no SSD rerun.

Outputs reports/waveshare_contact_imageplane_results.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import eval_waveshare_contact as geo
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix

# Must match eval_waveshare_contact_dl.py exactly (identical test set).
PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
T = 16
TRAIN_FRAC = 0.60
VAL_FRAC = 0.15

TAU_GRID = [0, 1, 2, 3, 4, 6, 8, 10]          # strict touch gap (px)
TAU2_EXTRA = [0, 2, 4, 8]                      # tau2 = tau + extra (V3 near-touch)

OUT_JSON = _ROOT / "reports" / "waveshare_contact_imageplane_results.json"

# Reference numbers from the existing contact report (matched 695-frame test).
REFERENCE = {
    "Geometric":   dict(acc=0.914, prec=0.268, rec=0.268, f1=0.268, far=0.046),
    "MV-STGCN":    dict(acc=0.709, prec=0.148, rec=0.829, f1=0.252, far=0.298),
    "Thermo-X3D":  dict(acc=0.940, prec=0.484, rec=0.366, f1=0.417, far=0.024),
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def box_gap(a, b) -> float:
    """Minimum edge-to-edge distance between two (x,y,w,h) boxes (0 if overlap)."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    dx = max(0.0, max(a[0], b[0]) - min(ax2, bx2))
    dy = max(0.0, max(a[1], b[1]) - min(ay2, by2))
    return math.hypot(dx, dy)


def camera_stats(dets) -> tuple[int, float]:
    """(n_boxes, min_pairwise_gap). min_gap = inf when fewer than 2 boxes."""
    n = len(dets)
    if n < 2:
        return n, math.inf
    boxes = [d.bbox for d in dets]
    best = math.inf
    for i in range(n):
        for j in range(i + 1, n):
            best = min(best, box_gap(boxes[i], boxes[j]))
    return n, best


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def _cam_state(stat, tau: float, tau2: float) -> str:
    n, gap = stat
    if n >= 2 and gap < tau:
        return "TOUCH"
    if n >= 2 and gap < tau2:
        return "NEAR"
    if n == 1:
        return "MERGED"
    return "CLEAR"


def predict(cams, version: str, tau: float, tau2: float) -> int:
    states = [_cam_state(s, tau, tau2) for s in cams]
    votes = sum(1 for s in states if s == "TOUCH")
    if version == "V1":          # majority
        return 1 if votes >= 2 else 0
    if version == "V2":          # unanimous
        return 1 if votes == 3 else 0
    # V3 relaxed: >=2 touch AND third corroborates (no CLEAR camera)
    return 1 if (votes >= 2 and "CLEAR" not in states) else 0


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def _best_params(records_val, version: str):
    """Return (best_tau, best_tau2, best_cm) maximising val F1 (tie-break recall)."""
    yt = [r["label"] for r in records_val]
    best = None
    grid = []
    for tau in TAU_GRID:
        tau2_options = [tau + e for e in TAU2_EXTRA] if version == "V3" else [tau]
        for tau2 in tau2_options:
            yp = [predict(r["cams"], version, tau, tau2) for r in records_val]
            cm = binary_confusion_matrix(yt, yp)
            grid.append((tau, tau2, cm))
            key = (cm.f1, cm.recall)
            if best is None or key > (best[2].f1, best[2].recall):
                best = (tau, tau2, cm)
    return best[0], best[1], best[2], grid


def _scene_rows(records_test, version, tau, tau2):
    by_scene = defaultdict(lambda: ([], []))
    for r in records_test:
        by_scene[r["scene"]][0].append(r["label"])
        by_scene[r["scene"]][1].append(predict(r["cams"], version, tau, tau2))
    rows, agg = [], BinaryConfusionMatrix(0, 0, 0, 0)
    for scene in sorted(by_scene):
        yt, yp = by_scene[scene]
        cm = binary_confusion_matrix(yt, yp)
        agg = agg + cm
        rows.append({"scene": scene, "acc": cm.accuracy, "prec": cm.precision,
                     "rec": cm.recall, "f1": cm.f1, "far": cm.false_alarm_rate,
                     "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn, "total": cm.total})
    return rows, agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("  CONTACT DETECTION — image-plane bounding-box voting (homography-free)")
    print("=" * 78)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
    print(f"  Loaded SSD checkpoint → {geo.SSD_CKPT}")
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    # ---- Build per-frame camera box-stats over the matched val/test split ----
    records = []
    t0 = time.time(); n = 0
    for scene in sorted(labels):
        try:
            session = idx.find(scene)
        except Exception:
            continue
        if not all(ch in session.channels_with_data for ch in CHANNELS):
            continue
        frame_indices = sorted(fi for fi in labels[scene] if 0 <= fi < session.n_frames)
        if len(frame_indices) < T + 2:
            continue
        n_tr = int(len(frame_indices) * TRAIN_FRAC)
        n_va = int(len(frame_indices) * (TRAIN_FRAC + VAL_FRAC))
        for k, fi in enumerate(frame_indices):
            split = "train" if k < n_tr else ("val" if k < n_va else "test")
            if split == "train":
                continue
            cams = tuple(
                camera_stats(ssd.predict(pre[ch].predict(geo.get_frame(session, ch, fi))))
                for ch in CHANNELS
            )
            records.append({"split": split, "scene": scene,
                            "label": int(labels[scene][fi]), "cams": cams})
            n += 1
            if n % 300 == 0:
                print(f"      ... {n} frames  ({(time.time()-t0)/n*1000:.0f} ms/frame)", flush=True)

    val = [r for r in records if r["split"] == "val"]
    test = [r for r in records if r["split"] == "test"]
    print(f"  val={len(val)} (pos={sum(r['label'] for r in val)})  "
          f"test={len(test)} (pos={sum(r['label'] for r in test)})")

    versions = {"V1": "Majority (>=2 cameras)",
                "V2": "Unanimous (all 3 cameras)",
                "V3": "Unanimous-relaxed (2 + corroborating 3rd)"}
    results = {"profile": PROFILE.name, "protocol": "timeline_60_15_25",
               "test_frames": len(test), "test_pos": sum(r['label'] for r in test),
               "versions": {}}

    for v, desc in versions.items():
        tau, tau2, val_cm, grid = _best_params(val, v)
        rows, agg = _scene_rows(test, v, tau, tau2)
        thr = f"tau={tau}px" + (f", tau2={tau2}px" if v == "V3" else "")
        print("\n" + "=" * 78)
        print(f"  {v} — {desc}   [{thr}]   (val F1={val_cm.f1:.1%})")
        print("=" * 78)
        geo._print_scene_table(rows, f"Per-scene [test]")
        geo._print_cm(agg, f"Aggregate [test]  —  {v}: {desc}")
        results["versions"][v] = {
            "description": desc, "tau_px": tau, "tau2_px": tau2,
            "val_sweep": [{"tau": t, "tau2": t2, "f1": cm.f1, "prec": cm.precision,
                           "rec": cm.recall} for t, t2, cm in grid],
            "test": {"aggregate": {"acc": agg.accuracy, "prec": agg.precision,
                                   "rec": agg.recall, "f1": agg.f1,
                                   "far": agg.false_alarm_rate, "tp": agg.tp,
                                   "tn": agg.tn, "fp": agg.fp, "fn": agg.fn,
                                   "total": agg.total},
                     "per_scene": rows}}

    # ---- Summary vs the existing detectors ----
    print("\n" + "=" * 78)
    print("  SUMMARY — image-plane voting vs existing contact detectors (test)")
    print("=" * 78)
    hdr = f"  {'Method':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FAR':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, m in REFERENCE.items():
        print(f"  {name:<28} {m['acc']:>6.1%} {m['prec']:>7.1%} {m['rec']:>7.1%} "
              f"{m['f1']:>7.1%} {m['far']:>7.1%}")
    for v, desc in versions.items():
        a = results["versions"][v]["test"]["aggregate"]
        label = f"Image-plane {v} ({desc.split()[0]})"
        print(f"  {label:<28} {a['acc']:>6.1%} {a['prec']:>7.1%} {a['rec']:>7.1%} "
              f"{a['f1']:>7.1%} {a['far']:>7.1%}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n  JSON results → {OUT_JSON}")


if __name__ == "__main__":
    main()
