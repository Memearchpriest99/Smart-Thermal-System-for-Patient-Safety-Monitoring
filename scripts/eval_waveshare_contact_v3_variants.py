"""V3 image-plane touch detector — five improvements, each evaluated separately.

Baseline V3 (image-plane majority-with-corroboration) reaches F1 28.6% / FAR
14.8% on the matched 695-frame test set. Error analysis identified five levers;
this script implements each as an independent variant (NOT cumulative), re-tunes
its thresholds on the val split, and reports it on the test split so we can
judge what each improvement is worth on its own.

  V3.0  baseline      : >=2 cameras TOUCH and no CLEAR camera.
  V3.1  merge-aware   : a single-blob (MERGED) camera counts toward the quorum,
                        provided >=1 camera shows a real TOUCH and none is CLEAR.
                        (Targets the blob-merge false negatives in fights.)
  V3.2  persistence   : V3.0, but fire only if >=2 of the last 3 frames fire.
                        (Targets transient crowd-overlap false positives.)
  V3.3  size-norm gap : TOUCH uses gap / min(box width) instead of raw pixels,
                        so "touching" adapts to apparent person size / distance.
  V3.4  box-merge     : merge over-segmented boxes (one person split in two)
                        before voting. (Targets single-person false positives.)
  V3.5  X3D ensemble  : V3.0 OR Thermo-X3D(conf > theta). (Complementarity:
                        V3 catches fights X3D misses; X3D quiet where V3 sprays.)

All variants share the same SSD detections (and timeline split) as the existing
report. Outputs reports/waveshare_contact_v3_variants_results.json.
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
from eval_waveshare_contact_imageplane import box_gap
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix

PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
T = 16
TRAIN_FRAC = 0.60
VAL_FRAC = 0.15
X3D_CKPT = _ROOT / "checkpoints" / "thermo_x3d_detector" / "Waveshare_26984.thalg"

TAU_GRID = [0, 1, 2, 3, 4, 6, 8, 10]
TAU2_EXTRA = [0, 2, 4, 8]
RHO_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]      # size-normalised gap (V3.3)
RHO2_EXTRA = [0.0, 0.1, 0.2, 0.4]
THETA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # X3D conf threshold (V3.5)
MERGE_IOU = 0.2          # V3.4: merge two boxes if IoU>=this ...
MERGE_CONTAIN = 0.7      # ... or if intersection/min-area >= this (containment)

OUT_JSON = _ROOT / "reports" / "waveshare_contact_v3_variants_results.json"


# ---------------------------------------------------------------------------
# Box metrics
# ---------------------------------------------------------------------------

def _inter(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]; bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    return iw * ih


def _iou(a, b):
    inter = _inter(a, b)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def min_gap(boxes):
    if len(boxes) < 2:
        return math.inf
    return min(box_gap(boxes[i], boxes[j])
               for i in range(len(boxes)) for j in range(i + 1, len(boxes)))


def min_norm_gap(boxes):
    """min over pairs of  gap / min(width_i, width_j)."""
    if len(boxes) < 2:
        return math.inf
    best = math.inf
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ref = max(1.0, min(boxes[i][2], boxes[j][2]))
            best = min(best, box_gap(boxes[i], boxes[j]) / ref)
    return best


def merge_oversegmented(boxes):
    """Greedily merge boxes that are likely one person (high IoU or containment)
    into their union bounding box."""
    bs = [tuple(b) for b in boxes]
    changed = True
    while changed and len(bs) > 1:
        changed = False
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = bs[i], bs[j]
                inter = _inter(a, b)
                min_area = min(a[2] * a[3], b[2] * b[3]) or 1.0
                if _iou(a, b) >= MERGE_IOU or inter / min_area >= MERGE_CONTAIN:
                    x1 = min(a[0], b[0]); y1 = min(a[1], b[1])
                    x2 = max(a[0] + a[2], b[0] + b[2]); y2 = max(a[1] + a[3], b[1] + b[3])
                    bs = [bs[k] for k in range(len(bs)) if k not in (i, j)]
                    bs.append((x1, y1, x2 - x1, y2 - y1))
                    changed = True
                    break
            if changed:
                break
    return bs


# ---------------------------------------------------------------------------
# Per-camera state + combination
# ---------------------------------------------------------------------------

def _state(n, metric, t, t2):
    if n >= 2 and metric < t:
        return "T"
    if n >= 2 and metric < t2:
        return "N"
    if n == 1:
        return "M"
    return "C"


def _cam_states(cams_boxes, metric_fn, t, t2, premerge=False):
    out = []
    for boxes in cams_boxes:
        b = merge_oversegmented(boxes) if premerge else boxes
        out.append(_state(len(b), metric_fn(b), t, t2))
    return out


def _combine(states, mergeaware: bool) -> int:
    votes = states.count("T"); merged = states.count("M"); clear = "C" in states
    if mergeaware:
        return 1 if (votes >= 1 and votes + merged >= 2 and not clear) else 0
    return 1 if (votes >= 2 and not clear) else 0


# ---------------------------------------------------------------------------
# Decision builders (return list of preds aligned to `recs`)
# ---------------------------------------------------------------------------

def _base_preds(recs, params, *, metric_fn=min_gap, mergeaware=False, premerge=False):
    t, t2 = params
    return [_combine(_cam_states(r["boxes"], metric_fn, t, t2, premerge), mergeaware)
            for r in recs]


def _persist(recs, raw_preds):
    """Require current positive AND >=2 of last 3 frames (per scene, fi order)."""
    idx = {id(r): k for k, r in enumerate(recs)}
    by_scene = defaultdict(list)
    for r in recs:
        by_scene[r["scene"]].append(r)
    out = [0] * len(recs)
    for scene, rs in by_scene.items():
        rs = sorted(rs, key=lambda x: x["fi"])
        for i, r in enumerate(rs):
            window = rs[max(0, i - 2):i + 1]
            k = sum(raw_preds[idx[id(w)]] for w in window)
            out[idx[id(r)]] = 1 if (raw_preds[idx[id(r)]] == 1 and k >= 2) else 0
    return out


def decisions(recs, variant, params, x3dconf=None):
    if variant == "V3.0":
        return _base_preds(recs, params)
    if variant == "V3.1":
        return _base_preds(recs, params, mergeaware=True)
    if variant == "V3.2":
        return _persist(recs, _base_preds(recs, params))
    if variant == "V3.3":
        return _base_preds(recs, params, metric_fn=min_norm_gap)
    if variant == "V3.4":
        return _base_preds(recs, params, premerge=True)
    if variant == "V3.5":
        t, t2, theta = params
        base = _base_preds(recs, (t, t2))
        return [1 if (b or (x3dconf[id(r)] > theta)) else 0
                for b, r in zip(base, recs)]
    raise ValueError(variant)


def _grid(variant):
    if variant == "V3.3":
        return [(r, r + e) for r in RHO_GRID for e in RHO2_EXTRA]
    if variant == "V3.5":
        return [(t, t + e, th) for t in TAU_GRID for e in TAU2_EXTRA for th in THETA_GRID]
    return [(t, t + e) for t in TAU_GRID for e in TAU2_EXTRA]


# ---------------------------------------------------------------------------
# Build caches
# ---------------------------------------------------------------------------

def build_boxes(idx, ssd, pre, labels):
    """val/test records with per-camera SSD box lists."""
    recs = []
    for scene in sorted(labels):
        try:
            session = idx.find(scene)
        except Exception:
            continue
        if not all(ch in session.channels_with_data for ch in CHANNELS):
            continue
        fis = sorted(fi for fi in labels[scene] if 0 <= fi < session.n_frames)
        if len(fis) < T + 2:
            continue
        n_tr = int(len(fis) * TRAIN_FRAC); n_va = int(len(fis) * (TRAIN_FRAC + VAL_FRAC))
        for k, fi in enumerate(fis):
            split = "train" if k < n_tr else ("val" if k < n_va else "test")
            if split == "train":
                continue
            boxes = tuple([d.bbox for d in ssd.predict(pre[ch].predict(geo.get_frame(session, ch, fi)))]
                          for ch in CHANNELS)
            recs.append({"split": split, "scene": scene, "fi": fi,
                         "label": int(labels[scene][fi]), "boxes": boxes})
    return recs


def build_x3d_conf(idx, pre, labels, recs):
    """Per-record Thermo-X3D confidence (replay full sessions to warm the buffer)."""
    det = ThermoX3DDetector.load(X3D_CKPT)
    want = {(r["scene"], r["fi"]): r for r in recs}
    conf = {}
    for scene in sorted(labels):
        try:
            session = idx.find(scene)
        except Exception:
            continue
        if not all(ch in session.channels_with_data for ch in CHANNELS):
            continue
        fis = sorted(fi for fi in labels[scene] if 0 <= fi < session.n_frames)
        if len(fis) < T + 2:
            continue
        det.reset()
        for fi in fis:
            triplet = tuple(pre[ch].predict(geo.get_frame(session, ch, fi)) for ch in CHANNELS)
            ev = det.predict(triplet)
            if (scene, fi) in want:
                conf[id(want[(scene, fi)])] = float(ev.confidence)
    return conf


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def tune_and_test(val, test, variant, x3dconf=None):
    yt_val = [r["label"] for r in val]
    best = None
    for params in _grid(variant):
        yp = decisions(val, variant, params, x3dconf)
        cm = binary_confusion_matrix(yt_val, yp)
        if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
            best = (params, cm)
    params = best[0]
    yt = [r["label"] for r in test]
    yp = decisions(test, variant, params, x3dconf)
    cm = binary_confusion_matrix(yt, yp)
    # per scene
    by = defaultdict(lambda: ([], []))
    for r, p in zip(test, yp):
        by[r["scene"]][0].append(r["label"]); by[r["scene"]][1].append(p)
    rows = []
    for scene in sorted(by):
        a, b = by[scene]; c = binary_confusion_matrix(a, b)
        rows.append({"scene": scene, "rec": c.recall, "fp": c.fp,
                     "tp": c.tp, "fn": c.fn, "total": c.total})
    return params, cm, rows


def main():
    print("=" * 80)
    print("  V3 IMAGE-PLANE TOUCH — five improvements, evaluated separately")
    print("=" * 80)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    print("  [1] SSD boxes over val+test ...", flush=True)
    t0 = time.time()
    recs = build_boxes(idx, ssd, pre, labels)
    val = [r for r in recs if r["split"] == "val"]
    test = [r for r in recs if r["split"] == "test"]
    print(f"      val={len(val)} (pos={sum(r['label'] for r in val)})  "
          f"test={len(test)} (pos={sum(r['label'] for r in test)})  ({time.time()-t0:.0f}s)")

    print("  [2] Thermo-X3D confidences (replaying sessions) ...", flush=True)
    t0 = time.time()
    x3dconf = build_x3d_conf(idx, pre, labels, recs)
    print(f"      {len(x3dconf)} frames scored  ({time.time()-t0:.0f}s)")

    variants = {
        "V3.0": "baseline (>=2 TOUCH, no CLEAR)",
        "V3.1": "merge-aware quorum",
        "V3.2": "+ temporal persistence (>=2 of 3)",
        "V3.3": "size-normalised gap",
        "V3.4": "merge over-segmented boxes",
        "V3.5": "OR Thermo-X3D ensemble",
    }

    results = {}
    summary = []
    base_f1 = base_far = None
    for v, desc in variants.items():
        params, cm, rows = tune_and_test(val, test, v, x3dconf)
        results[v] = {"desc": desc, "params": params,
                      "test": {"acc": cm.accuracy, "prec": cm.precision, "rec": cm.recall,
                               "f1": cm.f1, "far": cm.false_alarm_rate, "tp": cm.tp,
                               "tn": cm.tn, "fp": cm.fp, "fn": cm.fn},
                      "per_scene": rows}
        if v == "V3.0":
            base_f1, base_far = cm.f1, cm.false_alarm_rate
        summary.append((v, desc, params, cm))

    # ---- Report ----
    print("\n" + "=" * 80)
    print("  CONFUSION MATRICES (test: 41 contact / 654 no-contact)")
    print("=" * 80)
    for v, desc, params, cm in summary:
        print(f"\n  {v} — {desc}   params={params}")
        print(f"    TP={cm.tp:<3} FN={cm.fn:<3} | FP={cm.fp:<3} TN={cm.tn:<3}   "
              f"P={cm.precision:.1%} R={cm.recall:.1%} F1={cm.f1:.1%} FAR={cm.false_alarm_rate:.1%}")

    print("\n" + "=" * 80)
    print("  WHAT EACH IMPROVEMENT IS WORTH  (Δ vs V3.0 baseline)")
    print("=" * 80)
    hdr = f"  {'Variant':<8} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FAR':>7}   {'ΔF1':>7} {'ΔFAR':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for v, desc, params, cm in summary:
        df1 = (cm.f1 - base_f1) * 100; dfar = (cm.false_alarm_rate - base_far) * 100
        print(f"  {v:<8} {cm.precision:>6.1%} {cm.recall:>7.1%} {cm.f1:>7.1%} "
              f"{cm.false_alarm_rate:>7.1%}   {df1:>+6.1f}pt {dfar:>+6.1f}pt")
    print("\n  (ΔF1 / ΔFAR in percentage points; FAR lower = better.)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
