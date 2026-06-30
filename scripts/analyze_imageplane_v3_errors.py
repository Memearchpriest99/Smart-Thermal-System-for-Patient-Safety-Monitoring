"""Error analysis for the image-plane V3 contact detector.

Re-runs SSD over the matched val/test split, reconstructs per-camera box stats,
and dissects exactly which frames V3 (tau=1px, tau2=9px) gets wrong and why.
Also prints per-camera state distributions on FN/FP frames and tries a few rule
variants (merged-aware, temporal persistence) at the same thresholds.

Read-only analysis; writes nothing except stdout (captured to a log).
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict, Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import eval_waveshare_contact as geo
import eval_waveshare_contact_imageplane as ip
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
T, TRAIN_FRAC, VAL_FRAC = ip.T, ip.TRAIN_FRAC, ip.VAL_FRAC
TAU, TAU2 = 1.0, 9.0          # V3's chosen thresholds


def build_records(idx, ssd, pre, labels):
    records = []
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
            cams = tuple(ip.camera_stats(ssd.predict(pre[ch].predict(geo.get_frame(session, ch, fi))))
                         for ch in CHANNELS)
            records.append({"split": split, "scene": scene, "fi": fi,
                            "label": int(labels[scene][fi]), "cams": cams})
    return records


def states(cams, tau=TAU, tau2=TAU2):
    return [ip._cam_state(s, tau, tau2) for s in cams]


def fmt_cams(cams):
    out = []
    for (n, gap), st in zip(cams, states(cams)):
        g = "inf" if gap == math.inf else f"{gap:.1f}"
        out.append(f"n{n}/g{g}/{st[:1]}")     # state initial: T/N/M/C
    return "  ".join(out)


def main():
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)
    print("Building records (SSD over val+test) ...", flush=True)
    records = build_records(idx, ssd, pre, labels)
    test = [r for r in records if r["split"] == "test"]
    print(f"test={len(test)} (pos={sum(r['label'] for r in test)})\n")

    # Baseline V3 decisions
    for r in test:
        r["pred"] = ip.predict(r["cams"], "V3", TAU, TAU2)
        r["st"] = states(r["cams"])
    fn = [r for r in test if r["label"] == 1 and r["pred"] == 0]
    fp = [r for r in test if r["label"] == 0 and r["pred"] == 1]
    print(f"V3 baseline: FN={len(fn)}  FP={len(fp)}\n")

    # ---- FALSE NEGATIVES (missed contacts) ----
    print("=" * 78); print("  FALSE NEGATIVES — missed contacts (label=1, pred=0)"); print("=" * 78)
    print("  legend: nX=#boxes  gY=min gap px  state T/N/M/C  (TOUCH/NEAR/MERGED/CLEAR)\n")
    for r in fn:
        n_touch = r["st"].count("TOUCH"); n_merged = r["st"].count("MERGED")
        n_clear = r["st"].count("CLEAR")
        print(f"  {r['scene']:<18} fi={r['fi']:>4}  | {fmt_cams(r['cams'])}   "
              f"[T{n_touch} M{n_merged} C{n_clear}]")
    # FN state summary
    print("\n  FN state distribution (per camera, across the missed frames):")
    cnt = Counter(s for r in fn for s in r["st"])
    print("   ", dict(cnt))
    why_touchlt2 = sum(1 for r in fn if r["st"].count("TOUCH") < 2)
    why_clear = sum(1 for r in fn if r["st"].count("TOUCH") >= 2 and "CLEAR" in r["st"])
    n_allmerged = sum(1 for r in fn if all(s == "MERGED" for s in r["st"]))
    n_ge2merged = sum(1 for r in fn if r["st"].count("MERGED") >= 2)
    print(f"    missed because <2 TOUCH votes : {why_touchlt2}/{len(fn)}")
    print(f"    missed because CLEAR veto     : {why_clear}/{len(fn)}")
    print(f"    frames with >=2 MERGED cams   : {n_ge2merged}/{len(fn)}  "
          f"(all 3 merged: {n_allmerged})")

    # ---- FALSE POSITIVES (false alarms) ----
    print("\n" + "=" * 78); print("  FALSE POSITIVES — false alarms (label=0, pred=1)"); print("=" * 78)
    by_scene = defaultdict(list)
    for r in fp:
        by_scene[r["scene"]].append(r)
    for scene in sorted(by_scene, key=lambda s: -len(by_scene[s])):
        rs = by_scene[scene]
        n_touch_dist = Counter(r["st"].count("TOUCH") for r in rs)
        n_merged_any = sum(1 for r in rs if "MERGED" in r["st"])
        gaps = [min(g for (n, g) in r["cams"] if n >= 2) for r in rs
                if any(n >= 2 for (n, _) in r["cams"])]
        med_gap = sorted(gaps)[len(gaps)//2] if gaps else float("nan")
        print(f"  {scene:<20} FP={len(rs):>3}  #TOUCH votes dist={dict(n_touch_dist)}  "
              f"merged-in-some={n_merged_any}  median min-gap={med_gap:.1f}px")
    # a few concrete FP examples from the worst scene
    worst = max(by_scene, key=lambda s: len(by_scene[s]))
    print(f"\n  Example FP frames from '{worst}':")
    for r in by_scene[worst][:8]:
        print(f"    fi={r['fi']:>4}  | {fmt_cams(r['cams'])}")

    # ---- Rule variants at the same thresholds ----
    print("\n" + "=" * 78); print("  RULE VARIANTS (same tau=1, tau2=9; test split)"); print("=" * 78)
    yt = [r["label"] for r in test]

    def evaluate(name, predfn):
        yp = [predfn(r) for r in test]
        cm = binary_confusion_matrix(yt, yp)
        print(f"  {name:<46} P={cm.precision:5.1%} R={cm.recall:5.1%} "
              f"F1={cm.f1:5.1%} FAR={cm.false_alarm_rate:5.1%}  (TP{cm.tp} FP{cm.fp} FN{cm.fn})")
        return cm

    evaluate("V3 baseline", lambda r: ip.predict(r["cams"], "V3", TAU, TAU2))

    # Variant A — merged-aware: a merged camera counts toward the 2 needed,
    # provided >=1 real TOUCH and no CLEAR camera.
    def v_merged(r):
        st = r["st"]
        ev = st.count("TOUCH") + st.count("MERGED")
        return 1 if (st.count("TOUCH") >= 1 and ev >= 2 and "CLEAR" not in st) else 0
    evaluate("A: merged-aware (TOUCH>=1, TOUCH+MERGED>=2, no CLEAR)", v_merged)

    # Variant B — temporal persistence: V3 positive only if it also fired in
    # >=1 of the previous 2 frames of the same scene (removes 1-frame flickers).
    order = defaultdict(list)
    for r in test:
        order[r["scene"]].append(r)
    base = {id(r): ip.predict(r["cams"], "V3", TAU, TAU2) for r in test}
    persist = {}
    for scene, rs in order.items():
        rs_sorted = sorted(rs, key=lambda x: x["fi"])
        for i, r in enumerate(rs_sorted):
            window = rs_sorted[max(0, i-2):i+1]
            # require current positive AND >=2 of last 3 positive (incl. current)
            k = sum(base[id(w)] for w in window)
            persist[id(r)] = 1 if (base[id(r)] == 1 and k >= 2) else 0
    evaluate("B: V3 + persistence (>=2 of last 3 frames)", lambda r: persist[id(r)])

    # Variant C — merged-aware + persistence
    mC = {}
    for scene, rs in order.items():
        rs_sorted = sorted(rs, key=lambda x: x["fi"])
        raw = {id(r): v_merged(r) for r in rs_sorted}
        for i, r in enumerate(rs_sorted):
            window = rs_sorted[max(0, i-2):i+1]
            k = sum(raw[id(w)] for w in window)
            mC[id(r)] = 1 if (raw[id(r)] == 1 and k >= 2) else 0
    evaluate("C: merged-aware + persistence", lambda r: mC[id(r)])

    # Save FN/FP identifiers for image inspection
    print("\n  FN frames:", [(r['scene'], r['fi']) for r in fn])
    print("\n  FP (worst scene) frames:", [(worst, r['fi']) for r in by_scene[worst][:12]])


if __name__ == "__main__":
    main()
