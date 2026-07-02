"""Label-quality diagnostic for contact detection (config-D).

Question: are the bad results detection failures or mislabeled data?

Method:
 1. Re-run config-D per-frame, keep every (frame, label, pred).
 2. Boundary analysis — for each error, distance (in *annotated-frame* steps)
    to the nearest frame with the OPPOSITE label. Errors hugging a label
    transition => fuzzy/off-by-a-few labels. Mid-segment errors => real
    detection or annotation problems.
 3. Dump per-frame records to reports/label_quality_frames.json so frames
    can be rendered/inspected afterwards.
"""
import json, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import eval_waveshare_contact as geo
import eval_waveshare_contact_v8 as v8
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex

LO, LC = 2, 7

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
ssd = MobileNetSSDDetector.load(ab.RAW_CKPT)
pre = geo.fit_preprocessors(idx)
labels = geo._load_contact_labels(geo.CONTACT_CSV)

print("running config-D over all labeled frames ...", flush=True)
scenes = {}
t0 = time.time(); n = 0
for scene in sorted(labels):
    try: s = idx.find(scene)
    except Exception: continue
    if not all(ch in s.channels_with_data for ch in geo.CHANNELS): continue
    fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
    if not fis: continue
    recs = []
    for fi in fis:
        cams = [{"boxes": [d.bbox for d in ssd.predict(geo.get_frame(s, ch, fi))],
                 "resid": pre[ch].predict(geo.get_frame(s, ch, fi)).data.astype(np.float32)}
                for ch in geo.CHANNELS]
        r = {"fi": fi, "label": int(labels[scene][fi]), "cams": cams}
        r["base"] = ab.v9core_decision(r, "resid")
        recs.append(r); n += 1
        if n % 300 == 0:
            print(f"   ... {n} frames ({(time.time()-t0)/n*1000:.0f} ms/f)", flush=True)
    scenes[scene] = recs

# ---- per-frame predictions after morphology ----------------------------------
frames_out = {}
for sc in sorted(scenes):
    rs = scenes[sc]
    sm = v8.morph([r["base"] for r in rs], LO, LC)
    frames_out[sc] = [{"fi": r["fi"], "label": r["label"], "base": int(r["base"]),
                       "pred": int(p), "n_boxes": [len(c["boxes"]) for c in r["cams"]]}
                      for r, p in zip(rs, sm)]

out_path = _ROOT / "reports" / "label_quality_frames.json"
out_path.write_text(json.dumps(frames_out, indent=1))
print(f"wrote {out_path}")

# ---- boundary analysis --------------------------------------------------------
print("\n" + "=" * 96)
print("  Error position analysis: distance (annotated-frame steps) to nearest OPPOSITE label")
print("  small distance => error sits at a label transition (fuzzy labels)")
print("=" * 96)

def dist_to_opposite(seq_labels, i):
    """Steps from index i to nearest index with label != seq_labels[i]; None if uniform."""
    tgt = 1 - seq_labels[i]
    best = None
    for j, l in enumerate(seq_labels):
        if l == tgt:
            d = abs(j - i)
            best = d if best is None else min(best, d)
    return best

summary = {}
for sc, recs in frames_out.items():
    labs = [r["label"] for r in recs]
    fps = [i for i, r in enumerate(recs) if r["pred"] == 1 and r["label"] == 0]
    fns = [i for i, r in enumerate(recs) if r["pred"] == 0 and r["label"] == 1]
    if not fps and not fns: continue
    fp_d = [dist_to_opposite(labs, i) for i in fps]
    fn_d = [dist_to_opposite(labs, i) for i in fns]
    summary[sc] = {"fp_dists": fp_d, "fn_dists": fn_d,
                   "fp_frames": [recs[i]["fi"] for i in fps],
                   "fn_frames": [recs[i]["fi"] for i in fns]}

def bucket(dists):
    """counts: <=2 steps, 3-5, >5 or no opposite label in video (uniform)."""
    near = sum(1 for d in dists if d is not None and d <= 2)
    mid = sum(1 for d in dists if d is not None and 3 <= d <= 5)
    far = sum(1 for d in dists if d is not None and d > 5)
    uni = sum(1 for d in dists if d is None)
    return near, mid, far, uni

hdr = f"  {'Scene':<24} {'errs':>5} | {'FP<=2':>6} {'FP3-5':>6} {'FP>5':>5} {'FPuni':>6} | {'FN<=2':>6} {'FN3-5':>6} {'FN>5':>5} {'FNuni':>6}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
tot = {"fp": [0, 0, 0, 0], "fn": [0, 0, 0, 0]}
for sc in sorted(summary):
    s = summary[sc]
    bf = bucket(s["fp_dists"]); bn = bucket(s["fn_dists"])
    for k in range(4): tot["fp"][k] += bf[k]; tot["fn"][k] += bn[k]
    print(f"  {sc:<24} {len(s['fp_dists'])+len(s['fn_dists']):>5} | "
          f"{bf[0]:>6} {bf[1]:>6} {bf[2]:>5} {bf[3]:>6} | "
          f"{bn[0]:>6} {bn[1]:>6} {bn[2]:>5} {bn[3]:>6}")
print("  " + "-" * (len(hdr) - 2))
print(f"  {'TOTAL':<24} {sum(tot['fp'])+sum(tot['fn']):>5} | "
      f"{tot['fp'][0]:>6} {tot['fp'][1]:>6} {tot['fp'][2]:>5} {tot['fp'][3]:>6} | "
      f"{tot['fn'][0]:>6} {tot['fn'][1]:>6} {tot['fn'][2]:>5} {tot['fn'][3]:>6}")
print("\n  'uni' = video has no frame with the opposite label (e.g. all-negative video)")

# ---- suspicious segments ------------------------------------------------------
print("\n" + "=" * 96)
print("  Consecutive error runs of >=4 annotated frames (candidates for systematic mislabels)")
print("=" * 96)
for sc, recs in sorted(frames_out.items()):
    runs = []
    cur = []
    for r in recs:
        if r["pred"] != r["label"]:
            cur.append(r)
        else:
            if len(cur) >= 4: runs.append(cur)
            cur = []
    if len(cur) >= 4: runs.append(cur)
    for run in runs:
        kind = "FP" if run[0]["label"] == 0 else "FN"
        print(f"  {sc:<24} {kind} frames {run[0]['fi']}..{run[-1]['fi']} "
              f"({len(run)} annotated frames, boxes/cam ~{np.mean([np.mean(r['n_boxes']) for r in run]):.1f})")
