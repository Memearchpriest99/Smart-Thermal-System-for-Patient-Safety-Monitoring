"""False-alarm soak test: run the contact detectors over the usable archive
(~2h of ordinary 3-camera room activity, 06-23) and measure ALARMS PER HOUR.

Protocol:
  - per block: 12->8 Hz timestamp-grid resample, per-block p25 background
  - config-D (v9core + T1 morph) and X3D-T5 (th=0.2) per timestep
  - alarm EVENTS counted with the pipeline's 30 s cooldown semantics
  - events are cross-referenced against the 20 mined candidate windows:
    in-window alarms plausibly real (multi-person activity), out-of-window
    alarms are the suspect false alarms that matter for the ward metric

Outputs reports/soak_test_results.json (+ per-event list with seqs for
later visual spot-checks).
"""
import json, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import eval_waveshare_contact as geo
import eval_waveshare_contact_v8 as v8
import ablate_preprocessing as ab
from thermal_algorithms.core.types import Frame
from thermal_algorithms.preprocessing import TatenoPipeline
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from mine_thermal_archive import read_strided_safe, to_c, ARCHIVE, CAMS

LO, LC = 2, 7
BG_PCT = 25.0
TARGET_DT_US = 125_000
COOLDOWN_S = 30.0
X3D_TH = 0.2
DAY = "2026-06-23"

ssd = MobileNetSSDDetector.load(ab.RAW_CKPT)
x3d = ThermoX3DDetector.load(_ROOT / "checkpoints" / "thermo_x3d_detector" / "Waveshare_26984_relabel_T5.thalg")
cand = json.loads((_ROOT / "reports" / "archive_candidates.json").read_text())["verified"]
cand_by_file = {}
for c in cand:
    cand_by_file.setdefault(c["file"], []).append((c["seq_start"], c["seq_end"], c["id"]))

# blocks for the day (reuse mining index for usability)
mining = json.loads((_ROOT / "reports" / "archive_mining_index.json").read_text())
usable_files = [b["file"] for b in mining["blocks"] if b["day"] == DAY and b["timeline"]]
print(f"{len(usable_files)} usable {DAY} blocks", flush=True)

def load_block_8hz(fname):
    """(frames (M,3,62,80) degC, seqs (M,)) resampled to 8 Hz, or None."""
    streams = {}
    for cam in CAMS:
        p = ARCHIVE / DAY / cam / fname
        with h5py.File(p) as h:
            fr, cov_f = read_strided_safe(h["frames"], 1)
            sq, cov_s = read_strided_safe(h["seqs"], 1)
            ts, cov_t = read_strided_safe(h["timestamps"], 1)
            m = min(len(fr), len(sq), len(ts))
            if m < 50: return None
            streams[cam] = (fr[:m], sq[:m], ts[:m])
    t0 = max(s[2][0] for s in streams.values())
    t1 = min(s[2][-1] for s in streams.values())
    grid = np.arange(t0, t1, TARGET_DT_US)
    if len(grid) < 50: return None
    chans, seq0 = [], None
    for cam in CAMS:
        fr, sq, ts = streams[cam]
        idx = np.clip(np.searchsorted(ts, grid), 0, len(ts) - 1)
        prev = np.clip(idx - 1, 0, len(ts) - 1)
        take = np.where(np.abs(ts[idx] - grid) <= np.abs(ts[prev] - grid), idx, prev)
        chans.append(to_c(fr[take]))
        if cam == "cam_0": seq0 = sq[take]
    return np.stack(chans, 1), seq0     # (M, 3, 62, 80), (M,)

def alarm_events(stream, seqs, cooldown_steps):
    events, cool = [], -10**9
    for i, p in enumerate(stream):
        if p == 1 and i >= cool:
            events.append(int(seqs[i]))
            cool = i + cooldown_steps
    return events

results = {"day": DAY, "bg_pct": BG_PCT, "cooldown_s": COOLDOWN_S, "blocks": []}
tot_h = 0.0
tot_ev = {"config_D": [], "x3d_T5": []}
t0 = time.time()
for fname in usable_files:
    blk = load_block_8hz(fname)
    if blk is None:
        print(f"  {fname}: unreadable, skipped", flush=True)
        continue
    frames, seqs = blk
    M = frames.shape[0]
    hours = M / 8.0 / 3600.0
    tot_h += hours
    pre = {c: TatenoPipeline(geo.PROFILE).fit(
              [Frame(data=np.percentile(frames[:, c], BG_PCT, axis=0),
                     timestamp=0.0, camera_id=c)]) for c in range(3)}
    x3d.reset()
    d_base, xconf = [], []
    for i in range(M):
        raws = [Frame(data=frames[i, c], timestamp=i / 8.0, camera_id=c) for c in range(3)]
        resids = tuple(pre[c].predict(raws[c]) for c in range(3))
        dets = [ssd.predict(r) for r in raws]
        d_base.append(ab.v9core_decision(
            {"cams": [{"boxes": [d.bbox for d in dets[c]],
                       "resid": resids[c].data.astype(np.float32)} for c in range(3)]}, "resid"))
        xconf.append(float(x3d.predict(resids).confidence))
    streams = {"config_D": list(v8.morph(d_base, LO, LC)),
               "x3d_T5": [1 if c > X3D_TH else 0 for c in xconf]}
    spans = cand_by_file.get(fname, [])
    brec = {"file": fname, "timesteps": M, "hours": round(hours, 3), "events": {}}
    for name, stream in streams.items():
        evs = alarm_events(stream, seqs, int(COOLDOWN_S * 8))
        in_w = [e for e in evs if any(a <= e <= b for a, b, _ in spans)]
        out_w = [e for e in evs if e not in in_w]
        brec["events"][name] = {"total": len(evs), "in_candidate_windows": len(in_w),
                                "outside": len(out_w), "outside_seqs": out_w}
        tot_ev[name].append((len(in_w), len(out_w)))
    results["blocks"].append(brec)
    el = time.time() - t0
    print(f"  {fname}: {M} steps | " +
          " | ".join(f"{n}: {brec['events'][n]['total']} ev "
                     f"({brec['events'][n]['outside']} outside)" for n in streams) +
          f"  ({el:.0f}s)", flush=True)

print("\n" + "=" * 78)
print(f"  SOAK RESULT — {tot_h:.2f} h of 3-camera footage, cooldown {COOLDOWN_S:.0f}s")
for name, pairs in tot_ev.items():
    inw = sum(a for a, _ in pairs); outw = sum(b for _, b in pairs)
    print(f"  {name:<10} alarms: {inw + outw:>4} total | {inw:>3} in mined activity windows | "
          f"{outw:>4} outside  →  {outw / tot_h:.1f} suspect false alarms/hour")
results["summary"] = {n: {"in": sum(a for a, _ in p), "out": sum(b for _, b in p),
                          "out_per_hour": round(sum(b for _, b in p) / tot_h, 2)}
                      for n, p in tot_ev.items()}
results["total_hours"] = round(tot_h, 3)
(_ROOT / "reports" / "soak_test_results.json").write_text(json.dumps(results, indent=1))
print(f"\nwrote reports/soak_test_results.json")
