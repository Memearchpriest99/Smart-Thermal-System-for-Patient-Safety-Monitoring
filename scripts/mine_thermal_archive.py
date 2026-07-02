"""Mine the raw thermal_captures h5 archive for contact-candidate segments.

Screens ~5.2h of 3-camera Waveshare recordings (room-1) for multi-person,
contact-likely activity, then ranks candidate windows and verifies the top
ones with the raw-SSD person detector.

Archive format (per file):
  frames (N, 62, 80) uint16 deci-Kelvin (degC = v/10 - 273.15)
  seqs (N,) uint64, timestamps (N,) microseconds (~12 Hz)
  NOTE: zstd HDF5 filter (32015) — `import hdf5plugin` is REQUIRED before
  h5py.File or reads fail with "can't open directory" / missing-filter errors.

Blocks: files with identical basenames across cam_0/1/2 form a 3-camera block;
blocks missing a readable camera are skipped (14 corrupt cam_2 files on 06-30).

Outputs:
  reports/archive_mining_index.json   per-block screening timelines + skip list
  reports/archive_candidates.json     top-ranked SSD-verified candidate windows
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import hdf5plugin  # noqa: F401  (registers zstd filter 32015)
import h5py
import numpy as np
from scipy import ndimage

ARCHIVE = Path(r"C:\Users\Guy\Desktop\Final Project\More data\DAR&BASH\DAR&BASH\thermal_captures\fallback\room-1")
CAMS = ("cam_0", "cam_1", "cam_2")
STRIDE = 6                 # screen every 6th frame (~2 Hz)
BG_SAMPLES = 200           # frames sampled per block for the median background
WARM_C = 2.0               # residual threshold (degC above background)
AREA_MIN, AREA_MAX = 15, 150   # person-plausible blob area at 62x80
NEAR_PX = 6.0              # blob min-distance for "contact-likely"
MERGE_FACTOR = 1.6         # blob area >= factor * median person area -> merge flag
MIN_WIN_S = 12.0           # min candidate window length
PAD_S = 5.0                # padding added to each side of a window
MAX_LEN_S = 90.0           # cap candidate length
TOP_N = 20
SCREEN_HZ = 12.0 / STRIDE  # effective screening rate

def to_c(u16):
    return u16.astype(np.float32) / 10.0 - 273.15

def read_strided_safe(dset, stride, chunk=1200):
    """Strided read tolerant of corrupt zstd chunks: reads in pieces, keeps the
    readable prefix, stops at the first failing piece. Returns (array, n_source
    frames covered)."""
    parts = []
    n = dset.shape[0]
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        # keep global stride phase across pieces
        start = lo + (-lo) % stride
        if start >= hi:
            continue
        try:
            parts.append(dset[start:hi:stride])
        except OSError:
            return (np.concatenate(parts, 0) if parts else
                    np.empty((0,) + dset.shape[1:], dset.dtype)), lo
    return (np.concatenate(parts, 0) if parts else
            np.empty((0,) + dset.shape[1:], dset.dtype)), n

def blob_stats(resid):
    """(n_person_blobs, min_pairwise_centroid_dist, merge_flag)."""
    mask = resid > WARM_C
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, np.inf, False
    areas = ndimage.sum_labels(np.ones_like(lab), lab, index=range(1, n + 1))
    keep = [(i + 1, a) for i, a in enumerate(areas) if AREA_MIN <= a <= AREA_MAX * 2]
    person = [i for i, a in keep if a <= AREA_MAX]
    big = [a for _, a in keep if a > AREA_MAX]
    merge = False
    if person and big:
        med = float(np.median([a for i, a in keep if i in person])) if person else AREA_MIN
        merge = any(a >= MERGE_FACTOR * max(med, AREA_MIN) for a in big)
    elif big:
        merge = True     # only oversized blobs — likely two bodies fused
    cents = ndimage.center_of_mass(np.ones_like(lab), lab, index=person) if person else []
    mind = np.inf
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            d = ((cents[i][0] - cents[j][0]) ** 2 + (cents[i][1] - cents[j][1]) ** 2) ** 0.5
            mind = min(mind, d)
    n_bodies = len(person) + 2 * len(big)   # a big blob counts as ~2 bodies
    return n_bodies, mind, merge

# ---------------------------------------------------------------------------
# 1. index blocks
# ---------------------------------------------------------------------------
by_key = defaultdict(dict)   # (date, basename) -> {cam: path}
for day_dir in sorted(ARCHIVE.iterdir()):
    if not day_dir.is_dir(): continue
    for cam in CAMS:
        for f in sorted((day_dir / cam).glob("*.h5")):
            by_key[(day_dir.name, f.name)][cam] = f

blocks, skipped = [], []
for (day, name), cams in sorted(by_key.items()):
    if set(cams) != set(CAMS):
        skipped.append({"day": day, "file": name, "reason": "missing camera(s)"})
        continue
    ok = True
    for cam, p in cams.items():
        try:
            with h5py.File(p) as h:
                _ = h["frames"].shape
        except OSError:
            skipped.append({"day": day, "file": name, "reason": f"unreadable {cam}"})
            ok = False
            break
    if ok:
        blocks.append({"day": day, "file": name, "paths": {c: str(p) for c, p in cams.items()}})
print(f"blocks: {len(blocks)} usable, {len(skipped)} skipped", flush=True)

# ---------------------------------------------------------------------------
# 2. screen each block
# ---------------------------------------------------------------------------
index = {"stride": STRIDE, "screen_hz": SCREEN_HZ, "blocks": [], "skipped": skipped}
t0 = time.time()
for bi, blk in enumerate(blocks):
    per_cam = {}
    n_frames = None
    truncated = False
    for cam in CAMS:
        with h5py.File(blk["paths"][cam]) as h:
            raw, cov_f = read_strided_safe(h["frames"], STRIDE)
            seq_raw, cov_s = read_strided_safe(h["seqs"], STRIDE)
            covered = min(cov_f, cov_s)
            if covered < h["frames"].shape[0]:
                truncated = True
            n_keep = (covered + STRIDE - 1) // STRIDE
            frames = to_c(raw[:n_keep])
            seqs = seq_raw[:n_keep][:len(frames)]
            n_frames = covered if n_frames is None else min(n_frames, covered)
        if len(frames) < 10:
            per_cam = None
            break
        bg_idx = np.linspace(0, len(frames) - 1, min(BG_SAMPLES, len(frames))).astype(int)
        bg = np.median(frames[bg_idx], axis=0)
        stats = [blob_stats(f - bg) for f in frames]
        per_cam[cam] = {"seqs": seqs, "stats": stats}
    if per_cam is None:
        skipped.append({"day": blk["day"], "file": blk["file"],
                        "reason": "corrupt data (unreadable prefix)"})
        blk["timeline"] = []
        continue
    if truncated:
        print(f"      note: {blk['file']} partially corrupt — using readable prefix", flush=True)
    # combine per timestep (align by position — same capture, same length after stride)
    L = min(len(per_cam[c]["stats"]) for c in CAMS)
    timeline = []
    for i in range(L):
        s = [per_cam[c]["stats"][i] for c in CAMS]
        n_multi = sum(1 for n, _, _ in s if n >= 2)
        near = any(d < NEAR_PX for _, d, _ in s)
        merge = any(m for _, _, m in s)
        bodies = max(n for n, _, _ in s)
        timeline.append({
            "seq": int(per_cam["cam_0"]["seqs"][i]),
            "bodies": int(bodies), "cams_multi": int(n_multi),
            "near": bool(near), "merge": bool(merge),
        })
    blk["timeline"] = timeline
    index["blocks"].append({"day": blk["day"], "file": blk["file"],
                            "n_frames": int(n_frames), "timeline": timeline})
    el = time.time() - t0
    print(f"  [{bi+1}/{len(blocks)}] {blk['day']}/{blk['file']}: {n_frames} frames "
          f"({el:.0f}s elapsed)", flush=True)

(_ROOT / "reports" / "archive_mining_index.json").write_text(json.dumps(index, indent=1))
print(f"wrote reports/archive_mining_index.json ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------------------
# 3. find + rank candidate windows
# ---------------------------------------------------------------------------
candidates = []
for blk in blocks:
    tl = blk["timeline"]
    interesting = [(t["cams_multi"] >= 2 or t["bodies"] >= 2) for t in tl]
    # contiguous runs
    runs, cur = [], None
    for i, flag in enumerate(interesting):
        if flag and cur is None: cur = i
        elif not flag and cur is not None:
            runs.append((cur, i)); cur = None
    if cur is not None: runs.append((cur, len(tl)))
    for a, b in runs:
        dur = (b - a) / SCREEN_HZ
        if dur < MIN_WIN_S: continue
        seg = tl[a:b]
        near_n = sum(1 for t in seg if t["near"])
        merge_n = sum(1 for t in seg if t["merge"])
        score = (near_n + 2 * merge_n) / len(seg) + 0.2 * min(dur / 30.0, 2.0)
        pad = int(PAD_S * SCREEN_HZ)
        a2, b2 = max(0, a - pad), min(len(tl) - 1, b + pad - 1)
        # cap length around the densest part
        max_steps = int(MAX_LEN_S * SCREEN_HZ)
        if b2 - a2 > max_steps:
            b2 = a2 + max_steps
        candidates.append({
            "day": blk["day"], "file": blk["file"], "paths": blk["paths"],
            "seq_start": tl[a2]["seq"], "seq_end": tl[b2]["seq"],
            "duration_s": round((b2 - a2) / SCREEN_HZ, 1),
            "score": round(float(score), 3),
            "near_frac": round(near_n / len(seg), 3),
            "merge_frac": round(merge_n / len(seg), 3),
        })

candidates.sort(key=lambda c: -c["score"])
top = candidates[:TOP_N]
print(f"\ncandidate windows: {len(candidates)} total, verifying top {len(top)} with raw-SSD ...",
      flush=True)

# ---------------------------------------------------------------------------
# 4. SSD verification of top candidates (~1 Hz subsample, cam with most activity)
# ---------------------------------------------------------------------------
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.core.types import Frame
ssd = MobileNetSSDDetector.load(ab.RAW_CKPT)

verified = []
for ci, c in enumerate(top):
    counts = []
    for cam in CAMS:
        with h5py.File(c["paths"][cam]) as h:
            seqs, _cov = read_strided_safe(h["seqs"], 1)
            lo = int(np.searchsorted(seqs, c["seq_start"]))
            hi = int(np.searchsorted(seqs, c["seq_end"]))
            step = 12   # ~1 Hz
            for k in range(lo, hi, step):
                try:
                    data = to_c(h["frames"][k])
                except OSError:
                    continue   # corrupt chunk — skip this sample
                fr = Frame(data=data, timestamp=0.0, camera_id=int(cam[-1]))
                counts.append(len(ssd.predict(fr)))
    mean_p = float(np.mean(counts)) if counts else 0.0
    c["ssd_persons_mean"] = round(mean_p, 2)
    c["id"] = f"{c['day'].replace('2026-', '')}_{ci:02d}"
    if mean_p >= 1.5:
        verified.append(c)
    print(f"  {c['id']}  {c['day']}/{c['file']}  {c['duration_s']}s  score={c['score']} "
          f"ssd={mean_p:.2f}  {'KEEP' if mean_p >= 1.5 else 'drop'}", flush=True)

(_ROOT / "reports" / "archive_candidates.json").write_text(json.dumps(
    {"n_windows_total": len(candidates), "verified": verified}, indent=1))
print(f"\n{len(verified)} verified candidates -> reports/archive_candidates.json", flush=True)
