"""Convert selected archive candidates into annotatable Layout-B sessions.

Usage: python scripts/extract_archive_sessions.py <selection.csv>
  selection.csv lines: candidate_id,extract   (1 = extract)

For each selected candidate (from reports/archive_candidates.json):
  - reads the 3 cams' frames over [seq_start, seq_end] (+ corruption tolerance)
  - aligns cameras by seq number, resamples 12->8 Hz on a 125 ms timestamp grid
  - converts uint16 deci-Kelvin -> float32 degC
  - writes datasets/waveshare_work/arch<MMDD>_<nn>/ matching the
    reorganize_waveshare.py conventions (ch{k}_raw_data.npz key 'frames',
    ch{k}_frames/frame_%05d.png inferno renders, classes.txt, meta.json)

Sessions are then discoverable by DatasetIndex (Layout B) and openable in the
annotator. Consolidated contact_labels.csv / labels.xlsx are NOT touched —
merge after annotation (see scripts/_integrate_2men_clash.py pattern).
"""
import csv, json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import hdf5plugin  # noqa: F401  (zstd filter 32015)
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

CAMS = ("cam_0", "cam_1", "cam_2")
TARGET_DT_US = 125_000          # 8 Hz grid
DST_ROOT = _ROOT / "datasets" / "waveshare_work"

def to_c(u16):
    return u16.astype(np.float32) / 10.0 - 273.15

def read_range_safe(h, lo, hi):
    """frames[lo:hi], seqs, timestamps — tolerant of corrupt chunks (returns
    the readable prefix of the range)."""
    fr, sq, ts = [], [], []
    step = 600
    for a in range(lo, hi, step):
        b = min(a + step, hi)
        try:
            fr.append(h["frames"][a:b])
            sq.append(h["seqs"][a:b])
            ts.append(h["timestamps"][a:b])
        except OSError:
            break
    if not fr:
        return None
    return (np.concatenate(fr, 0), np.concatenate(sq, 0), np.concatenate(ts, 0))

def render_png(frame_c, path, scale=4):
    f = frame_c
    lo, hi = float(f.min()), float(f.max())
    norm = (f - lo) / (hi - lo + 1e-6)
    rgb = (cm.inferno(norm)[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgb).resize((80 * scale, 62 * scale), Image.BICUBIC)
    im.save(path)

selection_path = Path(sys.argv[1])
selected = set()
for row in csv.DictReader(open(selection_path)):
    if row["extract"].strip() == "1":
        selected.add(row["candidate_id"].strip())
cands = {c["id"]: c for c in
         json.loads((_ROOT / "reports" / "archive_candidates.json").read_text())["verified"]}
missing = selected - set(cands)
assert not missing, f"unknown candidate ids: {missing}"
print(f"extracting {len(selected)} of {len(cands)} candidates")

for n, cid in enumerate(sorted(selected)):
    c = cands[cid]
    name = f"arch{c['day'].replace('2026-', '').replace('-', '')}_{cid.split('_')[-1]}"
    dst = DST_ROOT / name
    assert not dst.exists(), f"{dst} already exists"

    # read + align
    streams = {}
    for cam in CAMS:
        with h5py.File(c["paths"][cam]) as h:
            seqs_all, _ = None, None
            try:
                seqs_all = h["seqs"][:]
            except OSError:
                pass
            if seqs_all is not None:
                lo = int(np.searchsorted(seqs_all, c["seq_start"]))
                hi = int(np.searchsorted(seqs_all, c["seq_end"]))
            else:
                lo, hi = 0, h["frames"].shape[0]
            r = read_range_safe(h, lo, hi)
            assert r is not None, f"{cid}/{cam}: nothing readable"
            streams[cam] = r

    # 8 Hz grid on the overlapping time span
    t0 = max(s[2][0] for s in streams.values())
    t1 = min(s[2][-1] for s in streams.values())
    grid = np.arange(t0, t1, TARGET_DT_US)
    chans = []
    for cam in CAMS:
        fr, sq, ts = streams[cam]
        idx = np.searchsorted(ts, grid)
        idx = np.clip(idx, 0, len(ts) - 1)
        prev = np.clip(idx - 1, 0, len(ts) - 1)
        take = np.where(np.abs(ts[idx] - grid) <= np.abs(ts[prev] - grid), idx, prev)
        chans.append(to_c(fr[take]))
    M = min(ch.shape[0] for ch in chans)
    chans = [ch[:M] for ch in chans]

    # write session
    dst.mkdir(parents=True)
    for k, arr in enumerate(chans):
        np.savez_compressed(dst / f"ch{k}_raw_data.npz", frames=arr)
        fdir = dst / f"ch{k}_frames"
        fdir.mkdir()
        for i in range(M):
            render_png(arr[i], fdir / f"frame_{i:05d}.png")
    (dst / "classes.txt").write_text("fire\nperson\n")
    (dst / "meta.json").write_text(json.dumps({
        "scene": name, "sensor": "waveshare", "setup": None,
        "original_timestamp": c["file"].split("_")[0], "fps": 8.0,
        "n_frames_per_ch": {str(k): M for k in range(3)},
        "duplicate_of": None,
        "alignment_fixes": [f"12->8 Hz timestamp-resampled from archive "
                            f"{c['day']}/{c['file']} seq {c['seq_start']}..{c['seq_end']} "
                            f"(candidate {cid}, score {c['score']})"],
        "truncations": [],
    }, indent=2))
    print(f"  [{n+1}/{len(selected)}] {name}: {M} frames x 3 cams "
          f"({M/8:.0f}s) <- {cid}")

print("\ndone — sessions are discoverable by DatasetIndex and openable in the annotator")
