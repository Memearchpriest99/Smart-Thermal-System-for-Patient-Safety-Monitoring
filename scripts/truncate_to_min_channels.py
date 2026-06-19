"""Truncate scenes with per-channel frame-count mismatches to the common minimum.

A few scenes have one channel that dropped a frame during capture (e.g.
personrunning 34/33/34, setup2_1_man_running 94/95/95). This aligns ALL channels
in such a scene to the shortest channel's length so every frame index has a
synchronized 3-camera triplet.

For each over-length channel it:
  * truncates ch{N}_raw_data.npz to the common minimum M (drops trailing frames)
  * deletes frame_XXXXX.png / .txt for indices >= M
and updates the scene's meta.json (n_frames_per_ch + a 'truncations' record).

Operates on dataset/ only (the raw backup is untouched). Auto-detects which
scenes need it; idempotent (already-uniform scenes are skipped).

Note: this does NOT edit labels.xlsx. xlsx rows for the dropped (channel, frame)
become orphans (they show up under D5 in the agreement report) — handle
separately if you want them removed.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATASET = Path("dataset")
CHANNELS = (0, 1, 2)


def channel_len(scene_dir: Path, ch: int) -> int:
    return int(np.load(scene_dir / f"ch{ch}_raw_data.npz")["frames"].shape[0])


def main() -> None:
    scenes = sorted(d for d in DATASET.iterdir() if d.is_dir())
    total_truncated = 0
    for scene_dir in scenes:
        if not (scene_dir / "ch0_raw_data.npz").is_file():
            continue
        lengths = {ch: channel_len(scene_dir, ch) for ch in CHANNELS}
        if len(set(lengths.values())) == 1:
            continue  # already uniform
        m = min(lengths.values())
        print(f"{scene_dir.name}: channels {lengths} -> truncating all to {m}")
        record = []
        for ch in CHANNELS:
            if lengths[ch] == m:
                continue
            # 1. truncate npz
            npz_path = scene_dir / f"ch{ch}_raw_data.npz"
            arr = np.load(npz_path)["frames"]
            np.savez_compressed(npz_path, frames=arr[:m])
            # 2. delete surplus png/txt (indices >= m)
            removed = 0
            fdir = scene_dir / f"ch{ch}_frames"
            for ext in ("png", "txt"):
                for f in glob.glob(str(fdir / f"frame_*.{ext}")):
                    idx = int(Path(f).stem.split("_")[1])
                    if idx >= m:
                        Path(f).unlink()
                        removed += 1
            print(f"    ch{ch}: {lengths[ch]} -> {m}  (removed {removed} png+txt files)")
            record.append({"channel": ch, "from": lengths[ch], "to": m})
            total_truncated += 1

        # 3. update meta.json
        meta_path = scene_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["n_frames_per_ch"] = {str(ch): m for ch in CHANNELS}
        meta.setdefault("truncations", []).extend(record)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone. Channels truncated: {total_truncated}")


if __name__ == "__main__":
    main()
