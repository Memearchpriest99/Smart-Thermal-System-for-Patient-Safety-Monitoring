"""Reorganize the raw thermal dataset into a clean, normalized `dataset/` tree.

Reads:  dataset_raw/<scene>/[<timestamp>/]ch{0,1,2}_{raw_data.npz, video.mp4, frames/}
Writes: dataset/<scene>/ch{0,1,2}_raw_data.npz + ch{N}_frames/(png+txt) + classes.txt + meta.json

What it does (see plan / dataset/CLAUDE.md for rationale):
  * Flattens the two folder layouts (timestamped subfolder vs. channels-at-scene-root)
    into one uniform flat layout (one session per scene).
  * Drops .mp4 videos and the redundant per-`ch*_frames/classes.txt`.
  * Writes one canonical `classes.txt` = "fire\\nperson" per scene.
  * FIXES npz<->png<->txt off-by-one desyncs: aligns each channel to the longest
    contiguous prefix 0..M-1 present in npz AND png AND txt, truncating the npz
    array and dropping orphan png/txt accordingly.  Every fix is recorded.
  * Detects byte-identical duplicate recordings (md5 over the 3 npz arrays) and
    records `duplicate_of` in meta.json — does NOT delete them (their xlsx
    annotations differ; left for manual review).
  * Tags each scene with its camera `setup` (2 if name starts "setup2_", else 1).
  * Leaves CORRUPT (single-pixel-spike) frames fully intact.
  * Emits dataset/_manifest.csv summarizing every (scene, channel).

The raw tree is read-only here; nothing under dataset_raw/ is modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252; force UTF-8 so progress output never crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAW_ROOT = Path("dataset_raw")
OUT_ROOT = Path("dataset")
CANONICAL_CLASSES = "fire\nperson\n"
FPS = 8.0
CHANNELS = (0, 1, 2)
_TS_RE = re.compile(r"^\d{8}_\d{6}$")
_FRAME_RE = re.compile(r"frame_(\d+)")


def find_session_dir(scene_dir: Path) -> Path:
    """Return the directory holding ch*_raw_data.npz (timestamp subfolder or scene root)."""
    if (scene_dir / "ch0_raw_data.npz").is_file():
        return scene_dir
    for sub in sorted(scene_dir.iterdir()):
        if sub.is_dir() and (sub / "ch0_raw_data.npz").is_file():
            return sub
    raise FileNotFoundError(f"No ch0_raw_data.npz found under {scene_dir}")


def frame_indices(frames_dir: Path, ext: str) -> set[int]:
    out: set[int] = set()
    if not frames_dir.is_dir():
        return out
    for f in frames_dir.glob(f"*.{ext}"):
        m = _FRAME_RE.search(f.name)
        if m:
            out.add(int(m.group(1)))
    return out


def aligned_length(npz_n: int, png: set[int], txt: set[int]) -> int:
    """Longest contiguous prefix 0..M-1 present in npz range AND png AND txt."""
    m = 0
    while m < npz_n and m in png and m in txt:
        m += 1
    return m


def npz_md5(path: Path) -> str:
    arr = np.load(path)["frames"]
    return hashlib.md5(arr.tobytes()).hexdigest()


def main() -> None:
    if not RAW_ROOT.is_dir():
        raise SystemExit(f"Raw dataset not found at {RAW_ROOT.resolve()}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    scenes = sorted(
        d.name for d in RAW_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )
    print(f"Found {len(scenes)} scenes under {RAW_ROOT}/")

    # Pass 1: fingerprint each scene's npz triple to detect duplicates.
    fingerprints: dict[str, tuple[str, str, str]] = {}
    for scene in scenes:
        sess = find_session_dir(RAW_ROOT / scene)
        fingerprints[scene] = tuple(
            npz_md5(sess / f"ch{ch}_raw_data.npz") for ch in CHANNELS
        )  # type: ignore[assignment]
    duplicate_of: dict[str, str | None] = {s: None for s in scenes}
    seen: dict[tuple, str] = {}
    for scene in scenes:
        fp = fingerprints[scene]
        if fp in seen:
            duplicate_of[scene] = seen[fp]
            duplicate_of[seen[fp]] = scene  # point both ways
        else:
            seen[fp] = scene

    manifest_rows: list[dict] = []

    # Pass 2: build the clean copy.
    for scene in scenes:
        raw_sess = find_session_dir(RAW_ROOT / scene)
        out_scene = OUT_ROOT / scene
        out_scene.mkdir(parents=True, exist_ok=True)

        original_ts = raw_sess.name if _TS_RE.match(raw_sess.name) else None
        setup = 2 if scene.startswith("setup2_") else 1
        n_frames_per_ch: dict[str, int] = {}
        alignment_fixes: list[dict] = []

        for ch in CHANNELS:
            npz_path = raw_sess / f"ch{ch}_raw_data.npz"
            frames_dir = raw_sess / f"ch{ch}_frames"
            arr = np.load(npz_path)["frames"]
            npz_n = int(arr.shape[0])
            png = frame_indices(frames_dir, "png")
            txt = frame_indices(frames_dir, "txt")

            m = aligned_length(npz_n, png, txt)
            dropped_npz = npz_n - m
            dropped_png = sum(1 for i in png if i >= m)
            dropped_txt = sum(1 for i in txt if i >= m)
            if dropped_npz or dropped_png or dropped_txt:
                alignment_fixes.append({
                    "channel": ch, "npz_orig": npz_n, "png_orig": len(png),
                    "txt_orig": len(txt), "aligned_to": m,
                    "dropped_npz_frames": dropped_npz,
                    "dropped_png": dropped_png, "dropped_txt": dropped_txt,
                })

            # Write aligned npz (truncate to first M frames).
            out_npz = out_scene / f"ch{ch}_raw_data.npz"
            np.savez_compressed(out_npz, frames=arr[:m])

            # Copy aligned png + txt (indices 0..M-1 only).
            out_fdir = out_scene / f"ch{ch}_frames"
            out_fdir.mkdir(exist_ok=True)
            n_label_pos = 0
            for i in range(m):
                src_png = frames_dir / f"frame_{i:05d}.png"
                src_txt = frames_dir / f"frame_{i:05d}.txt"
                if src_png.is_file():
                    shutil.copy2(src_png, out_fdir / src_png.name)
                if src_txt.is_file():
                    shutil.copy2(src_txt, out_fdir / src_txt.name)
                    if src_txt.stat().st_size > 0:
                        n_label_pos += 1

            n_frames_per_ch[str(ch)] = m
            manifest_rows.append({
                "scene": scene, "setup": setup, "channel": ch,
                "n_npz": m, "n_png": m, "n_txt": m,
                "n_label_pos": n_label_pos,
                "npz_orig": npz_n, "png_orig": len(png), "txt_orig": len(txt),
                "alignment_fixed": bool(dropped_npz or dropped_png or dropped_txt),
                "duplicate_of": duplicate_of[scene] or "",
            })

        # Canonical classes.txt + meta.json (drop mp4 + per-frame classes.txt).
        (out_scene / "classes.txt").write_text(CANONICAL_CLASSES, encoding="utf-8")
        meta = {
            "scene": scene,
            "setup": setup,
            "original_timestamp": original_ts,
            "fps": FPS,
            "n_frames_per_ch": n_frames_per_ch,
            "duplicate_of": duplicate_of[scene],
            "alignment_fixes": alignment_fixes,
        }
        (out_scene / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        flag = ""
        if duplicate_of[scene]:
            flag += f"  [DUP of {duplicate_of[scene]}]"
        if alignment_fixes:
            flag += f"  [FIXED {len(alignment_fixes)} ch]"
        print(f"  {scene:32s} setup{setup}  frames/ch={n_frames_per_ch}{flag}")

    # Write manifest.
    manifest_path = OUT_ROOT / "_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        cols = ["scene", "setup", "channel", "n_npz", "n_png", "n_txt",
                "n_label_pos", "npz_orig", "png_orig", "txt_orig",
                "alignment_fixed", "duplicate_of"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(manifest_rows)

    n_fixed = sum(1 for r in manifest_rows if r["alignment_fixed"])
    n_dups = sum(1 for s in scenes if duplicate_of[s])
    print(f"\nDone. {len(scenes)} scenes → {OUT_ROOT}/")
    print(f"  alignment-fixed channels: {n_fixed}")
    print(f"  duplicate scenes flagged: {n_dups}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
