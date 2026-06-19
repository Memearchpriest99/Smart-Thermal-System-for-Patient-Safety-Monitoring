"""
Reorganize the raw Waveshare recordings into the MLX90640 on-disk convention.
=============================================================================
The raw capture layout is::

    datasets/waveshare/<session>/<timestamp>/
        ch0_thermal.npz   (key 'frames', shape (N, 62, 80) float32 degC)
        ch0_video.mp4
        ch0/frame_00000.png ...
        ch1_thermal.npz / ch1_video.mp4 / ch1/...
        ch2_thermal.npz / ch2_video.mp4 / ch2/...

This script copies (never moves / never edits the source) each session into::

    datasets/waveshare_organized/<session>/
        ch0_frames/frame_00000.png ...       (aligned, contiguous)
        ch0_raw_data.npz   (key 'frames', aligned to ch0_frames)
        ch1_frames/ ... ch1_raw_data.npz
        ch2_frames/ ... ch2_raw_data.npz
        ch0_video.mp4 / ch1_video.mp4 / ch2_video.mp4   (carried over as-is)
        classes.txt        ("fire\nperson\n")
        meta.json          (mirrors the MLX schema)

Channel frame counts in the raw capture are slightly misaligned (the PNG count
and the npz length can differ, and ch2 is frequently one frame short).  We
truncate every channel at the tail to a single common length ``M`` so all three
channels share contiguous frame indices ``0 .. M-1`` -- exactly what the
training stack and the annotator assume.

Usage::

    python scripts/reorganize_waveshare.py
    python scripts/reorganize_waveshare.py --src datasets/waveshare --dst datasets/waveshare_organized
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

CHANNELS = (0, 1, 2)
FPS = 8.0
CLASSES = ("fire", "person")
IMG_EXT = ".png"


def _png_frames(ch_dir: Path) -> list[Path]:
    """Sorted list of frame PNGs in a raw chX/ directory."""
    if not ch_dir.is_dir():
        return []
    return sorted(p for p in ch_dir.iterdir() if p.suffix.lower() == IMG_EXT)


def _npz_len(npz_path: Path) -> int:
    if not npz_path.is_file():
        return 0
    with np.load(npz_path) as z:
        return int(z["frames"].shape[0])


def reorganize_session(session_dir: Path, dst_session: Path) -> dict:
    """Reorganize one raw session folder; return its meta.json dict."""
    # The raw session holds exactly one timestamped capture subfolder.
    timestamps = [p for p in session_dir.iterdir() if p.is_dir()]
    if not timestamps:
        raise RuntimeError(f"no timestamp subfolder in {session_dir}")
    capture = sorted(timestamps)[0]
    timestamp = capture.name

    # ── Gather per-channel raw counts ──────────────────────────────────────
    raw_pngs: dict[int, list[Path]] = {}
    raw_npz: dict[int, Path] = {}
    raw_counts: dict[int, dict[str, int]] = {}
    for ch in CHANNELS:
        pngs = _png_frames(capture / f"ch{ch}")
        npz = capture / f"ch{ch}_thermal.npz"
        raw_pngs[ch] = pngs
        raw_npz[ch] = npz
        raw_counts[ch] = {"png": len(pngs), "npz": _npz_len(npz)}

    # ── Common aligned length across every available stream of every ch ────
    lengths = [
        c[k]
        for c in raw_counts.values()
        for k in ("png", "npz")
        if c[k] > 0
    ]
    if not lengths:
        raise RuntimeError(f"no frames found in {capture}")
    m = min(lengths)

    truncations: list[dict] = []
    dst_session.mkdir(parents=True, exist_ok=True)

    # ── Copy frames + npz per channel, truncated to m ──────────────────────
    n_frames_per_ch: dict[str, int] = {}
    for ch in CHANNELS:
        pngs = raw_pngs[ch]
        if not pngs and raw_counts[ch]["npz"] == 0:
            continue  # channel absent entirely

        frames_out = dst_session / f"ch{ch}_frames"
        frames_out.mkdir(exist_ok=True)
        for new_idx, src_png in enumerate(pngs[:m]):
            shutil.copy2(src_png, frames_out / f"frame_{new_idx:05d}{IMG_EXT}")

        # Thermal npz -> chX_raw_data.npz (truncated, key 'frames')
        if raw_npz[ch].is_file():
            with np.load(raw_npz[ch]) as z:
                frames = z["frames"][:m]
            np.savez_compressed(dst_session / f"ch{ch}_raw_data.npz", frames=frames)

        # Carry the colormap preview video over untouched.
        src_mp4 = capture / f"ch{ch}_video.mp4"
        if src_mp4.is_file():
            shutil.copy2(src_mp4, dst_session / f"ch{ch}_video.mp4")

        n_frames_per_ch[str(ch)] = m

        for stream in ("png", "npz"):
            orig = raw_counts[ch][stream]
            if orig > m:
                truncations.append(
                    {"channel": ch, "stream": stream, "from": orig, "to": m}
                )

    # ── classes.txt ────────────────────────────────────────────────────────
    (dst_session / "classes.txt").write_text(
        "\n".join(CLASSES) + "\n", encoding="utf-8"
    )

    # ── meta.json (mirrors the MLX schema) ──────────────────────────────────
    meta = {
        "scene": session_dir.name,
        "sensor": "waveshare",
        "setup": None,
        "original_timestamp": timestamp,
        "fps": FPS,
        "n_frames_per_ch": n_frames_per_ch,
        "duplicate_of": None,
        "alignment_fixes": [],
        "truncations": truncations,
    }
    (dst_session / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    meta["_raw_counts"] = raw_counts  # for the summary table only
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="datasets/waveshare")
    ap.add_argument("--dst", default="datasets/waveshare_organized")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"source not found: {src}")

    sessions = sorted(p for p in src.iterdir() if p.is_dir())
    print(f"Reorganizing {len(sessions)} session(s): {src}  ->  {dst}\n")

    header = f"{'session':<24} {'ch0 (raw->aln)':>16} {'ch1':>10} {'ch2':>10}"
    print(header)
    print("-" * len(header))

    for session_dir in sessions:
        dst_session = dst / session_dir.name
        meta = reorganize_session(session_dir, dst_session)
        rc = meta["_raw_counts"]
        m = next(iter(meta["n_frames_per_ch"].values()), 0)

        def cell(ch: int) -> str:
            png = rc[ch]["png"]
            return f"{png}->{m}" if png else "-"

        print(
            f"{session_dir.name:<24} {cell(0):>16} {cell(1):>10} {cell(2):>10}"
        )

    print(f"\nDone. Organized dataset at: {dst.resolve()}")


if __name__ == "__main__":
    main()
