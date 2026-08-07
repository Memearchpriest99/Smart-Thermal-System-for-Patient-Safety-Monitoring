#!/usr/bin/env python3
"""Pinpoint the exact frame that crashes FireSVMDetector.fit() during the
full-corpus run. Mirrors train_fire()'s example stream but logs each frame's
identity/stats BEFORE processing it, unbuffered, so the last line before a
native crash tells us exactly which frame is the culprit (faulthandler's own
post-mortem traceback can be corrupted by the access violation itself).

Usage: python scripts/_debug_fire_crash.py [--stop-after N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

import numpy as np  # noqa: E402

from thermal_algorithms.core.sensor_profile import WAVESHARE_26984  # noqa: E402
from thermal_algorithms.preprocessing.global_norm_pipeline import GlobalNormPreprocessor  # noqa: E402
from thermal_algorithms.fire_detection.fire_svm import _frame_to_feature_vector  # noqa: E402
from thermal_algorithms.training import (  # noqa: E402
    DatasetIndex,
    FireFrameDataset,
    iter_synth_sessions,
    session_train_test_split,
)

DATA_ROOT = _REPO_ROOT.parent / "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-after", type=int, default=90_000)
    args = ap.parse_args()

    waveshare_index = DatasetIndex(DATA_ROOT / "waveshare_work", sensor_profile=WAVESHARE_26984, fps=8.0)
    train_sessions, _ = session_train_test_split(waveshare_index.labeled_sessions())
    train_scenes = {s.scene for s in train_sessions}

    preprocessor = GlobalNormPreprocessor(WAVESHARE_26984)
    preprocessor.fit()

    morph_kernel = np.ones((3, 3), dtype=np.uint8)

    def examples():
        yield "waveshare", None, from_waveshare()
        for session in iter_synth_sessions(DATA_ROOT):
            yield session.scene, session, session.fire_examples()

    def from_waveshare():
        yield from FireFrameDataset(waveshare_index, scenes=train_scenes)

    i = 0
    for source_label, _session, stream in examples():
        for frame, alert in stream:
            i += 1
            d = frame.data
            print(
                f"#{i:>7} src={source_label!r:<20} cam={frame.camera_id} ts={frame.timestamp:.3f} "
                f"shape={d.shape} dtype={d.dtype} contig={d.flags['C_CONTIGUOUS']} "
                f"min={float(np.nanmin(d)):.2f} max={float(np.nanmax(d)):.2f} "
                f"nan={bool(np.isnan(d).any())} inf={bool(np.isinf(d).any())}",
                flush=True,
            )
            proc = preprocessor.predict(frame)
            fv = _frame_to_feature_vector(proc.data.astype(np.float32), morph_kernel, 256)
            if i >= args.stop_after:
                print(f"Reached --stop-after={args.stop_after} without crashing.", flush=True)
                return 0
    print(f"Stream exhausted after {i} frames without crashing.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
