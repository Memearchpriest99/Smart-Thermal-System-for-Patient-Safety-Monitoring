"""Record a session from the Waveshare Thermal Camera Module(s) on the Pi.

Uses the acquisition pipeline from the Waveshare wiki
(https://www.waveshare.com/wiki/Thermal-Camera-Module): I2C register
config + SPI frame readout gated on DATA_READY, via pysenxor.

Output follows the repo's raw capture layout, ready for
``scripts/reorganize_waveshare.py`` and the annotator::

    <root>/<scene>/<YYYYMMDD_HHMMSS>/ch{N}_thermal.npz  (+ ch{N}/*.png)

Usage (on the Raspberry Pi)::

    # Single camera, wiki default wiring, 60 s at 8 fps
    python scripts/acquire_mi48.py 2ppl_walk --duration 60

    # Three cameras from a wiring config file
    python scripts/acquire_mi48.py 2ppl_walk --duration 60 --config cams.json

The config file is a JSON list of MI48CameraConfig overrides, e.g.::

    [
      {"camera_id": 0, "i2c_address": 64, "spi_device": 0, "spi_cs_pin": "BCM7",
       "data_ready_pin": "BCM24", "reset_pin": "BCM23"},
      {"camera_id": 1, "i2c_address": 65, "spi_device": 1, "spi_cs_pin": "BCM8",
       "data_ready_pin": "BCM25", "reset_pin": "BCM22"}
    ]

Prerequisites (see the wiki's Bookworm tutorial):
  - SPI + I2C enabled via ``sudo raspi-config``
  - ``dtoverlay=spi0-0cs`` added under ``dtparam=spi=on`` in
    /boot/firmware/config.txt (CS is driven manually over GPIO)
  - pysenxor installed: unzip Thermal_Camera_Hat.zip; pip install -e pysenxor-master/
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermal_algorithms.acquisition import (
    MI48Camera,
    MI48CameraConfig,
    SessionRecorder,
)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "waveshare"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("scene", help="Scene name, e.g. '2ppl_walk'")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--duration", type=float, help="Recording length in seconds")
    group.add_argument("--frames", type=int, help="Number of frames to record")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"Dataset root (default: {DEFAULT_ROOT})")
    p.add_argument("--fps", type=float, default=8.0,
                   help="MI48 frame rate (default: 8, project standard)")
    p.add_argument("--config", type=Path, default=None,
                   help="JSON list of per-camera MI48CameraConfig overrides")
    p.add_argument("--hflip", action="store_true",
                   help="Horizontally flip frames (forward-looking mount)")
    p.add_argument("--no-png", dest="png", action="store_false",
                   help="Skip the colormapped preview PNGs")
    return p.parse_args()


def build_configs(args: argparse.Namespace) -> list[MI48CameraConfig]:
    if args.config is None:
        return [MI48CameraConfig(fps=args.fps, hflip=args.hflip)]
    overrides = json.loads(args.config.read_text())
    return [
        MI48CameraConfig(**{"fps": args.fps, "hflip": args.hflip, **o})
        for o in overrides
    ]


def main() -> None:
    args = parse_args()
    configs = build_configs(args)

    with contextlib.ExitStack() as stack:
        cameras = []
        for cfg in configs:
            cam = stack.enter_context(MI48Camera(cfg))
            cam.start()
            cameras.append(cam)
            print(f"camera {cfg.camera_id}: streaming at {cfg.fps} fps "
                  f"(I2C 0x{cfg.i2c_address:02X}, SPI {cfg.spi_bus}.{cfg.spi_device})")

        recorder = SessionRecorder(
            cameras, root=args.root, scene=args.scene, write_pngs=args.png
        )
        print(f"Recording to {recorder.session_dir} — Ctrl+C to stop early.")

        def on_tick(i: int, frames: dict) -> None:
            if i % 40 == 0:  # every ~5 s at 8 fps
                temps = {ch: f"{fr.data.min():.1f}..{fr.data.max():.1f}°C"
                         for ch, fr in frames.items()}
                print(f"frame {i:5d}  {temps}")

        out = recorder.record(
            duration_s=args.duration, n_frames=args.frames, on_tick=on_tick
        )
        print(f"Saved {recorder.n_frames} frames/channel to {out}")


if __name__ == "__main__":
    main()
