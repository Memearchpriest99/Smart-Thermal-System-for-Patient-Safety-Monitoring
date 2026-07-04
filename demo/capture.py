"""Offscreen capture of the demo UI — no window, no screen grab.

Runs the real source + engine threads for a while, then composes the exact
UI layout (header, three panels, contact strip, footer) as a PIL image.
Used to verify the demo end-to-end on machines without a display and to
produce documentation shots.

    python -m demo.capture datasets/waveshare_work/2ppl_fight \
        --after 14 --out outputs/demo/demo_contact.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw

from demo import overlay
from demo.engine import DetectionEngine
from demo.sources import ReplaySource

PANEL_SCALE = 6


def compose_window(snap: dict, source_name: str, session: str,
                   threshold: float) -> Image.Image:
    pw, ph = 80 * PANEL_SCALE, 62 * PANEL_SCALE
    pad, gap = 18, 12
    cap_h, head_h, strip_h, foot_h = 30, 46, 96, 26
    width = 3 * (pw + 4) + 2 * gap + 2 * pad
    height = head_h + (ph + 4 + cap_h) + strip_h + foot_h + 4 * 10

    img = Image.new("RGB", (width, height), overlay.BG)
    d = ImageDraw.Draw(img)

    # header
    d.text((pad, 12), "SMART THERMAL MONITOR",
           font=overlay._font(18, bold=True), fill=overlay.TEXT)
    d.text((pad + 260, 19), "fire · person · contact — privacy-preserving "
           "thermal sensing", font=overlay._font(11), fill=overlay.MUTED)
    chip = f" {source_name} "
    d.rounded_rectangle([width - pad - 78, 12, width - pad, 34], radius=6,
                        fill=(29, 39, 51))
    d.text((width - pad - 66, 16), chip.strip(),
           font=overlay._font(11, bold=True), fill=overlay.TEAL)
    d.text((width - pad - 160, 17), time.strftime("%H:%M:%S"),
           font=overlay._font(11), fill=overlay.MUTED)

    # panels
    y0 = head_h + 10
    for cam in range(3):
        c = snap["cams"][cam]
        panel = overlay.render_panel(c["raw"], c["boxes"], c["fire"],
                                     c["fire_bbox"], scale=PANEL_SCALE)
        x0 = pad + cam * (pw + 4 + gap)
        d.rectangle([x0 - 1, y0 - 1, x0 + pw + 2, y0 + ph + cap_h + 2],
                    outline=overlay.PANEL_EDGE, fill=overlay.PANEL)
        img.paste(panel, (x0 + 1, y0 + 1))
        n = len(c["boxes"])
        parts = [f"CAMERA {cam}", "·", f"{n} person{'s' if n != 1 else ''}"]
        if not c["bg_ready"]:
            parts += ["·", "calibrating background"]
        if c["fire"]:
            parts += ["·", f"FIRE {c['fire_conf']:.2f}"]
        d.text((x0 + 9, y0 + ph + 8), "  ".join(parts),
               font=overlay._font(11, bold=True),
               fill=overlay.AMBER if c["fire"] else overlay.MUTED)

    # contact strip
    ct = snap["contact"]
    strip = overlay.render_contact_strip(
        ct["history"], ct["conf"], ct["alarmed"], threshold,
        width - 2 * pad, strip_h)
    img.paste(strip, (pad, y0 + ph + cap_h + 14))

    # footer
    cam_ms = max(c["proc_ms"] for c in snap["cams"])
    d.text((pad, height - foot_h),
           f"session: {session}    tick {snap['tick']}  @ "
           f"{snap['tick_fps']:.1f} Hz      per-camera {cam_ms:.1f} ms   "
           f"contact {ct['proc_ms']:.1f} ms      models: FireSVM · "
           f"MobileNet-SSD (onnx) · Thermo-X3D T5v2 (onnx-ftz)",
           font=overlay._font(10), fill=overlay.MUTED)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--after", type=float, default=12.0)
    ap.add_argument("--wait-alarm", action="store_true",
                    help="capture at the first contact alarm (timeout=--after)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--checkpoints", type=Path, default=_ROOT / "checkpoints")
    args = ap.parse_args()

    engine = DetectionEngine(args.checkpoints)
    engine.start()
    source = ReplaySource(args.sessions, fps=args.fps)
    source.start(engine.submit)
    if args.wait_alarm:
        deadline = time.monotonic() + args.after
        snap = engine.snapshot()
        while time.monotonic() < deadline:
            snap = engine.snapshot()
            if snap["contact"]["alarmed"]:
                time.sleep(0.15)   # let the panels catch the same moment
                snap = engine.snapshot()
                break
            time.sleep(0.1)
    else:
        time.sleep(args.after)
        snap = engine.snapshot()
    source.stop()
    engine.stop()

    img = compose_window(snap, source.name, source.current_session,
                         engine._contact.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"capture -> {out}  (tick {snap['tick']}, "
          f"contact conf {snap['contact']['conf']}, "
          f"alarmed {snap['contact']['alarmed']})")


if __name__ == "__main__":
    main()
