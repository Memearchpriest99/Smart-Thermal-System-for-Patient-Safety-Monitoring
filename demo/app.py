"""Smart Thermal System — live demo UI.

Three thermal feeds with person / fire overlays and contact (touch)
detection on a strip at the bottom. All inference runs on worker threads
(see demo/engine.py); the UI thread only renders.

    python -m demo.app --replay datasets/waveshare_work/2ppl_fight
    python -m demo.app --replay <dir1> <dir2> ...      # cycles sessions
    python -m demo.app --live --config demo/cams_pi.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from PIL import ImageTk

from demo import overlay
from demo.engine import DetectionEngine
from demo.sources import LiveMI48Source, ReplaySource

REFRESH_MS = 70
PANEL_SCALE = 6


class DemoApp:
    def __init__(self, root: tk.Tk, source, engine: DetectionEngine,
                 screenshot_after: float = 0.0, screenshot_path: str = "",
                 exit_after: float = 0.0) -> None:
        self.root = root
        self.source = source
        self.engine = engine
        self._t0 = time.monotonic()
        self._shot_after = screenshot_after
        self._shot_path = screenshot_path
        self._shot_done = False
        self._exit_after = exit_after
        self._imgs = [None, None, None]
        self._strip_img = None

        root.title("Smart Thermal System — Patient Safety Monitor")
        root.configure(bg=overlay.BG)
        root.resizable(False, False)

        # ---- header ---------------------------------------------------
        header = tk.Frame(root, bg=overlay.BG)
        header.pack(fill="x", padx=18, pady=(14, 8))
        tk.Label(header, text="SMART THERMAL MONITOR", bg=overlay.BG,
                 fg=overlay.TEXT, font=("Segoe UI", 16, "bold")).pack(side="left")
        tk.Label(header, text="  fire · person · contact — privacy-preserving "
                 "thermal sensing", bg=overlay.BG, fg=overlay.MUTED,
                 font=("Segoe UI", 10)).pack(side="left", pady=(4, 0))
        self.mode_chip = tk.Label(header, text=f" {source.name} ", bg="#1d2733",
                                  fg=overlay.TEAL, font=("Segoe UI", 10, "bold"))
        self.mode_chip.pack(side="right")
        self.clock = tk.Label(header, text="", bg=overlay.BG, fg=overlay.MUTED,
                              font=("Segoe UI", 10))
        self.clock.pack(side="right", padx=12)

        # ---- camera panels ---------------------------------------------
        row = tk.Frame(root, bg=overlay.BG)
        row.pack(padx=18, pady=4)
        self.panels = []
        self.captions = []
        for cam in range(3):
            col = tk.Frame(row, bg=overlay.PANEL,
                           highlightthickness=1,
                           highlightbackground=overlay.PANEL_EDGE)
            col.grid(row=0, column=cam, padx=6)
            lbl = tk.Label(col, bg=overlay.PANEL, bd=0)
            lbl.pack(padx=2, pady=2)
            cap = tk.Label(col, text=f"CAMERA {cam}", bg=overlay.PANEL,
                           fg=overlay.MUTED, font=("Segoe UI", 10, "bold"),
                           anchor="w")
            cap.pack(fill="x", padx=10, pady=(0, 6))
            self.panels.append(lbl)
            self.captions.append(cap)

        # ---- contact strip ----------------------------------------------
        self.strip = tk.Label(root, bg=overlay.BG, bd=0)
        self.strip.pack(padx=18, pady=(8, 4))

        # ---- footer ------------------------------------------------------
        self.footer = tk.Label(root, text="", bg=overlay.BG, fg=overlay.MUTED,
                               font=("Segoe UI", 9), anchor="w")
        self.footer.pack(fill="x", padx=20, pady=(0, 10))

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(REFRESH_MS, self.refresh)

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        s = self.engine.snapshot()

        for cam in range(3):
            c = s["cams"][cam]
            img = overlay.render_panel(c["raw"], c["boxes"], c["fire"],
                                       c["fire_bbox"], scale=PANEL_SCALE)
            self._imgs[cam] = ImageTk.PhotoImage(img)
            self.panels[cam].configure(image=self._imgs[cam])

            n = len(c["boxes"])
            parts = [f"CAMERA {cam}", "·",
                     f"{n} person{'s' if n != 1 else ''}"]
            if not c["bg_ready"]:
                parts += ["·", "calibrating background"]
            if c["fire"]:
                parts += ["·", f"FIRE {c['fire_conf']:.2f}"]
            self.captions[cam].configure(
                text="  ".join(parts),
                fg=overlay.AMBER if c["fire"] else overlay.MUTED)

        ct = s["contact"]
        width = 3 * (80 * PANEL_SCALE + 4) + 2 * 12
        strip = overlay.render_contact_strip(
            ct["history"], ct["conf"], ct["alarmed"],
            self.engine._contact.threshold, width)
        self._strip_img = ImageTk.PhotoImage(strip)
        self.strip.configure(image=self._strip_img)

        cam_ms = max(c["proc_ms"] for c in s["cams"])
        self.footer.configure(text=(
            f"session: {self.source.current_session}    "
            f"tick {s['tick']}  @ {s['tick_fps']:.1f} Hz      "
            f"per-camera {cam_ms:.1f} ms   contact {ct['proc_ms']:.1f} ms      "
            f"models: FireSVM · MobileNet-SSD (onnx) · Thermo-X3D T5v2 (onnx-ftz)"))
        self.clock.configure(text=time.strftime("%H:%M:%S"))

        elapsed = time.monotonic() - self._t0
        if (self._shot_after and not self._shot_done
                and elapsed >= self._shot_after):
            self._shot_done = True
            self._screenshot()
        if self._exit_after and elapsed >= self._exit_after:
            self.close()
            return
        self.root.after(REFRESH_MS, self.refresh)

    def _screenshot(self) -> None:
        from PIL import ImageGrab

        self.root.update_idletasks()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        Path(self._shot_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(self._shot_path)
        print(f"screenshot -> {self._shot_path}")

    def close(self) -> None:
        self.source.stop()
        self.engine.stop()
        self.root.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Smart Thermal System demo")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", nargs="+", metavar="SESSION_DIR",
                      help="loop recorded session folder(s)")
    mode.add_argument("--live", action="store_true",
                      help="acquire from 3 MI48 cameras")
    ap.add_argument("--config", type=Path, default=None,
                    help="JSON list of MI48CameraConfig overrides (live mode)")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--checkpoints", type=Path, default=_ROOT / "checkpoints")
    ap.add_argument("--screenshot-after", type=float, default=0.0)
    ap.add_argument("--screenshot-path", type=str,
                    default=str(_ROOT / "outputs" / "demo" / "demo.png"))
    ap.add_argument("--exit-after", type=float, default=0.0)
    args = ap.parse_args()

    if args.live:
        from thermal_algorithms.acquisition import MI48CameraConfig

        if args.config:
            overrides = json.loads(args.config.read_text())
            configs = [MI48CameraConfig(**{"fps": args.fps, **o})
                       for o in overrides]
        else:
            raise SystemExit("--live requires --config with 3 camera configs")
        source = LiveMI48Source(configs, fps=args.fps)
    else:
        source = ReplaySource(args.replay, fps=args.fps)

    engine = DetectionEngine(args.checkpoints)
    engine.start()
    source.start(engine.submit)

    root = tk.Tk()
    DemoApp(root, source, engine,
            screenshot_after=args.screenshot_after,
            screenshot_path=args.screenshot_path,
            exit_after=args.exit_after)
    root.mainloop()


if __name__ == "__main__":
    main()
