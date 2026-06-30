"""Entry point for the Live Thermal Monitor.

    python -m apps.live_monitor.main --source synthetic
    python -m apps.live_monitor.main --source playback --session datasets/waveshare_work/<scene>
    python -m apps.live_monitor.main --source senxor          # Raspberry Pi only

Run with ``QT_QPA_PLATFORM=offscreen`` for a headless smoke test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thermal_algorithms.core.sensor_profile import PROFILES, WAVESHARE_26984
from thermal_algorithms.core.checkpoints import CheckpointRegistry

from apps.live_monitor.capture import PlaybackSource, SyntheticSource
from apps.live_monitor.detectors import default_checkpoint_root
from apps.live_monitor.runner import PipelineRunner


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Thermal Monitor (RPi5)")
    p.add_argument("--source", choices=["synthetic", "playback", "senxor"], default="synthetic")
    p.add_argument("--profile", choices=list(PROFILES), default=WAVESHARE_26984.name)
    p.add_argument("--session", type=str, default=None, help="dataset session dir for --source playback")
    p.add_argument("--checkpoints", type=str, default=None, help="checkpoint registry root")
    p.add_argument("--restricted", action="store_true", help="start in restricted-area mode")
    p.add_argument("--target-fps", type=float, default=None)
    p.add_argument("--smoke", action="store_true", help="build everything, process a few frames, exit")
    return p.parse_args(argv)


def build_sources(args, profile):
    if args.source == "synthetic":
        return [SyntheticSource(i, profile, with_fire=(i == 0), n_actors=2, seed=7) for i in range(3)]
    if args.source == "playback":
        if not args.session:
            raise SystemExit("--source playback requires --session <dir>")
        session = Path(args.session)
        return [PlaybackSource(i, profile, session_dir=session) for i in range(3)]
    if args.source == "senxor":
        from apps.live_monitor.capture.senxor_source import SenxorSource
        return [SenxorSource(i, profile) for i in range(3)]
    raise SystemExit(f"unknown source {args.source!r}")


def main(argv=None) -> int:
    args = _parse_args(argv)
    profile = PROFILES[args.profile]
    target_fps = args.target_fps or profile.sample_rate_hz
    registry = CheckpointRegistry(args.checkpoints or default_checkpoint_root())
    sources = build_sources(args, profile)
    runner = PipelineRunner(sources, profile, registry, restricted=args.restricted)

    if args.smoke:
        return _smoke(runner)

    # Import Qt lazily so non-UI smoke runs need no display.
    from PyQt6.QtWidgets import QApplication
    from apps.live_monitor.pipeline_worker import PipelineWorker, configure_cpu, make_worker_thread
    from apps.live_monitor.ui.main_window import MainWindow

    cpu = configure_cpu()
    app = QApplication(sys.argv[:1])
    worker = PipelineWorker(runner, sources, target_fps=target_fps)
    thread = make_worker_thread(worker)
    default_dir = Path(args.session) if args.session else Path.cwd()
    win = MainWindow(runner, worker, default_save_dir=default_dir)
    win.statusBar().showMessage(
        f"source={args.source}  profile={profile.name}  cpu={cpu}", 8000)
    win.resize(1280, 760)
    win.show()
    thread.start()
    return app.exec()


def _smoke(runner: PipelineRunner) -> int:
    """Headless: start sources, process a handful of triplets, report, exit."""
    import time

    for s in runner._sources:  # noqa: SLF001 - internal smoke helper
        s.start()
    processed = 0
    deadline = time.monotonic() + 8.0
    while processed < 5 and time.monotonic() < deadline:
        rr = runner.poll()
        if rr is not None:
            processed += 1
            r = rr.result
            print(f"[smoke] #{processed} {rr.timings_ms['pipeline']:.1f}ms "
                  f"dets={[len(d) for d in r.detections]} "
                  f"fire={r.fire_alarm} contact={r.contact_alarm}")
        else:
            time.sleep(0.005)
    runner.stop()
    print(f"[smoke] selection={runner.selection}")
    return 0 if processed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
