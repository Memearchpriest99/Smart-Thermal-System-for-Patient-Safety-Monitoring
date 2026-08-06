#!/usr/bin/env python3
"""Split the consolidated waveshare_work ``contact_labels.csv`` back into
per-session files.

``consolidate_waveshare_labels.py`` merges per-session ``contact_labels.csv``
files into one dataset-level file (with a leading ``session`` column) and then
deletes the per-session originals. ``ContactFrameDataset``/``DatasetIndex``
(``thermal_algorithms/training/datasets.py``) only ever read a *per-session*
``contact_labels.csv`` at each session's root — they have no knowledge of the
consolidated format. This script reverses the merge so the dataset is usable
by the training stack again, without touching the consolidated file (kept as
the source of truth / for xlsx-style reporting).

If a session already has its own ``contact_labels.csv`` (e.g. restored
separately), its content is verified against the consolidated file rather than
overwritten; a mismatch aborts loudly rather than silently picking a winner.

Usage::

    python scripts/split_waveshare_contact_labels.py
    python scripts/split_waveshare_contact_labels.py --root path/to/waveshare_work
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thermal_algorithms.training.label_io import load_contact_labels, save_contact_labels  # noqa: E402

DEFAULT_ROOT = _REPO_ROOT.parent / "data" / "waveshare_work"


def split_contact_csv(dataset_dir: Path) -> dict[str, int]:
    merged_csv = dataset_dir / "contact_labels.csv"
    if not merged_csv.is_file():
        raise SystemExit(f"no consolidated contact_labels.csv at {merged_csv}")

    per_session: dict[str, dict[int, int]] = {}
    with merged_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            session = row["session"]
            frame_idx = int(row["frame_idx"])
            contact = int(row["contact"])
            per_session.setdefault(session, {})[frame_idx] = contact

    counts: dict[str, int] = {}
    for session, labels in per_session.items():
        session_dir = dataset_dir / session
        if not session_dir.is_dir():
            print(f"  WARNING: session dir missing for {session!r}, skipping", file=sys.stderr)
            continue

        existing_path = session_dir / "contact_labels.csv"
        if existing_path.is_file():
            existing = load_contact_labels(existing_path)
            if existing != labels:
                raise SystemExit(
                    f"existing per-session contact_labels.csv for {session!r} disagrees "
                    f"with the consolidated file — refusing to overwrite. Investigate manually."
                )
            counts[session] = len(existing)
            continue

        save_contact_labels(labels, existing_path)
        counts[session] = len(labels)

    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args()

    dataset_dir = Path(args.root)
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset dir not found: {dataset_dir}")

    print(f"Splitting consolidated contact_labels.csv in {dataset_dir} ...")
    counts = split_contact_csv(dataset_dir)

    print(f"\n{'session':<24}{'frames':>8}")
    print("-" * 32)
    for session, n in sorted(counts.items()):
        print(f"{session:<24}{n:>8}")
    print("-" * 32)
    print(f"{'TOTAL':<24}{sum(counts.values()):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
