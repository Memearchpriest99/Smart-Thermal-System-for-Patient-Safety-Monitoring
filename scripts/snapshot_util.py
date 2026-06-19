"""Shared snapshot helper for labels.xlsx edits.

Retention policy (user preference): keep ONLY the latest snapshot. Each call
creates a new timestamped snapshot and then deletes every other backup
(prior `labels.bak.*.xlsx` and the pristine `labels.pre_fire_fix.xlsx`), so
exactly one backup file ever exists — the most recent one.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path


def snapshot_xlsx(xlsx: str | Path) -> Path:
    """Snapshot `xlsx`, keeping only this latest snapshot. Returns the new path."""
    xlsx = Path(xlsx)
    d = xlsx.parent
    # Create the newest snapshot FIRST (so a backup always exists if a later
    # deletion fails), then remove every other backup.
    new = d / f"labels.bak.{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    shutil.copy2(xlsx, new)
    for b in list(d.glob("labels.pre_fire_fix.xlsx")) + list(d.glob("labels.bak.*.xlsx")):
        if b.resolve() != new.resolve():
            b.unlink()
    return new
