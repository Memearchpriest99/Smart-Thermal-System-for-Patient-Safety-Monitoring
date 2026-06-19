"""Reconcile xlsx human-count one-hot (single/two/three) to the YOLO person count.

Scope: the four scenes the user asked to align to YOLO. For every (channel, frame)
row in these sheets, the human-count columns are overwritten from the YOLO
class-1 (person) box count for that exact view:

    persons 0 -> 0/0/0,  1 -> single,  2 -> two,  3 -> three

NOTE: this intentionally changes the human-count semantics for these scenes from
"people in the scene" to "person boxes visible in this camera view" (matches YOLO).

Untouched: the `touch` column (xlsx-only, no YOLO equivalent) and the `CORRUPT`
8th-column flags. `fire` is verified against YOLO and only rewritten if it somehow
disagrees (it shouldn't — global D1 is already 0).

The pristine pre-edit workbook is preserved at labels.pre_fire_fix.xlsx.
Re-run scripts/check_label_agreement.py afterwards to confirm.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from thermal_algorithms.training.label_io import FIRE_CLASS_ID, PERSON_CLASS_ID

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATASET = Path("dataset")
XLSX = Path("dataset_raw/labels.xlsx")
LOCK = XLSX.with_name("~$" + XLSX.name)
DEFAULT_SCENES = ["2pplfight", "2pplwithhairdryer", "2pplwithtouch", "3ppl"]


def _resolve_scenes(argv: list[str]) -> list[str]:
    """--all [--skip A B ...] -> every scene minus skips; else explicit names;
    else the default reviewed scenes. Examples:
        python scripts/reconcile_human_counts.py "crosswalk ppl"
        python scripts/reconcile_human_counts.py --all --skip 2pplrun 3pplfight
    """
    if "--all" in argv:
        skip = set(argv[argv.index("--skip") + 1:]) if "--skip" in argv else set()
        return sorted(
            d.name for d in DATASET.iterdir()
            if d.is_dir() and (d / "ch0_raw_data.npz").is_file() and d.name not in skip
        )
    return argv or DEFAULT_SCENES


SCENES = _resolve_scenes(sys.argv[1:])


def yolo_counts(scene: str, ch: int, frame: int) -> tuple[int, bool]:
    """Return (person_count, fire_present) from a YOLO label file."""
    txt = DATASET / scene / f"ch{ch}_frames" / f"frame_{frame:05d}.txt"
    if not txt.is_file():
        return 0, False
    persons = 0
    fire = False
    for line in txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls = int(line.split()[0])
        if cls == PERSON_CLASS_ID:
            persons += 1
        elif cls == FIRE_CLASS_ID:
            fire = True
    return persons, fire


def header_map(ws) -> dict:
    return {c.value: c.column for c in next(ws.iter_rows(min_row=1, max_row=1))}


def main() -> None:
    if LOCK.exists():
        raise SystemExit("ABORT: Excel has labels.xlsx open (~$ lock present). Close it first.")
    # SAFETY: snapshot the workbook before any edit (keeps only the latest).
    from snapshot_util import snapshot_xlsx
    snap = snapshot_xlsx(XLSX)
    print(f"Snapshot: {snap}")
    print(f"Reconciling scenes: {SCENES}")

    wb = openpyxl.load_workbook(XLSX)
    grand = 0
    for scene in SCENES:
        if scene not in wb.sheetnames:
            print(f"  WARN: sheet {scene!r} missing"); continue
        ws = wb[scene]
        h = header_map(ws)
        c_ch, c_fr = h["channel"], h["frame"]
        c_s, c_t, c_th = h["single human"], h["two humans"], h["three humans"]
        c_fire = h["fire"]
        changed_rows = 0
        for row in ws.iter_rows(min_row=2):
            if row[c_ch - 1].value is None or row[c_fr - 1].value is None:
                continue
            try:
                ch = int(row[c_ch - 1].value); fr = int(row[c_fr - 1].value)
            except (ValueError, TypeError):
                continue
            persons, fire = yolo_counts(scene, ch, fr)
            if persons > 3:
                print(f"  WARN: {scene} ch{ch} f{fr} YOLO persons={persons} > 3, "
                      f"capping to 'three'")
                persons = 3
            want = {c_s: int(persons == 1), c_t: int(persons == 2),
                    c_th: int(persons == 3), c_fire: int(fire)}
            row_changed = False
            for col, target in want.items():
                cell = row[col - 1]
                cur = int(cell.value) if cell.value not in (None, "") else 0
                if cur != target:
                    cell.value = target
                    row_changed = True
            if row_changed:
                changed_rows += 1
        print(f"{scene:20s}: {changed_rows} rows reconciled to YOLO")
        grand += changed_rows

    wb.save(XLSX)
    print(f"\nSaved {XLSX}. Rows changed: {grand}")


if __name__ == "__main__":
    main()
