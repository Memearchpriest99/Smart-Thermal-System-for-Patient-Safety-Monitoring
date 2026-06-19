#!/usr/bin/env python3
"""Consolidate & clean the ``waveshare_work`` dataset.

One-shot maintenance script (see plan
``i-ve-updated-the-waveshare-work-quizzical-quill``). It:

  1. Merges every per-session ``contact_labels.csv`` into a single
     ``datasets/waveshare_work/contact_labels.csv`` with a leading ``session``
     column.
  2. Merges every per-session ``labels.xlsx`` into a single
     ``datasets/waveshare_work/labels.xlsx`` (one sheet per session), preserving
     each sheet's tab colour and per-cell fills (the manual "green = reviewed"
     review markers).
  3. After the merged files are built *and verified*, backs up the per-session
     label files to a zip, then deletes them along with all ``ch*_video.mp4``
     videos so only frames + labels remain.

Build-and-verify happens before any deletion: if verification fails, nothing is
removed.
"""

from __future__ import annotations

import copy
import csv
import sys
import zipfile
from pathlib import Path

import openpyxl

# Make ``thermal_algorithms`` importable when run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thermal_algorithms.training.label_io import load_contact_labels  # noqa: E402

DATASET_DIR = _REPO_ROOT / "datasets" / "waveshare_work"
MERGED_CSV = DATASET_DIR / "contact_labels.csv"
MERGED_XLSX = DATASET_DIR / "labels.xlsx"


def discover_sessions(dataset_dir: Path) -> list[Path]:
    """Return sorted session dirs (immediate subdirs holding contact_labels.csv)."""
    sessions = [
        d
        for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "contact_labels.csv").is_file()
    ]
    return sorted(sessions, key=lambda p: p.name)


def merge_contact_csv(sessions: list[Path], out_path: Path) -> dict[str, int]:
    """Merge per-session contact CSVs into one. Returns {session: frame_count}."""
    counts: dict[str, int] = {}
    rows: list[tuple[str, int, int]] = []
    for session in sessions:
        labels = load_contact_labels(session / "contact_labels.csv")
        counts[session.name] = len(labels)
        for frame_idx in sorted(labels):
            rows.append((session.name, frame_idx, labels[frame_idx]))

    rows.sort(key=lambda r: (r[0], r[1]))
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["session", "frame_idx", "contact"])
        writer.writerows(rows)
    return counts


def merge_xlsx(sessions: list[Path], out_path: Path) -> dict[str, dict]:
    """Merge per-session workbooks into one (sheet per session).

    Preserves header + data rows verbatim, the sheet tab colour, and per-cell
    fills. Returns {session: {"rows": n, "tab": tabColor}}.
    """
    merged = openpyxl.Workbook()
    merged.remove(merged.active)  # drop the default empty sheet

    info: dict[str, dict] = {}
    for session in sessions:
        src_wb = openpyxl.load_workbook(session / "labels.xlsx")
        # Each per-session workbook has exactly one sheet.
        src_ws = src_wb[src_wb.sheetnames[0]]

        # Sheet names are guaranteed unique and <=31 chars (session names).
        dst_ws = merged.create_sheet(title=session.name)
        for row in src_ws.iter_rows():
            for cell in row:
                dst_cell = dst_ws.cell(
                    row=cell.row, column=cell.column, value=cell.value
                )
                # Preserve manual review formatting (e.g. green fills).
                if cell.has_style:
                    dst_cell.fill = copy.copy(cell.fill)
                    dst_cell.font = copy.copy(cell.font)

        # Preserve the tab colour (green = reviewed).
        dst_ws.sheet_properties.tabColor = src_ws.sheet_properties.tabColor

        info[session.name] = {
            "rows": src_ws.max_row,
            "tab": src_ws.sheet_properties.tabColor,
        }
        src_wb.close()

    merged.save(out_path)
    return info


def verify(
    sessions: list[Path],
    counts: dict[str, int],
    xlsx_info: dict[str, dict],
    merged_csv: Path,
    merged_xlsx: Path,
) -> None:
    """Raise AssertionError if merged outputs don't match the sources."""
    # CSV: total data rows == sum of per-session frame counts.
    with merged_csv.open(newline="", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    expected_csv = sum(counts.values())
    assert len(csv_rows) == expected_csv, (
        f"merged CSV has {len(csv_rows)} rows, expected {expected_csv}"
    )
    assert len({r["session"] for r in csv_rows}) == len(sessions), (
        "merged CSV session count mismatch"
    )

    # XLSX: one sheet per session, each sheet row count matches its source.
    wb = openpyxl.load_workbook(merged_xlsx)
    assert len(wb.sheetnames) == len(sessions), (
        f"merged xlsx has {len(wb.sheetnames)} sheets, expected {len(sessions)}"
    )
    for session in sessions:
        assert session.name in wb.sheetnames, f"missing sheet {session.name!r}"
        got = wb[session.name].max_row
        want = xlsx_info[session.name]["rows"]
        assert got == want, (
            f"sheet {session.name!r}: {got} rows, expected {want}"
        )
    wb.close()


def backup_label_files(sessions: list[Path], dataset_dir: Path) -> Path:
    """Zip every per-session contact_labels.csv + labels.xlsx before deletion."""
    backup_path = dataset_dir.parent / "_label_backup_waveshare_work.zip"
    with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for session in sessions:
            for name in ("contact_labels.csv", "labels.xlsx"):
                src = session / name
                if src.is_file():
                    zf.write(src, arcname=f"{session.name}/{name}")
    return backup_path


def delete_originals(sessions: list[Path]) -> tuple[int, int]:
    """Delete per-session label files and all mp4 videos. Returns (labels, mp4)."""
    label_deleted = 0
    mp4_deleted = 0
    for session in sessions:
        for name in ("contact_labels.csv", "labels.xlsx"):
            target = session / name
            if target.is_file():
                target.unlink()
                label_deleted += 1
        for mp4 in session.glob("*.mp4"):
            mp4.unlink()
            mp4_deleted += 1
    return label_deleted, mp4_deleted


def main() -> int:
    if not DATASET_DIR.is_dir():
        print(f"ERROR: dataset dir not found: {DATASET_DIR}", file=sys.stderr)
        return 1

    sessions = discover_sessions(DATASET_DIR)
    print(f"Discovered {len(sessions)} sessions in {DATASET_DIR}\n")

    print("Merging contact labels ...")
    counts = merge_contact_csv(sessions, MERGED_CSV)
    print(f"  -> {MERGED_CSV}")

    print("Merging label workbooks ...")
    xlsx_info = merge_xlsx(sessions, MERGED_XLSX)
    print(f"  -> {MERGED_XLSX}\n")

    print("Verifying merged outputs ...")
    verify(sessions, counts, xlsx_info, MERGED_CSV, MERGED_XLSX)
    print("  OK\n")

    # Per-session summary table.
    print(f"{'session':<24}{'frames':>8}{'xlsx_rows':>11}  tab_color")
    print("-" * 60)
    for session in sessions:
        tab = xlsx_info[session.name]["tab"] or "-"
        print(
            f"{session.name:<24}{counts[session.name]:>8}"
            f"{xlsx_info[session.name]['rows']:>11}  {tab}"
        )
    print("-" * 60)
    print(
        f"{'TOTAL':<24}{sum(counts.values()):>8}"
        f"{sum(i['rows'] for i in xlsx_info.values()):>11}\n"
    )

    print("Backing up per-session label files ...")
    backup = backup_label_files(sessions, DATASET_DIR)
    print(f"  -> {backup}\n")

    print("Deleting originals (per-session labels + mp4 videos) ...")
    label_deleted, mp4_deleted = delete_originals(sessions)
    print(f"  deleted {label_deleted} per-session label files")
    print(f"  deleted {mp4_deleted} mp4 videos\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
