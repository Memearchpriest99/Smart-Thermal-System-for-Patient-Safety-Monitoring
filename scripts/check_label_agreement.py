"""Cross-check YOLO .txt labels against labels.xlsx and report disagreements.

Report-only — never modifies any label. Reads:
  * dataset/<scene>/ch{N}_frames/frame_XXXXX.txt   (YOLO: class 0=fire, 1=person)
  * dataset/<scene>/meta.json                       (aligned frame counts)
  * dataset_raw/labels.xlsx                          (per-frame activity labels)

xlsx schema (one sheet per scene): channel, frame, single human, two humans,
three humans, fire, touch  [, "CORRUPT"]. Frames are 0-based and align with the
YOLO frame index and npz index.

Comparisons, per (scene, channel, frame):
  D1  fire mismatch        : (xlsx fire==1) != (YOLO has any class-0 box)
  D2  person-count mismatch: YOLO #class-1 boxes != (1*single + 2*two + 3*three)
  D3  one-hot violation    : more than one of single/two/three set in xlsx
  D4  CORRUPT frames       : listed with both sides' values (single-pixel spikes)
  D5  structural gaps      : xlsx row with no thermal frame (dropped in alignment),
                             or YOLO frame with no xlsx row
  touch                    : xlsx-only label — reported as counts (no YOLO equivalent)

Plus a duplicate-pair section contrasting the conflicting xlsx annotations of the
byte-identical recordings (2pplfight/2pplrun, 3ppl/3pplfight) for manual review.

Outputs: dataset/_label_agreement_report.md and dataset/_corrupt_frames.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import openpyxl

# Make the repo root importable when run as `python scripts/check_label_agreement.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from thermal_algorithms.training.label_io import FIRE_CLASS_ID, PERSON_CLASS_ID

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATASET = Path("dataset")
XLSX = Path("dataset_raw/labels.xlsx")
REPORT = DATASET / "_label_agreement_report.md"
CORRUPT_CSV = DATASET / "_corrupt_frames.csv"
CHANNELS = (0, 1, 2)
DUP_PAIRS = [("2pplfight", "2pplrun"), ("3ppl", "3pplfight")]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_yolo(txt_path: Path) -> tuple[int, bool]:
    """Return (person_count, fire_present) from a YOLO label file."""
    if not txt_path.is_file():
        return 0, False
    person = 0
    fire = False
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls = int(line.split()[0])
        if cls == PERSON_CLASS_ID:
            person += 1
        elif cls == FIRE_CLASS_ID:
            fire = True
    return person, fire


def read_xlsx_sheet(ws) -> dict[tuple[int, int], dict]:
    """Parse one worksheet → {(channel, frame): {...}}."""
    out: dict[tuple[int, int], dict] = {}
    rows = ws.iter_rows(values_only=True)
    header = next(rows, None)  # skip header
    for row in rows:
        if row is None or row[0] is None or row[1] is None:
            continue
        try:
            ch = int(row[0]); fr = int(row[1])
        except (ValueError, TypeError):
            continue
        def g(i):
            try:
                return int(row[i]) if row[i] not in (None, "") else 0
            except (ValueError, TypeError):
                return 0
        corrupt = any(
            isinstance(c, str) and c.strip().upper() == "CORRUPT" for c in row[2:]
        )
        out[(ch, fr)] = {
            "single": g(2), "two": g(3), "three": g(4),
            "fire": g(5), "touch": g(6), "corrupt": corrupt,
        }
    return out


def aligned_lengths(scene: str) -> dict[int, int]:
    meta = json.loads((DATASET / scene / "meta.json").read_text())
    return {int(k): int(v) for k, v in meta["n_frames_per_ch"].items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    scenes = sorted(d.name for d in DATASET.iterdir() if d.is_dir())

    d1: list[str] = []   # fire mismatch
    d2: list[str] = []   # person-count mismatch
    d3: list[str] = []   # one-hot violation
    d4: list[tuple] = [] # corrupt frames
    d5_dropped: list[str] = []   # xlsx row beyond aligned length
    d5_no_xlsx: list[str] = []   # YOLO frame with no xlsx row
    touch_total = 0
    compared = 0

    for scene in scenes:
        if scene not in wb.sheetnames:
            d5_no_xlsx.append(f"{scene}: no matching xlsx sheet")
            continue
        xlsx = read_xlsx_sheet(wb[scene])
        lengths = aligned_lengths(scene)

        # xlsx-driven comparisons
        for (ch, fr), x in sorted(xlsx.items()):
            if ch not in CHANNELS:
                continue
            if x["touch"] == 1:
                touch_total += 1
            if x["corrupt"]:
                d4.append((scene, ch, fr, x))
            # one-hot check (independent of YOLO)
            if (x["single"] + x["two"] + x["three"]) > 1:
                d3.append(f"{scene} ch{ch} f{fr}: single/two/three = "
                          f"{x['single']}/{x['two']}/{x['three']}")
            # beyond aligned length → dropped/orphan
            if fr >= lengths.get(ch, 0):
                d5_dropped.append(f"{scene} ch{ch} f{fr} (aligned len={lengths.get(ch,0)})")
                continue
            txt = DATASET / scene / f"ch{ch}_frames" / f"frame_{fr:05d}.txt"
            person_y, fire_y = read_yolo(txt)
            human_x = 1 * x["single"] + 2 * x["two"] + 3 * x["three"]
            fire_x = bool(x["fire"])
            compared += 1
            if fire_x != fire_y:
                d1.append(f"{scene} ch{ch} f{fr}: xlsx_fire={int(fire_x)} "
                          f"yolo_fire={int(fire_y)}"
                          f"{' [CORRUPT]' if x['corrupt'] else ''}")
            if person_y != human_x:
                d2.append(f"{scene} ch{ch} f{fr}: yolo_persons={person_y} "
                          f"xlsx_humans={human_x} "
                          f"(s/t/th={x['single']}/{x['two']}/{x['three']})"
                          f"{' [CORRUPT]' if x['corrupt'] else ''}")

        # YOLO-driven structural: frames with no xlsx row
        for ch in CHANNELS:
            for fr in range(lengths.get(ch, 0)):
                if (ch, fr) not in xlsx:
                    d5_no_xlsx.append(f"{scene} ch{ch} f{fr}: YOLO frame, no xlsx row")

    # ---- Duplicate-pair contrast ----
    def sheet_summary(name: str) -> dict:
        if name not in wb.sheetnames:
            return {}
        x = read_xlsx_sheet(wb[name])
        return {
            "rows": len(x),
            "touch": sum(v["touch"] for v in x.values()),
            "fire": sum(v["fire"] for v in x.values()),
            "single": sum(v["single"] for v in x.values()),
            "two": sum(v["two"] for v in x.values()),
            "three": sum(v["three"] for v in x.values()),
            "corrupt": sum(1 for v in x.values() if v["corrupt"]),
        }

    # ---- Write corrupt CSV ----
    with CORRUPT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "channel", "frame", "single", "two", "three", "fire", "touch"])
        for scene, ch, fr, x in d4:
            w.writerow([scene, ch, fr, x["single"], x["two"], x["three"], x["fire"], x["touch"]])

    # ---- Write report ----
    L = []
    L.append("# Label Agreement Report — YOLO `.txt` vs `labels.xlsx`\n")
    L.append(f"Frames compared: **{compared}**  |  "
             f"D1 fire={len(d1)}  D2 person-count={len(d2)}  "
             f"D3 one-hot={len(d3)}  D4 corrupt={len(d4)}  "
             f"D5 dropped={len(d5_dropped)} / no-xlsx={len(d5_no_xlsx)}  |  "
             f"touch frames (xlsx-only)={touch_total}\n")

    def section(title, items, limit=None):
        L.append(f"\n## {title} ({len(items)})\n")
        if not items:
            L.append("_None._\n"); return
        shown = items if limit is None else items[:limit]
        for it in shown:
            L.append(f"- {it}")
        if limit and len(items) > limit:
            L.append(f"- … and {len(items) - limit} more")
        L.append("")

    section("D1 — Fire disagreements", d1)
    section("D2 — Person-count disagreements", d2)
    section("D3 — xlsx one-hot violations", d3)
    L.append(f"\n## D4 — CORRUPT frames ({len(d4)})\n")
    L.append("Single-pixel noise spikes (kept intentionally). See `_corrupt_frames.csv`.\n")
    for scene, ch, fr, x in d4:
        L.append(f"- {scene} ch{ch} f{fr}  "
                 f"(s/t/th={x['single']}/{x['two']}/{x['three']} fire={x['fire']} touch={x['touch']})")
    section("D5 — xlsx rows beyond aligned thermal length (dropped/orphan)", d5_dropped)
    section("D5 — YOLO frames with no xlsx row", d5_no_xlsx, limit=100)

    L.append("\n## Duplicate-pair annotation conflict (MANUAL REVIEW)\n")
    L.append("Thermal npz + YOLO labels are byte-identical within each pair, but the "
             "xlsx annotations differ — the same footage was annotated twice. Decide which "
             "annotation is authoritative.\n")
    for a, b in DUP_PAIRS:
        sa, sb = sheet_summary(a), sheet_summary(b)
        L.append(f"\n### `{a}`  vs  `{b}`\n")
        L.append(f"| metric | {a} | {b} |")
        L.append("|---|---|---|")
        for k in ("rows", "touch", "fire", "single", "two", "three", "corrupt"):
            L.append(f"| {k} | {sa.get(k,'-')} | {sb.get(k,'-')} |")
        L.append("")

    REPORT.write_text("\n".join(L), encoding="utf-8")

    # ---- Console summary ----
    print(f"Compared {compared} frames across {len(scenes)} scenes.")
    print(f"  D1 fire mismatch     : {len(d1)}")
    print(f"  D2 person-count diff : {len(d2)}")
    print(f"  D3 one-hot violation : {len(d3)}")
    print(f"  D4 CORRUPT frames    : {len(d4)}")
    print(f"  D5 dropped rows      : {len(d5_dropped)}")
    print(f"  D5 YOLO w/o xlsx row : {len(d5_no_xlsx)}")
    print(f"  touch frames (xlsx)  : {touch_total}")
    print(f"\nReport : {REPORT}")
    print(f"Corrupt: {CORRUPT_CSV}")


if __name__ == "__main__":
    main()
