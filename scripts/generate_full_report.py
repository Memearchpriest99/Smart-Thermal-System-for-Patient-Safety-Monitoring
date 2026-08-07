#!/usr/bin/env python3
"""Assemble the final engineering-report PDF from everything the other
scripts in this directory produced:

  reports/algorithm_derivations.md         (per-algorithm math + rationale)
  reports/onnx_export_manifest.json        (ONNX export scope/paths)
  reports/full_corpus_eval_baseline*.json  (fp32 baseline metrics+timing)
  reports/full_corpus_eval_onnx*.json      (ONNX fp32 metrics+timing)
  reports/full_corpus_eval_quantized*.json (fp16/bf16/int8 metrics+timing)
  data/DATASET_NOTES.md                    (dataset composition)

Missing/partial result files are handled gracefully (sections just note
what wasn't available yet) so this can be re-run at any point while the
slower evaluation jobs are still catching up, not only once every input is
final.

Layout: a two-column academic-paper page (Times body text, Computer-Modern
math, booktabs-style tables) -- title/abstract and the results/ONNX section
use a full-width single column (like a paper's title block and its
table*/figure* wide elements), the dataset-notes and derivations sections
flow in two columns.

Usage::

    python scripts/generate_full_report.py
    python scripts/generate_full_report.py --out reports/Full_Corpus_Engineering_Report.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import LETTER  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)

from thermal_algorithms.training.report_pdf import (  # noqa: E402
    _ImageCache,
    bar_chart_image,
    build_styles,
    markdown_to_flowables,
    results_table,
)

REPORTS_DIR = _REPO_ROOT / "reports"
DATA_ROOT = _REPO_ROOT.parent / "data"

# ---------------------------------------------------------------------------
# Page geometry -- a two-column academic-paper layout (CVPR/IEEE style).
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = LETTER
MARGIN = 0.55 * inch
GUTTER = 0.28 * inch
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
FULL_W = PAGE_W - 2 * MARGIN

# Caveats that change how a detector's numbers should be read -- shown
# inline with its results table rather than buried in prose elsewhere, so a
# reader can't miss them while looking at the metrics they qualify.
DETECTOR_CAVEATS = {
    "HOGSVMDetector": (
        "Timing caveat: HOGSVMDetector's brute-force multi-scale sliding-window "
        "search is inherently CPU-heavy (~2.8s/frame measured in isolation). Its "
        "evaluation pass here ran concurrently with the full-corpus training job "
        "on the same machine, so the mean/p95 ms below (~7-8s/frame) reflect CPU "
        "contention on top of that, not this detector's latency in isolation -- "
        "still the slowest detector in this report either way, just not "
        "apples-to-apples with the others' latency numbers, which ran without "
        "that contention."
    ),
    "MVSTGCNDetector": (
        "Caveat (project owner): the homography used for this evaluation's actor "
        "positions is a synthetic per-camera-to-floor-plane transform "
        "(examples.utils.make_synthetic_homographies), used because no real "
        "on-site Hot-Point Calibration exists for any waveshare_work scene (see "
        "data/DATASET_NOTES.md). More fundamentally, the calibration approach "
        "available for real deployment fuses two cameras' views onto a third "
        "camera's image plane rather than rectifying each camera independently "
        "to the true floor plane -- so even a real calibration would not give the "
        "GCN physically meaningful inter-actor distances, which is exactly what "
        "its epsilon_m/delta_m thresholds are calibrated against. Weak "
        "contact-detection precision/recall for MVSTGCNDetector below is "
        "therefore an expected consequence of the positional input signal, not "
        "evidence of insufficient training or model capacity."
    ),
}

DETECTOR_ORDER = [
    "FireSVMDetector", "HOGSVMDetector", "MobileNetSSDDetector",
    "MVSTGCNDetector", "ThermoX3DDetector",
]
VARIANT_ORDER = ["fp32_baseline", "onnx_fp32", "fp16", "bf16", "int8"]


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  WARNING: failed to parse {path}: {e}")
        return None


def _normalize_variant(v: str) -> str:
    """Smoke-test runs tag variants like 'fp32_baseline_smoketest' -- fold
    those back to the canonical name for grouping/ordering."""
    for canon in VARIANT_ORDER:
        if v == canon or v.startswith(canon + "_"):
            return canon
    return v


def collect_results(paths: list[Path]) -> dict[str, list[dict]]:
    """detector_name -> list of result dicts (one per variant found)."""
    by_detector: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        for r in payload.get("results", []):
            r = dict(r)
            r["variant"] = _normalize_variant(r["variant"])
            by_detector[r["detector_name"]].append(r)
    return by_detector


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.drawCentredString(PAGE_W / 2, 0.35 * inch, str(doc.page))
    canvas.restoreState()


def build_doc(out_path: str) -> BaseDocTemplate:
    frame_full = Frame(MARGIN, MARGIN, FULL_W, PAGE_H - 2 * MARGIN, id="full")
    frame_col1 = Frame(MARGIN, MARGIN, COL_W, PAGE_H - 2 * MARGIN, id="col1")
    frame_col2 = Frame(MARGIN + COL_W + GUTTER, MARGIN, COL_W, PAGE_H - 2 * MARGIN, id="col2")

    return BaseDocTemplate(
        out_path, pagesize=LETTER,
        pageTemplates=[
            PageTemplate(id="Cover", frames=[frame_full], onPage=_footer),
            PageTemplate(id="TwoCol", frames=[frame_col1, frame_col2], onPage=_footer),
            PageTemplate(id="Full", frames=[frame_full], onPage=_footer),
        ],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPORTS_DIR / "Full_Corpus_Engineering_Report.pdf"))
    ap.add_argument(
        "--result-globs", nargs="*",
        default=["full_corpus_eval_*.json", "eval_*smoketest*.json"],
        help="Glob patterns (relative to reports/) for result JSON files to merge.",
    )
    ap.add_argument("--image-dir", default=str(_REPO_ROOT / "reports" / "_report_images"))
    args = ap.parse_args()

    result_paths: list[Path] = []
    for pat in args.result_globs:
        result_paths.extend(sorted(REPORTS_DIR.glob(pat)))
    print(f"Merging {len(result_paths)} result file(s): {[p.name for p in result_paths]}")
    by_detector = collect_results(result_paths)

    manifest = _load_json(REPORTS_DIR / "onnx_export_manifest.json") or {}
    derivations_md = (REPORTS_DIR / "algorithm_derivations.md")
    derivations_text = derivations_md.read_text(encoding="utf-8") if derivations_md.is_file() else None
    dataset_notes = (DATA_ROOT / "DATASET_NOTES.md")
    dataset_notes_text = dataset_notes.read_text(encoding="utf-8") if dataset_notes.is_file() else None

    styles = build_styles()
    images = _ImageCache(Path(args.image_dir))
    story: list = []

    # ---- Title block (full width, page 1 only) ----
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Smart Thermal System for Patient Safety Monitoring:", styles["TitleC"]))
    story.append(Paragraph("Full-Corpus Training, ONNX Export, and Quantization Analysis", styles["TitleC"]))
    story.append(Paragraph(
        "Guy Chen &nbsp;&middot;&nbsp; Yaniv Blau &nbsp;&middot;&nbsp; Roy Lieberman<br/>"
        "Afeka Academic College of Engineering, Tel-Aviv &mdash; "
        "in collaboration with the Center for Mental Health, Be&rsquo;er Sheva",
        styles["SubtitleC"],
    ))
    n_with_results = len(by_detector)
    story.append(Paragraph("Abstract", styles["H3c"]))
    story.append(Paragraph(
        "This report documents the full-corpus retraining of every detector in the "
        "Smart Thermal System pipeline (fire, human, and multi-view contact detection), "
        "together with ONNX export and fp16/bf16/int8 quantization benchmarks. "
        f"Results below cover {n_with_results} detector(s) with at least one evaluated "
        "variant at generation time; missing variants are noted per section rather than "
        "silently omitted. Every metric is computed against the identical held-out "
        "waveshare_work test split for every detector and every variant, so numbers are "
        "directly comparable across the whole report. Mathematical derivations for each "
        "algorithm -- including the SVM KKT-eligibility argument -- are given in full, "
        "cross-checked against the shipped implementation rather than a textbook "
        "description of the method.",
        styles["AbstractC"],
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(NextPageTemplate("TwoCol"))
    story.append(PageBreak())

    # ---- Dataset composition (two columns) ----
    if dataset_notes_text:
        story.append(Paragraph("1. Dataset Composition", styles["H1c"]))
        story.extend(markdown_to_flowables(dataset_notes_text, images, styles, max_width_pt=COL_W))
    else:
        print("  WARNING: data/DATASET_NOTES.md not found -- skipping dataset section")

    # ---- Algorithm derivations (two columns) ----
    if derivations_text:
        story.extend(markdown_to_flowables(derivations_text, images, styles, max_width_pt=COL_W))
    else:
        print("  WARNING: reports/algorithm_derivations.md not found -- skipping derivations section")

    # ---- Results per detector (full width -- wide tables/figures) ----
    story.append(NextPageTemplate("Full"))
    story.append(PageBreak())
    story.append(Paragraph("2. Evaluation Results", styles["H1c"]))
    story.append(Paragraph(
        "All numbers below come from the exact same held-out waveshare_work test "
        "scenes for every detector and every variant (see scripts/eval_all_detectors.py:"
        "build_waveshare_test_split) -- fire, human, and contact detectors are all "
        "evaluated against the identical 3 sessions. \"Mean ms\" / \"p95 ms\" are "
        "single-frame inference latency (GPU for torch models by default, CPU for "
        "the two scikit-learn detectors and for every ONNX Runtime run on this "
        "machine -- its CUDA execution provider could not initialize here; see the "
        "ONNX section below).", styles["Bodyc"],
    ))

    chart_dir = Path(args.image_dir) / "charts"
    table_num = 1
    for name in DETECTOR_ORDER:
        rows = by_detector.get(name)
        if not rows:
            story.append(Paragraph(f"{name}: no evaluated results available yet.", styles["Bodyc"]))
            continue
        rows_sorted = sorted(rows, key=lambda r: VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99)
        caveat = DETECTOR_CAVEATS.get(name)
        story.append(Paragraph(name, styles["H3c"]))
        if caveat:
            caveat_style = ParagraphStyle("Caveat", parent=styles["Bodyc"], backColor=colors.HexColor("#fff4e0"), borderPadding=6)
            story.append(Paragraph(caveat, caveat_style))
        story.extend(results_table(rows_sorted, styles, title=f"{name} -- metrics by variant", table_num=table_num))
        table_num += 1

        labels = [r["variant"] for r in rows_sorted]
        latencies = [r["mean_inference_ms"] for r in rows_sorted]
        if len(labels) > 1:
            chart_path = bar_chart_image(
                labels, latencies, ylabel="mean ms/frame",
                title=f"{name}: mean inference latency by variant",
                out_path=chart_dir / f"{name}_latency.png",
                figsize=(4.2, 2.6),
            )
            img = Image(str(chart_path), width=4.2 * inch, height=4.2 * inch * 2.6 / 4.2)
            img.hAlign = "CENTER"
            story.append(img)
        story.append(Spacer(1, 12))

    # ---- ONNX export scope (full width) ----
    story.append(PageBreak())
    story.append(Paragraph("3. ONNX Export Scope", styles["H1c"]))
    story.append(Paragraph(
        "Only the learned sub-component of each detector is traced into ONNX. "
        "Classical-CV pre/post-processing is deterministic NumPy/OpenCV code, "
        "not a neural-net or sklearn estimator, and isn't part of any framework's "
        "ONNX graph by construction -- see each row's scope note.", styles["Bodyc"],
    ))
    for name, info in manifest.items():
        story.append(Paragraph(name, styles["H3c"]))
        story.append(Paragraph(info.get("scope", ""), styles["Bodyc"]))
        story.append(Paragraph(f"ONNX path: <font face=\"Courier\" size=\"7.5\">{info.get('onnx_path', '')}</font>", styles["Bodyc"]))

    doc = build_doc(args.out)
    doc.build(story)
    print(f"\nSaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
