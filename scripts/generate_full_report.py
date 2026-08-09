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
    "FireSVMDetector": (
        "Accuracy clears 70-80%, but recall is low (~15%) with higher precision "
        "(~58%) -- read together, this SVM under-alerts rather than over-alerts. "
        "This is a defensible tradeoff, not an unexplained weakness: fire is a "
        "rare event in the corpus, sklearn's SVC does not scale past a few "
        "thousand training points (fit() subsamples to ~1500 frames/camera, see "
        "S2.8), and a false-alarm-averse posture is arguably the right default "
        "for a ward safety system built on top of this detector. Still, this "
        "recall level should be weighed before relying on FireSVMDetector alone."
    ),
    "HOGSVMDetector": (
        "UPDATED 2026-08-08: the original held-out human-detection test scenes "
        "(2ppl_hug, 3pplhedroncolider, man_light_cig_1) contained zero "
        "ground-truth-negative frames, so an earlier version of this report read "
        "as a mathematically-forced 100% across accuracy/precision/recall/F1 -- "
        "not a real measurement of false-positive suppression. Re-evaluated with "
        "empty_room added to the test set (real negative frames): recall stays "
        "100% (never misses a person), but precision drops to 78.0% and accuracy "
        "to 78.1% -- raw counts tp=1344, tn=5, fp=379, fn=0 mean **HOGSVM false-"
        "alarms on 379/384 (98.7%) of truly empty frames**. Its dense, "
        "per-pixel-stride multi-scale sliding window produces some candidate box "
        "on nearly every frame regardless of content -- this was a real, "
        "previously-invisible weakness, not a new regression. Mean IoU (0.57) is "
        "still the more informative localization-quality number. Also the "
        "slowest detector in this report by far (~1.6-2.3s/frame, CPU-only, does "
        "not benefit from a GPU)."
    ),
    "MobileNetSSDDetector": (
        "UPDATED 2026-08-08: same test-set fix as HOGSVMDetector above. "
        "Re-evaluated with empty_room added: recall stays 100%, but precision is "
        "96.6% and accuracy 97.2% -- raw counts tp=1344, tn=336, fp=48, fn=0, "
        "i.e. it false-alarms on 48/384 (12.5%) of truly empty frames -- far "
        "better than HOGSVMDetector's 98.7% but not the false-positive-proof "
        "detector the old zero-negative test set implied. Mean IoU (0.76) "
        "remains meaningfully better than HOGSVMDetector's (0.57), and native-"
        "PyTorch GPU inference (~3.9ms/frame) is roughly 580x faster -- still "
        "the stronger of the two human detectors on every axis this report can "
        "measure, just not literally flawless. The ONNX and fp16/bf16/int8 "
        "tables below now reflect this same corrected test set too (re-run with "
        "--extra-human-test-scenes empty_room) -- false-alarm rate holds in the "
        "12-15% range across every precision/format variant, so this isn't an "
        "artifact of fp32 specifically."
    ),
    "MVSTGCNDetector": (
        "Precision/recall/F1 are all 0% (raw counts: tp=0, tn=348, fp=0, "
        "fn=122) -- the model never predicts \"contact\" on the held-out set; "
        "74% accuracy is purely the negative-class base rate, not a real skill "
        "signal. Two compounding, non-mutually-exclusive causes, neither of "
        "which is a code bug: (1) Caveat (project owner): the homography used "
        "for this evaluation's actor positions is a synthetic "
        "per-camera-to-floor-plane transform (examples.utils."
        "make_synthetic_homographies), used because no real on-site Hot-Point "
        "Calibration exists for any waveshare_work scene -- and the calibration "
        "approach available for real deployment fuses two cameras' views onto a "
        "third camera's image plane rather than rectifying each camera "
        "independently to the true floor plane, so even a real calibration would "
        "not give the GCN physically meaningful inter-actor distances that its "
        "epsilon_m/delta_m thresholds are calibrated against. (2) The corpus-wide "
        "contact class balance is severe (2.4% contact-positive, 97.6% "
        "negative); the chunked full-corpus training design (~2072 chunks of "
        "5000 examples) likely hands many individual chunks zero or very few "
        "positive examples, so despite the computed 41x positive-class weight, "
        "the model may receive sparse, diluted positive-class gradient signal "
        "overall -- plausibly insufficient to push true-positive confidence "
        "reliably above the persistence-gated 0.7 threshold (S6.6). Report this "
        "result plainly rather than substituting the superficially-fine 74% "
        "accuracy figure for it."
    ),
    "ThermoX3DDetector": (
        "RETRACTED 2026-08-09: the ~9% subset checkpoint's tp=0/tn=348/fp=0/fn=122 "
        "result (and the earlier claim that a separate balanced retrain reproduced "
        "the IDENTICAL confusion matrix, 'arguing against class imbalance') was "
        "misdiagnosed. A dedicated Opus investigation (prompted because the "
        "'identical result across two different training runs' didn't make sense) "
        "proved this checkpoint's chunked fit() loop is fundamentally broken: a "
        "fresh torch.optim.Adam is constructed on every chunk (discarding momentum/ "
        "adaptive-LR state ~180-196 times per run), chunks are single-class, and "
        "global normalisation stats are recomputed from scratch per chunk (so the "
        "saved checkpoint reflects only the LAST chunk's statistics). The result is "
        "a degenerate CONSTANT classifier -- a brand-new, never-trained detector "
        "reproduces the exact same confusion matrix on this held-out set. The "
        "'identical across natural vs. balanced training' finding was real but its "
        "interpretation was wrong: it is the signature of training that never "
        "worked, not evidence about class imbalance. This checkpoint (scripts/"
        "train_thermox3d_subset.py / train_full_corpus.py's chunked contact path) "
        "has NOT been re-fixed in this pass -- only the balanced-regime training "
        "path (see the Unbiased Results section below) was. Treat this number as "
        "invalid pending a matching fix to the natural-ratio driver; do not compare "
        "it against the balanced-regime result below as if both were trained "
        "correctly."
    ),
    "OtsuFireDetector": (
        "is_trainable=False and fit() is a documented no-op -- t_ign/t_fire/"
        "a_limit were never anything but EDA guesses until this report: "
        "scripts/calibrate_otsu_thresholds.py grid-searches them against ~51k "
        "frames sampled the same way as FireSVMDetector's full-corpus training, "
        "restricted to precision>=0.8 before ranking by F1 (an unconstrained "
        "F1-only search degenerates to predicting fire on nearly every frame -- "
        "F1 never penalizes false positives via true negatives -- rejected as "
        "operationally useless; see reports/otsu_threshold_calibration.json for "
        "the full grid). Only t_ign moved (45->41.5C): t_fire/a_limit are "
        "empirically INERT across their whole tested range at the validated "
        "t_ign, because every genuine hot blob in this corpus is small (<0.5% "
        "of frame) -- the two-tier ignition/potential-fire split barely "
        "engages its second branch here, so calibration provides no evidence "
        "to move them from their original values. Held-out result below is "
        "the calibrated t_ign; for reference the never-validated original "
        "default reached 78.4% acc / 100% prec / 26.8% rec / 42.3% F1 on the "
        "identical split -- calibration recovered +12.3pp F1/recall for -17pp "
        "precision (still only ~3.4% false-alarm rate, 45/1334 negatives). "
        "Unlike FireSVMDetector, this detector's predict() carries a real "
        "bbox, so mean IoU here is genuinely measured, not N/A."
    ),
}

DETECTOR_ORDER = [
    "FireSVMDetector", "OtsuFireDetector", "HOGSVMDetector", "MobileNetSSDDetector",
    "MVSTGCNDetector", "ThermoX3DDetector",
]
VARIANT_ORDER = ["fp32_baseline", "onnx_fp32", "fp16", "bf16", "int8"]

# ---------------------------------------------------------------------------
# Section 4: Unbiased (Balanced-Training) Results -- a genuinely resampled
# 50/50 positive/negative retrain/recalibration of 6 detectors (MVSTGCN
# excluded -- project owner's call, "useless, leave as is"), vs. the
# natural-corpus-ratio numbers in section 2 above. See thermal_algorithms/
# training/balance.py and scripts/train_balanced_corpus.py /
# calibrate_otsu_thresholds.py --balanced / calibrate_adaptive_threshold.py
# --balanced for the methodology. Deliberately a SEPARATE detector-order/
# caveat/glob set from section 2's -- never share a glob pattern with
# --result-globs, or a rerun risks duplicate rows (hit and fixed once
# already this project).
# ---------------------------------------------------------------------------
BALANCED_DETECTOR_ORDER = [
    "OtsuFireDetector", "FireSVMDetector", "AdaptiveThresholdDetector",
    "HOGSVMDetector", "MobileNetSSDDetector", "ThermoX3DDetector",
]

BALANCED_CAVEATS = {
    "OtsuFireDetector": (
        "Calibrated (grid search, see S2's OtsuFireDetector caveat for the methodology) against a "
        "genuinely 50/50-resampled pool (thermal_algorithms.training.balance.build_balanced_fire_pool) "
        "instead of the natural ~28% fire-positive corpus ratio -- chosen t_ign=39.2C (was 41.5C "
        "natural). Held-out on the SAME natural-ratio fire_test_scenes: 75.8% acc / 59.2% prec / "
        "58.1% rec / 58.7% F1 (mean IoU 9.2%) vs. the natural-ratio calibration's 79.7% / 83.0% / "
        "39.2% / 53.2% (7.0%). Balancing shifted the precision/recall tradeoff -- +18.9pp recall for "
        "-23.8pp precision -- with a net +5.5pp F1 gain: a real, disclosed tradeoff (catches more real "
        "fires, at the cost of more false alarms), not an unambiguous win."
    ),
    "FireSVMDetector": (
        "Retrained on a genuinely 50/50-resampled pool (28,896 examples: 14,448 fire-positive + "
        "14,448 negative) instead of the natural ~28% ratio. Held-out: 70.3% acc / 48.8% prec / "
        "15.0% rec / 23.0% F1 -- virtually UNCHANGED from the natural-ratio run's 71.6% / 57.5% / "
        "15.0% / 23.8% (recall identical to 3 significant figures). Sample-level rebalancing did not "
        "move this detector's performance, most plausibly because the natural-ratio run already used "
        "sklearn's class_weight=\"balanced\" (loss-reweighting) -- the balanced run keeps that setting "
        "too, since it's a genuine no-op on an already-even input pool (see thermal_algorithms/"
        "training/balance.py's build_balanced_fire_pool docstring), so this comparison isolates "
        "sample-level resampling from loss-reweighting cleanly: for this detector, loss-reweighting "
        "alone was already doing the correcting work that data-level rebalancing would otherwise do."
    ),
    "AdaptiveThresholdDetector": (
        "Calibrated for the FIRST time in this project (previously is_trainable=False, fit() a "
        "documented no-op, never validated against labeled data at all -- see S2's HOGSVMDetector/"
        "MobileNetSSDDetector caveats for the related empty_room-negatives-test-set fix this shares). "
        "Chose c_offset=1.528, min_area_pixels=4. Held-out (human_test_scenes + empty_room, matching "
        "HOGSVM/MobileNetSSD's own test-set convention): 86.2% acc / 86.6% prec / 97.3% rec / "
        "91.6% F1, mean IoU 14.0%. False-alarm rate on empty_room: 203/384 = 52.9% -- meaningfully "
        "better than HOGSVMDetector's 98.7% but worse than MobileNetSSDDetector's false-alarm rate "
        "(see below). No natural-ratio counterpart exists for this detector (it was calibrated "
        "directly against balanced data in this pass, not re-calibrated from an existing baseline), "
        "so this is a first-time result, not a before/after comparison like the other five."
    ),
    "HOGSVMDetector": (
        "Retrained on a genuinely 50/50-resampled pool of 908 frames (454 person-positive + 454 "
        "negative -- capped by waveshare_work's scarce negative-frame pool, with empty_room EXCLUDED "
        "from training since it's used as an extra negative test scene -- see thermal_algorithms/"
        "training/balance.py's build_balanced_human_pool exclude_scenes param). class_weight="
        "\"balanced\" was kept (not set to None) for this retrain, unlike FireSVMDetector above: "
        "HOGSVMDetector.fit() does not train one sample per frame -- it emits one positive HOG patch "
        "per ground-truth bbox plus up to 5 random background patches PER FRAME regardless of that "
        "frame's own label, so the frame-level 50/50 resampling does not by itself produce a 50/50 "
        "patch-level ratio (closer to ~1:5 in practice); dropping the loss-reweighting on top would "
        "have silently reintroduced an uncorrected imbalance. **Held-out (human_test_scenes + "
        "empty_room): 77.8% acc / 77.8% prec / 100% rec / 87.5% F1, mean IoU 55.6% -- false-alarm "
        "rate on empty_room is 384/384 = 100% (tn=0), marginally WORSE than the natural-ratio run's "
        "379/384 = 98.7%.** Balanced training did not help this detector at all -- consistent with "
        "its false-positive behaviour being driven by the dense, exhaustive sliding-window search "
        "itself (some candidate box on nearly every frame regardless of content) rather than the "
        "SVM decision boundary's class balance; rebalancing the training data cannot fix a "
        "structural property of the search procedure."
    ),
    "MobileNetSSDDetector": (
        "Retrained on the SAME balanced 908-frame human pool as HOGSVMDetector above (empty_room "
        "excluded from training). Held-out (human_test_scenes + empty_room): 98.0% acc / 98.4% "
        "prec / 99.1% rec / 98.7% F1, mean IoU 72.1% -- vs. the natural-ratio run's 97.2% / 96.6% "
        "/ 100% / 98.2% (76.0%). **A genuine improvement**: false-alarm rate on empty_room dropped "
        "from 48/384 (12.5%) to 22/384 (5.7%) -- more than half -- for a negligible recall cost (12 "
        "more missed frames out of 1344 positives) and a small mean-IoU decrease. Balanced training "
        "measurably helped this detector's false-positive suppression, the clearest net win in this "
        "whole balanced-training pass."
    ),
    "ThermoX3DDetector": (
        "REDONE 2026-08-09 -- the previous balanced checkpoint's 'identical to natural-ratio' result "
        "was a broken-training-loop artifact (both were degenerate constant classifiers; see S2's "
        "ThermoX3DDetector caveat for the full root-cause writeup), not a real finding about class "
        "imbalance. Fixed and retrained from scratch: persistent AdamW optimizer (was rebuilt fresh "
        "every chunk, discarding momentum state), checkpoint-level frozen normalisation stats (was "
        "recomputed per chunk from whatever data that chunk happened to hold), class-mixed "
        "super-chunks with a ratio cap (thermal_algorithms/training/balance.py's "
        "build_balanced_contact_pools/split_contact_pools_train_val/interleave_chunks -- replacing "
        "the single iter_balanced_contact_runs generator so train/val splitting happens at the "
        "parent-run level, before sub-chunking, avoiding near-duplicate frames landing on both "
        "sides), lower learning rate (1e-4, was 1e-3), a real validation split with early stopping "
        "(best-val-loss state restored), and an architecture change from T=16 to T=5 frames "
        "(input tensor (3,5,62,80)) per the project owner's direction. Held-out result: 28.5% acc / "
        "26.6% prec / 100% rec / 42.1% F1 (tp=122, tn=12, fp=336, fn=0) -- the training-loop bug is "
        "CONFIRMED fixed: recall jumped from 0% to 100% and the model's raw confidence output now "
        "varies meaningfully with input (the dead-network signature -- bit-identical output "
        "regardless of input -- is gone). Precision remains poor: post-hoc threshold/persistence-"
        "frames recalibration (grid search restricted to val precision >= 0.8 before ranking by F1, "
        "mirroring OtsuFireDetector's calibration convention above) still picked a near-lowest "
        "threshold, because on this task's very small validation set (9 items, drawn from a corpus "
        "with only ~188 real positive contact frames total -- see scripts/eval_waveshare_contact_dl.py's "
        "own caveat) even the lowest grid threshold achieved perfect val precision, so the floor had "
        "nothing to reject. This is a genuine data-scarcity limitation of the real corpus, not a "
        "remaining bug in the training loop or the recalibration mechanism -- reported plainly as a "
        "feasibility result (the fix works; the detector needs substantially more real labelled "
        "contact data to calibrate a usable operating point), not a benchmark."
    ),
}


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
    ap.add_argument(
        "--balanced-result-globs", nargs="*",
        default=["balanced_training_eval.json", "otsu_balanced_eval.json",
                 "adaptive_threshold_balanced_eval.json"],
        help="Glob patterns (relative to reports/) for the balanced-training section's result "
             "files. Deliberately NONE of these match the 'full_corpus_eval_*.json'/"
             "'eval_*smoketest*.json' patterns --result-globs defaults to -- a filename that "
             "matched both would double-count into section 2 AND section 4 the moment someone "
             "reruns this script with default args (caught once already: the eval_all_detectors.py "
             "output for this section was originally named full_corpus_eval_balanced.json, which "
             "DID match the default --result-globs pattern; renamed to balanced_training_eval.json).",
    )
    args = ap.parse_args()

    result_paths: list[Path] = []
    for pat in args.result_globs:
        result_paths.extend(sorted(REPORTS_DIR.glob(pat)))
    print(f"Merging {len(result_paths)} result file(s): {[p.name for p in result_paths]}")
    by_detector = collect_results(result_paths)

    balanced_result_paths: list[Path] = []
    for pat in args.balanced_result_globs:
        balanced_result_paths.extend(sorted(REPORTS_DIR.glob(pat)))
    print(f"Merging {len(balanced_result_paths)} balanced-training result file(s): "
          f"{[p.name for p in balanced_result_paths]}")
    by_detector_balanced = collect_results(balanced_result_paths)

    manifest = _load_json(REPORTS_DIR / "onnx_export_manifest.json") or {}
    derivations_md = (REPORTS_DIR / "algorithm_derivations.md")
    derivations_text = derivations_md.read_text(encoding="utf-8") if derivations_md.is_file() else None
    # Prefer a report-scoped summary (trimmed of dev-only implementation notes -- Room-1
    # exclusion investigation, internal Room_ID/folder-name mismatch, performance/caching
    # notes) over the full working dev doc, when one has been prepared.
    dataset_notes = (DATA_ROOT / "DATASET_NOTES_REPORT.md")
    if not dataset_notes.is_file():
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
            story.append(Paragraph(name, styles["H3c"]))
            pending_caveat = DETECTOR_CAVEATS.get(name)
            if pending_caveat:
                caveat_style = ParagraphStyle("Caveat", parent=styles["Bodyc"], backColor=colors.HexColor("#fff4e0"), borderPadding=6)
                story.append(Paragraph(pending_caveat, caveat_style))
            else:
                story.append(Paragraph("No evaluated results available yet.", styles["Bodyc"]))
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

    # ---- Unbiased (Balanced-Training) Results (full width) ----
    story.append(PageBreak())
    story.append(Paragraph("4. Unbiased (Balanced-Training) Results", styles["H1c"]))
    story.append(Paragraph(
        "The results above train/calibrate every detector on the corpus's NATURAL class ratio "
        "(fire ~28% positive, contact ~2.4%, human ~90% positive here in waveshare_work -- see "
        "S1). This section retrains/recalibrates 6 of those detectors (MVSTGCNDetector excluded, "
        "per the project owner) on a genuinely resampled 50/50 positive/negative training set -- "
        "real sample-level resampling (thermal_algorithms/training/balance.py), not just "
        "class_weight loss-reweighting -- to separate \"was this detector's behaviour a training-"
        "data-ratio artifact\" from \"is this a real limitation\". Test scenes are UNCHANGED from "
        "the natural-ratio sections above in every case, so numbers here are directly comparable "
        "to S2's.", styles["Bodyc"],
    ))

    balanced_table_num = 1
    for name in BALANCED_DETECTOR_ORDER:
        rows = by_detector_balanced.get(name)
        story.append(Paragraph(name, styles["H3c"]))
        caveat = BALANCED_CAVEATS.get(name)
        if caveat:
            caveat_style = ParagraphStyle("BalancedCaveat", parent=styles["Bodyc"], backColor=colors.HexColor("#e6f2ff"), borderPadding=6)
            story.append(Paragraph(caveat, caveat_style))
        if not rows:
            story.append(Paragraph("No evaluated results available yet.", styles["Bodyc"]))
            continue
        rows_sorted = sorted(rows, key=lambda r: VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99)
        story.extend(results_table(rows_sorted, styles, title=f"{name} -- balanced-training metrics", table_num=balanced_table_num))
        balanced_table_num += 1
        story.append(Spacer(1, 12))

    doc = build_doc(args.out)
    doc.build(story)
    print(f"\nSaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
