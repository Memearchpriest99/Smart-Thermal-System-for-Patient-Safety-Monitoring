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
# Config-D has no checkpoint and no registry entry (it is an evaluation-time
# pipeline composed in scripts/, not a ThermalAlgorithm), so it never appears
# in any result JSON -- it is listed in DETECTOR_ORDER purely so the contact
# group carries its caveat block and points at the S1 derivation.
CONFIG_D_NAME = "RBTCT (Rule-Based Temporal Contact Tracker)"

# Detectors whose SECTION 2 (natural-ratio) numbers are known-invalid rather
# than merely poor. Their rows stay in the report -- they are real
# measurements of a real checkpoint, and deleting measurements to make a
# report look tidier is not something this project does -- but the table
# title has to carry the warning, because a reader scrolling to a detector
# lands on the table, not on the caveat above it. Without this, S2's
# ThermoX3D table reads as a normal result: 0% F1 across the board and a
# meaningless "100.0% accuracy" on the ONNX/fp16/bf16 variants (a constant
# always-negative classifier scoring 100% on an all-negative variant subset).
S2_INVALID_RESULTS = {
    "ThermoX3DDetector": "INVALID -- degenerate checkpoint (tp=0); precision sweep is in S4",
}

# The single most important qualification in this report. Attached to every
# contact detector, because it applies to all of them equally and because a
# reader who sees only one detector's table would otherwise draw exactly the
# wrong conclusion from it.
CONTACT_CV_CAVEAT = (
    "<b>READ THIS BEFORE ANY CONTACT NUMBER IN THIS REPORT.</b> Every contact figure quoted "
    "elsewhere comes from ONE fixed train/test split. The corpus contains 246 contact-positive "
    "frames spread over exactly FIVE scenes, and the fixed split locks three of those five into "
    "test permanently -- so single-split contact numbers are extremely fragile. "
    "scripts/cv_contact_detectors.py therefore re-scores every contact detector under "
    "leave-one-scene-out cross-validation (each positive scene is the test fold once; all other "
    "scenes train/tune), pooling the folds' frames. Results:"
    "<br/><br/>"
    "<font face=\"Courier\" size=\"7.5\">"
    "detector      pooled_F1  recall   TNR   bal.acc    MCC<br/>"
    "always-alarm      0.432   1.000  0.000    0.500  0.000  &lt;- trivial baseline<br/>"
    "RBTCT             0.431   0.512  0.671    0.592  0.169<br/>"
    "ThermoX3D         0.448   0.907  0.187    0.547  0.113<br/>"
    "OR-ensemble       0.442   0.935  0.130    0.532  0.092<br/>"
    "</font>"
    "<br/>"
    "<b>Finding 1 -- F1 is uninformative on this task.</b> The pooled positive base rate is "
    "27.5%, so a detector that simply alarms on every frame scores F1 0.432. All three real "
    "detectors score 0.431-0.448. Judged on F1 alone, none of them is distinguishable from a "
    "broken always-on detector, and any F1-based ranking between them is noise. F1 ignores true "
    "negatives, which is precisely the quantity that matters when the operational failure mode is "
    "alarm fatigue. "
    "<b>Finding 2 -- once true negatives are credited, the ranking inverts.</b> By balanced "
    "accuracy and MCC (both of which reward correctly rejecting negatives), RBTCT is the "
    "strongest detector (0.592 / 0.169), ThermoX3D is marginal (0.547 / 0.113), and the "
    "OR-ensemble is the WEAKEST (0.532 / 0.092) -- it inherits both detectors' false alarms while "
    "adding little recall. The 'run both, alarm if either fires' posture recommended by the "
    "earlier investigation is hereby measured and NOT supported. "
    "<b>Finding 3 -- every detector has a total-failure fold.</b> Per-fold F1: RBTCT "
    "[0.00, 0.69, 0.27, 0.48, 0.39], ThermoX3D [0.29, 0.61, 0.00, 0.95, 0.38]. RBTCT scores zero "
    "on 2men_clash; ThermoX3D scores zero on 2ppl_hug. A single split showing either detector at "
    "~0.5 F1 conceals this entirely. Note also that ThermoX3D's 0.95 on 3pp_surprise is NOT skill: "
    "that scene is 95.7% positive, where always-alarm scores 0.978 -- the detector scored BELOW "
    "the trivial baseline on its best-looking fold. "
    "This CV reproduces the independent 4-fold result from the original 2026-07 investigation "
    "(pooled 44.3%, per-fold 32/51/65/<b>0</b>%) including the zero fold, from a separate "
    "implementation five weeks apart. Full data: reports/contact_cv_results.json."
)

# Variants that CANNOT exist for a given detector, and the reason. Rendered as
# explicit N/A rows rather than silently omitted -- a missing row is
# indistinguishable from "we forgot to run it", which is precisely the
# ambiguity this table is supposed to remove. Every reason below is either a
# property of the detector's implementation or a reproducible exception, not a
# guess.
_SKLEARN_FP = ("scikit-learn detector: libsvm/liblinear compute their decision function in "
               "float64 on CPU, so there is no mixed-precision code path to engage and no "
               "tensor-core benefit to gain -- casting inputs would add a no-op cast with "
               "identical arithmetic underneath")
_RULE_BASED = ("rule-based detector (is_trainable=False): no torch module and no ONNX graph "
               "exist, so there is nothing to cast or quantize -- its parameters are "
               "thresholds, not weights")

VARIANT_UNAVAILABLE: dict[str, dict[str, str]] = {
    "FireSVMDetector": {"fp16": _SKLEARN_FP, "bf16": _SKLEARN_FP},
    "HOGSVMDetector": {
        "fp16": _SKLEARN_FP, "bf16": _SKLEARN_FP,
        # Reproducible: onnxruntime.quantization.quantize_dynamic on the
        # skl2onnx-exported LinearSVC graph.
        "int8": ("ONNX Runtime dynamic quantization fails on the skl2onnx-exported LinearSVC "
                 "graph with ValueError: 'Failed to find proper ai.onnx domain' -- the exported "
                 "graph uses skl2onnx's ai.onnx.ml operator domain, which ORT's quantizer does "
                 "not handle"),
    },
    "ThermoX3DDetector": {
        # Reproducible with the same call that succeeds for MobileNet-SSD/MV-STGCN.
        "int8": ("ONNX Runtime dynamic quantization fails on the 3-D CNN graph with "
                 "ValueError: 'Expected onnx::Conv_662 to be an initializer' -- the legacy "
                 "TorchScript ONNX exporter emits that Conv weight as a graph input rather "
                 "than an initializer, which the quantizer requires"),
    },
    "OtsuFireDetector": {v: _RULE_BASED for v in ("onnx_fp32", "fp16", "bf16", "int8")},
    "AdaptiveThresholdDetector": {v: _RULE_BASED for v in ("onnx_fp32", "fp16", "bf16", "int8")},
    CONFIG_D_NAME: {v: _RULE_BASED for v in ("onnx_fp32", "fp16", "bf16", "int8")},
}


S4_VARIANT_UNAVAILABLE: dict[str, dict[str, str]] = {
    "ThermoX3DDetector": {
        "int8": ("quantization of the working (balanced) checkpoint SUCCEEDS but the resulting "
                 "graph fails to load: onnxruntime raises 'Node (/streams.2/block1/attention/"
                 "channel/mlp/mlp.0_1/Gemm_MatMul_quant) Op (MatMulInteger) [ShapeInferenceError] "
                 "Incompatible dimensions for matrix multiplication' -- the CBAM channel-attention "
                 "MLP is the same sub-graph the quantizer already needed a DefaultTensorType "
                 "work-around for. Note this is a DIFFERENT failure from the S2 checkpoint's, "
                 "which aborts earlier during quantization itself"),
    },
}


def _with_unavailable_rows(name: str, rows: list[dict], table: dict | None = None) -> tuple[list[dict], list[str]]:
    """Append a placeholder row for each variant this detector cannot support.
    Returns (rows, reason_notes) -- the notes are rendered under the table."""
    have = {r.get("variant") for r in rows}
    notes: list[str] = []
    for variant in VARIANT_ORDER:
        reason = (table if table is not None else VARIANT_UNAVAILABLE).get(name, {}).get(variant)
        if reason and variant not in have:
            rows = rows + [{"variant": variant, "unavailable": True}]
            notes.append(f"<b>{variant}</b>: not available -- {reason}.")
    return rows, notes


_HOMOGRAPHY_LIMIT = (
    "<b>Both homography-based contact detectors are limited by a calibration this sensor cannot "
    "support.</b> GeometricContactDetector and MVSTGCNDetector each work by projecting per-camera "
    "detections onto a shared floor plane through a homography, then measuring inter-person "
    "distance there. That requires a properly calibrated homography per camera -- and at this "
    "sensor's resolution (80x62) we were not able to obtain one: the Hot-Point Calibration "
    "procedure needs heated markers resolvable to a few pixels at known floor coordinates, and at "
    "this resolution the marker centroids are too coarse to solve a reliable transform. The "
    "self-calibrated fallback carries roughly +/-15 px of error, which is comparable to the "
    "contact decision threshold itself (delta ~18 px) -- i.e. the measurement is barely more "
    "informative than its own noise. <b>Consequence for reading this report:</b> "
    "GeometricContactDetector is presented as a derivation only (S1 §5) with no results table, "
    "because any number would characterise the calibration rather than the algorithm; and "
    "MVSTGCNDetector's numbers below, which ARE shown, must be read the same way -- as the result "
    "of an uncalibrated geometric front-end, not as a verdict on the graph-network architecture "
    "behind it. This is also the motivation for RBTCT (S1 §8), which discards the floor-plane "
    "projection entirely and measures contact in image space."
)

DETECTOR_CAVEATS = {
    "GeometricContactDetector": (
        _HOMOGRAPHY_LIMIT +
        "<br/><br/><b>No results are reported for this detector in any section.</b> It is included "
        "here, and derived in full in S1 §5 (DLT + SVD homography, foot-point projection, "
        "multi-view fusion, the delta_m distance rule), because the derivation documents the "
        "approach the project started from and the reason it was superseded -- not because the "
        "implementation is missing. It is present and tested in "
        "thermal_algorithms/contact_detection/geometric.py."
    ),
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
        _HOMOGRAPHY_LIMIT + "<br/><br/>"
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
        "correctly. <b>Only the fp32 baseline is listed here.</b> The ONNX/fp16/bf16/int8 sweep "
        "for this architecture was run against the WORKING (balanced) checkpoint and appears in "
        "S4 instead -- measuring the latency of a constant classifier at four precisions "
        "characterises nothing, and four more rows of zeros would only obscure the one row that "
        "matters here."
    ),
    CONFIG_D_NAME: (
        "The strongest HOMOGRAPHY-FREE contact detector found in this project, and the only one "
        "that uses no contact-specific training at all -- it composes the MobileNet-SSD person "
        "detector with a thermal-residual blob test, and every remaining parameter is a threshold "
        "tuned on validation. Its central observation is that when two people touch, their thermal "
        "signatures merge into ONE connected warm region while the person detector still reports "
        "TWO boxes; that disagreement is the contact signal, measured entirely in image space, so "
        "it sidesteps the homography noise (+/-15px, against a delta~18px decision threshold) that "
        "limits both homography-based contact detectors above. Full mathematical derivation -- "
        "adaptive residual threshold, the exactly-two-body merge predicate, the veto-based "
        "cross-camera quorum, and the 1-D morphological opening/closing on the decision stream -- "
        "is S1 &sect;8. It is not a ThermalAlgorithm subclass and has no checkpoint, so it falls "
        "outside eval_all_detectors.py's registry-driven sweep; the table below is produced by a "
        "dedicated driver, scripts/eval_config_d.py, on the IDENTICAL 470 held-out frames every "
        "other contact detector in this report is scored on. That driver applies two protocol "
        "tightenings over the historical config-D runs, both of which make this number stricter "
        "rather than more flattering: (1) the raw-input MobileNet-SSD it depends on is retrained "
        "with the contact TEST scenes excluded -- the original runs trained the person detector on "
        "a within-scene timeline split that included frames from the very scenes contact was then "
        "scored on, which is harmless for a person detector but leaks into a contact evaluation; "
        "and (2) the morphology parameters (l_open, l_close) are tuned on non-test scenes only and "
        "applied unchanged to test. For reference, the historical figures were F1 60.8% on a "
        "single within-scene split and 44.3% pooled over a 4-fold timeline cross-validation "
        "(per-fold 32/51/65/0%); the measured 52.7% below sits between them, as expected. "
        "<b>Config-D OUTPERFORMS the trained network on these frames: 52.7% F1 vs "
        "ThermoX3DDetector's 39.8% (S4), driven by far higher recall (83.6% vs 36.1%).</b> The two "
        "sit at opposite ends of the same tradeoff -- config-D catches most contacts but "
        "false-alarms on 46.8% of negative frames, ThermoX3D false-alarms on only 15.8% but misses "
        "two thirds of contacts -- so neither is simply 'better', and an operational choice "
        "between them is a decision about which failure is more tolerable in a ward. That a "
        "hand-written rule using NO contact training data beats a tuned network is itself the "
        "finding, and it reproduces the robust negative result from the original investigation: "
        "every learned variant of this rule underperformed the rule at this data scale (see "
        "reports/historical_investigations.md &sect;4). <b>One asymmetry to disclose:</b> "
        "config-D's morphology was tuned on 3pp_surprise alone, because 2men_clash -- the "
        "freshly-annotated scene ThermoX3D validates on -- exists only as a per-scene "
        "contact_labels.csv and was never appended to the consolidated waveshare_work/"
        "contact_labels.csv that this driver reads. The held-out TEST frames are identical for "
        "both detectors, so the comparison above is sound, but the two tuning sets differ. "
        "Promoting config-D to a first-class detector under thermal_algorithms/contact_detection/ "
        "remains open work."
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

# Report structure: detectors are grouped by TASK and presented in the order
# Human -> Fire -> Contact (project owner's directive, 2026-08-09), matching
# the order the runtime pipeline itself evaluates them and the order the
# derivations section introduces them. Both the natural-ratio (S2) and
# balanced (S4) result sections use the same grouping, so the two are
# directly comparable section-by-section.
TASK_ORDER = ["human", "fire", "contact"]
TASK_TITLES = {
    "human": "Human Detection",
    "fire": "Fire Detection",
    "contact": "Contact Detection",
}
DETECTOR_TASK = {
    "AdaptiveThresholdDetector": "human",
    "HOGSVMDetector": "human",
    "MobileNetSSDDetector": "human",
    "OtsuFireDetector": "fire",
    "FireSVMDetector": "fire",
    "GeometricContactDetector": "contact",
    "MVSTGCNDetector": "contact",
    "ThermoX3DDetector": "contact",
    CONFIG_D_NAME: "contact",
}

DETECTOR_ORDER = [
    # human
    "AdaptiveThresholdDetector", "HOGSVMDetector", "MobileNetSSDDetector",
    # fire
    "OtsuFireDetector", "FireSVMDetector",
    # contact -- GeometricContactDetector is listed deliberately WITHOUT any
    # results: it is derivation-only (S1 §5), and its entry exists so the
    # report states why rather than leaving a silent gap in the contact group.
    "GeometricContactDetector", "MVSTGCNDetector", "ThermoX3DDetector", CONFIG_D_NAME,
]
VARIANT_ORDER = ["fp32_baseline", "onnx_fp32", "fp16", "bf16", "int8"]


def _test_set_provenance(all_paths) -> str:
    """Build the per-task test-set table from what the result JSONs actually
    recorded, rather than restating it by hand.

    This answers "was every detector for a given task scored on the same test
    set?" -- a question the report previously left the reader to take on
    trust. Deriving it from the payloads means it cannot drift the way a
    hand-written claim can (and did, for the ONNX execution providers).
    """
    key = {"fire": "fire_test_scenes", "human": "human_test_scenes",
           "contact": "contact_test_scenes"}
    scenes: dict[str, set] = {t: set() for t in key}
    frames: dict[str, set] = {t: set() for t in key}
    dets: dict[str, set] = {t: set() for t in key}
    for path in all_paths:
        payload = _load_json(path)
        if not payload:
            continue
        rows = payload.get("results", [])
        # Only credit a payload's scene list for tasks it actually has rows
        # for. Every eval script writes all three scene-list keys regardless
        # of what it evaluated, so a contact-only run (--only ThermoX3DDetector)
        # still records a human_test_scenes list -- one drawn WITHOUT the
        # --extra-human-test-scenes empty_room union, since that flag wasn't
        # passed. Counting those would report a human-detection "MISMATCH"
        # that does not exist in any measured row. (Verified: every payload
        # containing human rows records the same 4 scenes.)
        tasks_present = {r.get("task") for r in rows}
        for task, k in key.items():
            if payload.get(k) and task in tasks_present:
                scenes[task].add(tuple(sorted(payload[k])))
        for r in rows:
            t = r.get("task")
            if t in dets:
                dets[t].add(r.get("detector_name", "?"))
                n = sum(r.get(c) or 0 for c in ("tp", "tn", "fp", "fn"))
                if n:
                    frames[t].add(n)

    lines = []
    for task in TASK_ORDER:
        sc, fr, dt = scenes[task], frames[task], sorted(dets[task])
        if not dt:
            continue
        scene_txt = (", ".join(list(sc)[0]) if len(sc) == 1
                     else "<b>MISMATCH</b>: " + " | ".join(", ".join(x) for x in sc))
        frame_txt = (f"{list(fr)[0]} frames" if len(fr) == 1
                     else f"<b>MISMATCH</b>: {sorted(fr)}")
        lines.append(
            f"<b>{TASK_TITLES[task]}</b> -- {scene_txt} ({frame_txt}); "
            f"shared by: {', '.join(dt)}.")
    return "<br/>".join(lines)



def _emit_unavailable_notes(story, styles, notes: list[str]) -> None:
    """Render the "why this variant does not exist" lines under a table."""
    if not notes:
        return
    style = ParagraphStyle("UnavailNote", parent=styles["Bodyc"], fontSize=7, leading=9,
                           textColor=colors.HexColor("#444444"))
    for n in notes:
        story.append(Paragraph(n, style))



def _with_cv_caveat(name: str, caveat):
    """Prepend the leave-one-scene-out CV caveat to every CONTACT detector's
    block. Contact is the only task in this report whose single-split numbers
    are actively misleading, and the qualification applies to all of its
    detectors equally -- attaching it per-detector means a reader who jumps
    straight to one table still sees it."""
    if DETECTOR_TASK.get(name) != "contact":
        return caveat
    return CONTACT_CV_CAVEAT + "<br/><br/>" + (caveat or "")


def _emit_task_heading(story, styles, name: str, current_task: str | None) -> str | None:
    """Emit a task-level (H2) heading when the detector list crosses from one
    task group into the next. Returns the new 'current task' marker. Keeping
    this in one helper means S2 and S4 can never drift out of sync on
    grouping/ordering."""
    task = DETECTOR_TASK.get(name)
    if task is not None and task != current_task:
        story.append(Paragraph(TASK_TITLES.get(task, task.title()), styles["H2c"]))
        return task
    return current_task

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
    # human
    "AdaptiveThresholdDetector", "HOGSVMDetector", "MobileNetSSDDetector",
    # fire
    "OtsuFireDetector", "FireSVMDetector",
    # contact
    "ThermoX3DDetector",
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
        "(input tensor (3,5,62,80)) per the project owner's direction. The training-loop bug is "
        "CONFIRMED fixed: recall went from 0% to ~100% and the model's raw confidence output now "
        "varies meaningfully with input (the dead-network signature -- bit-identical output "
        "regardless of input -- is gone). "
        "<b>CURRENT CHECKPOINT -- real-only + augmentation + Optuna (2026-08-09):</b> synthetic "
        "data was dropped entirely (see the failed experiment below) and the detector retrained on "
        "waveshare_work alone, with augmentation standing in for corpus size: horizontal flip, "
        "+/-15 deg rotation, and additive N(0,sigma) noise, applied to RAW frames before "
        "preprocessing so sigma stays interpretable in degrees C (thermal_algorithms/training/"
        "augment.py). Geometric transforms are drawn once per contiguous run -- constant across "
        "all 5 frames and all 3 cameras -- because a per-frame angle would inject artificial "
        "motion that this detector's temporal convolutions would learn as signal; noise is "
        "per-frame i.i.d., which is what sensor noise actually is. Two further directives: a "
        "window is labelled positive when >=2 of its 5 triplets are contact-positive (replacing "
        "the old label-by-last-frame rule, under which one borderline frame flipped a whole "
        "window), and weights initialise as identity matrices (Dirac kernels for convs, partial "
        "identity for linear layers). 30 Optuna trials (TPE + median pruning) tuned the rest. "
        "<b>Held-out result: 71.7% acc / 44.4% prec / 36.1% rec / 39.8% F1 (tp=44, tn=293, fp=55, "
        "fn=78).</b> F1 is statistically flat against both earlier regimes (42.1%, 41.1%) but the "
        "OPERATING POINT moved decisively: precision nearly doubled (26.6% -> 44.4%) and the "
        "false-alarm rate collapsed from 96.6% to <b>15.8%</b> (fp 336 -> 55). This is the first "
        "contact checkpoint in this project that is not effectively an always-alarm detector -- a "
        "distinction F1 alone hides, and the one that matters for a psychiatric-ward alarm, where "
        "a 96% false-alarm rate is unusable regardless of recall. The cost is real and must not be "
        "glossed: recall fell to 36.1%, i.e. it now misses roughly two thirds of contact events, "
        "which is its own safety problem. Empirical notes from the search: identity init BEAT "
        "standard init (the directive was validated, not merely followed), and the search moved "
        "noise sigma away from the specified 2.0 down to ~1.08 degC -- consistent with 2 degC "
        "approaching the corpus's own frame-wide std of 2.39 degC and beginning to bury the signal "
        "rather than regularise it. <b>The binding constraint is unchanged and bounds all of the "
        "above:</b> only two non-test scenes contain ANY contact positives -- 3pp_surprise (66, and "
        "flagged block-labelled/suspect by the label audit in reports/historical_investigations.md "
        "S5.1) and 2men_clash (58, freshly annotated). A leakage-free scene-level split puts one on "
        "each side, so every positive training window here derives from a single 66-frame clip. "
        "Augmentation multiplies frames; it cannot manufacture a second contact scenario, and the "
        "val-to-test recall drop (100% -> 36.1%) is exactly that generalisation gap. Full study in "
        "reports/contact_optuna_study.json; driver scripts/train_contact_real_optuna.py. "
        "<b>SUPERSEDED EXPERIMENT -- source-separated synth/real (2026-08-09):</b> at the project owner's "
        "direction the data regime was then changed so that TRAINING draws only on synthetic data "
        "(synth_room_1..5), VALIDATION mixes held-out synthetic with all 15 real non-test contact "
        "scenes (including the freshly-annotated 2men_clash), and TEST remains real-only -- the "
        "goal being to stop spending scarce real contact data on training and spend it on the "
        "decision-threshold calibration instead. Training chunks and the validation set are both "
        "balanced to an EXACT 50/50 positive/negative WINDOW split. "
        "<b>Held-out result: 30.4% acc / 26.3% prec / 93.4% rec / 41.1% F1 (tp=114, tn=29, fp=319, "
        "fn=8) -- statistically indistinguishable from the previous mixed-source regime's 42.1% F1 "
        "/ 26.6% precision. Moving real data out of training did NOT fix the precision problem.</b> "
        "The diagnostic that explains why is a measured DOMAIN GAP between the two sources: raw "
        "synthetic frames span only ~1 degree C (ambient ~29.9C, std ~0.11, people peaking ~31C) "
        "whereas real frames span ~23 degrees C (13-36C, std ~2.39, bodies at 36C against a "
        "structured background). After preprocessing, real data sits ~+7.9 sigma outside the "
        "synthetic training distribution. Per-source normalisation (each source scaled by its own "
        "statistics into a common ~N(0,1) space; the checkpoint saved under REAL statistics since "
        "all test/deployment input is real) recovers most of the numeric mismatch -- it cut "
        "validation loss from 7.09 to 0.81 -- but it cannot manufacture the background thermal "
        "STRUCTURE that synthetic scenes lack, and background structure is precisely what drives "
        "real-world false positives. The honest conclusion is that this synthetic corpus does not "
        "substitute for real contact data on this task; the binding constraint remains the ~250 "
        "real positive contact frames in the whole corpus (see reports/historical_investigations.md "
        "&sect;3.2 and &sect;4.1, where two independent earlier investigations reached the same "
        "data-limited conclusion). Reported as a feasibility result, not a benchmark."
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
        default=["full_corpus_eval_*.json", "config_d_eval.json",
                 "adaptive_threshold_eval.json", "otsu_threshold_eval.json"],
        help="Glob patterns (relative to reports/) for result JSON files to merge. "
             "NOTE: eval_*smoketest*.json is deliberately NOT included. Those files hold "
             "results from checkpoints_baseline_waveshare -- placeholder smoke-test weights "
             "-- and merging them alongside the real checkpoints_full_corpus results "
             "produced TWO rows per variant per detector, visually identical because "
             "_normalize_variant() folds 'fp32_baseline_smoketest' back to "
             "'fp32_baseline', but with completely different numbers (e.g. FireSVM 71.6% "
             "acc/15.0% recall from the real run vs 29.5%/100% from the smoke test, and a "
             "cold-start 831ms/13s-p95 latency). "
             "config_d_eval.json is listed explicitly: config-D is rule-based (no training, no "
             "checkpoint) so it has no natural-vs-balanced distinction and belongs in S2 only. "
             "It deliberately matches NEITHER the balanced globs NOR the two patterns above, so "
             "it can never be double-counted into both sections.",
    )
    ap.add_argument("--image-dir", default=str(_REPO_ROOT / "reports" / "_report_images"))
    ap.add_argument(
        "--balanced-result-globs", nargs="*",
        default=["balanced_training_eval.json", "otsu_balanced_eval.json",
                 "adaptive_threshold_balanced_eval.json",
                 "balanced_eval_onnx.json", "balanced_eval_quantized.json"],
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
        "Detectors are grouped by task and presented in the order <b>Human, Fire, Contact</b> -- "
        "matching both the order the runtime pipeline evaluates them and the order S1's "
        "derivations introduce them. The balanced-training section (S4) repeats the identical "
        "grouping, so the two sections can be read side by side detector-for-detector. "
        "Within each task, every detector and every precision variant is scored on the exact same "
        "held-out waveshare_work scenes (see scripts/eval_all_detectors.py:"
        "build_waveshare_test_split); the per-task split is listed below and is derived from what "
        "the result files themselves recorded, so it cannot drift from the numbers it describes. "
        "\"Mean ms\" / \"p95 ms\" are single-frame inference latency, and the <b>Measured on</b> "
        "column states what each row was timed on -- a latency is meaningless without it. Values "
        "marked * were reconstructed from the deterministic code path that produced the row rather "
        "than captured during the run; unmarked values were captured at measurement time.",
        styles["Bodyc"],
    ))
    story.append(Paragraph(_test_set_provenance(result_paths), styles["Bodyc"]))
    story.append(Paragraph(
        "<b>Two input-pipeline asymmetries to note</b> -- these are not test-set differences (the "
        "frames are identical) but they do affect strict like-for-like comparison. (1) "
        "OtsuFireDetector is evaluated on RAW frames while FireSVMDetector receives "
        "GlobalNormPreprocessor output: Otsu's thresholds are absolute degrees C, so feeding it a "
        "background-subtracted residual would be meaningless. (2) RBTCT uses Tateno residuals plus "
        "its own separately-trained raw-input MobileNet-SSD, while MVSTGCN and ThermoX3D use "
        "GlobalNorm -- derived in S1 §8.6, where the raw-detector/residual-blob split is shown to "
        "be a requirement of the method rather than a tuning choice. "
        "<b>Protocol warning:</b> reports/contact_cv_results.json uses a different protocol "
        "entirely (leave-one-scene-out over 5 scenes, 894 pooled frames, MVSTGCN not included) and "
        "its figures must never be compared row-wise against the 470-frame tables here.",
        styles["Bodyc"],
    ))

    chart_dir = Path(args.image_dir) / "charts"
    table_num = 1
    current_task: str | None = None
    for name in DETECTOR_ORDER:
        current_task = _emit_task_heading(story, styles, name, current_task)
        rows = by_detector.get(name)
        if not rows:
            story.append(Paragraph(name, styles["H3c"]))
            pending_caveat = _with_cv_caveat(name, DETECTOR_CAVEATS.get(name))
            if pending_caveat:
                caveat_style = ParagraphStyle("Caveat", parent=styles["Bodyc"], backColor=colors.HexColor("#fff4e0"), borderPadding=6)
                story.append(Paragraph(pending_caveat, caveat_style))
            else:
                story.append(Paragraph("No evaluated results available yet.", styles["Bodyc"]))
            continue
        rows, unavail_notes = _with_unavailable_rows(name, rows)
        rows_sorted = sorted(rows, key=lambda r: VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99)
        caveat = _with_cv_caveat(name, DETECTOR_CAVEATS.get(name))
        story.append(Paragraph(name, styles["H3c"]))
        if caveat:
            caveat_style = ParagraphStyle("Caveat", parent=styles["Bodyc"], backColor=colors.HexColor("#fff4e0"), borderPadding=6)
            story.append(Paragraph(caveat, caveat_style))
        warn = S2_INVALID_RESULTS.get(name)
        title = f"{name} -- metrics by variant" + (f"  [{warn}]" if warn else "")
        story.extend(results_table(rows_sorted, styles, title=title, table_num=table_num))
        table_num += 1
        _emit_unavailable_notes(story, styles, unavail_notes)

        # Chart only rows that were actually measured: the "unavailable"
        # placeholders carry no latency, and plotting them as 0 ms would
        # imply a variant is infinitely fast rather than non-existent.
        measured = [r for r in rows_sorted
                    if not r.get("unavailable")
                    and isinstance(r.get("mean_inference_ms"), (int, float))]
        labels = [r["variant"] for r in measured]
        latencies = [r["mean_inference_ms"] for r in measured]
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
        "to S2's. This section reports the fp32 baseline only -- the ONNX/fp16/bf16/int8 "
        "sweep was run against the natural-ratio checkpoints in S2, so the absence of those "
        "rows here means 'not run for the balanced checkpoints', not 'not supported'; "
        "S2 carries the per-detector convertibility notes.", styles["Bodyc"],
    ))

    balanced_table_num = 1
    balanced_task: str | None = None
    for name in BALANCED_DETECTOR_ORDER:
        balanced_task = _emit_task_heading(story, styles, name, balanced_task)
        rows = by_detector_balanced.get(name)
        story.append(Paragraph(name, styles["H3c"]))
        caveat = _with_cv_caveat(name, BALANCED_CAVEATS.get(name))
        if caveat:
            caveat_style = ParagraphStyle("BalancedCaveat", parent=styles["Bodyc"], backColor=colors.HexColor("#e6f2ff"), borderPadding=6)
            story.append(Paragraph(caveat, caveat_style))
        if not rows:
            story.append(Paragraph("No evaluated results available yet.", styles["Bodyc"]))
            continue
        rows, unavail_notes = _with_unavailable_rows(name, rows, S4_VARIANT_UNAVAILABLE)
        rows_sorted = sorted(rows, key=lambda r: VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99)
        story.extend(results_table(rows_sorted, styles, title=f"{name} -- balanced-training metrics", table_num=balanced_table_num))
        balanced_table_num += 1
        _emit_unavailable_notes(story, styles, unavail_notes)
        story.append(Spacer(1, 12))

    doc = build_doc(args.out)
    doc.build(story)
    print(f"\nSaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
