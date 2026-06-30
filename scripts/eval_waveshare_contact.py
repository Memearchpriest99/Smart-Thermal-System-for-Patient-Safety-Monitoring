"""Contact ("touch") detection — end-to-end loop on the Waveshare dataset.

Pipeline (§ 4.4.3.1, GeometricContactDetector)
----------------------------------------------
    3 thermal frames (cam0, cam1, cam2)
        ├─ Tateno background residual           (per-channel, runtime-matched)
        ├─ MobileNet-SSD.predict  per camera    → list[Detection] (class 1 = person)
        ▼
    GeometricContactDetector.predict(frames, detections=(d0,d1,d2))
        ├─ foot-point  P = (x+w/2, y+h)
        ├─ homography  H_k · [u,v,1]ᵀ           → reference-plane point
        ├─ fuse        cluster within ε, validate (N=1 discard / N=3 outlier)
        └─ contact     pairwise dist < δ        → ContactEvent

Homography by self-calibration
------------------------------
No floor coordinates were recorded for ``calibrate_room``. Instead we exploit
that the single calibration subject is seen by all three cameras at once: the
foot-point in each view is an image of one common floor point. RANSAC + DLT
over the ~146-frame track yields a planar homography H_{k->0} mapping every
camera onto **cam0's floor plane** (cam0 = identity). See
``multi_view.homography.calibrate_floor_homographies_from_tracks``.

Consequence: ε and δ are in **cam0 floor-pixel units, not metres** (and are
perspective-distorted). They are tuned empirically here. ε is seeded from the
calibration reprojection residual; δ is swept on a validation split.

Protocol
--------
* MobileNet-SSD is trained once on the human-detection train split (person
  boxes, class 1) of every non-empty scene, on Tateno-preprocessed frames.
* Contact labels come from the combined ``contact_labels.csv``
  (``session,frame_idx,contact``). Every scene in that file is evaluated:
  the 4 contact-positive scenes plus all-negative scenes (true negatives).
* Per scene the labeled frames are split 50/50 (sequential, no leakage) into
  val (δ tuning) and test (reporting).
* A frame is a positive prediction iff any actor pair is within δ on the
  reference plane.

Note: the SSD person detector and the contact decision are evaluated as one
stack; person-box training may overlap contact frames, but the contact label
is independent of the person boxes, so the contact decision itself is the
quantity under test.

Outputs a readable report to stdout and JSON to
reports/waveshare_contact_results.json.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.core.types import Frame
from thermal_algorithms.contact_detection.geometric import GeometricContactDetector
from thermal_algorithms.contact_detection.multi_view.homography import (
    calibrate_floor_homographies_from_tracks,
    project_foot_point,
)
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from thermal_algorithms.training import DatasetIndex, PERSON_CLASS_ID
from thermal_algorithms.training.label_io import load_yolo_labels
from thermal_algorithms.training.metrics import (
    binary_confusion_matrix,
    BinaryConfusionMatrix,
)

try:
    import torch
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
    _TORCH_OK = True
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _TORCH_OK = False
    _DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROFILE = WAVESHARE_26984
DATASET_ROOT = "datasets/waveshare_work"
CHANNELS = (0, 1, 2)
REF_CAMERA = 0
EMPTY_SCENE = "empty_room"
CALIB_SCENE = "calibrate_room"
CONTACT_CSV = Path(DATASET_ROOT) / "contact_labels.csv"

TRAIN_FRAC = 0.80            # SSD train split (person boxes)
VAL_FRAC = 0.10
SSD_EPOCHS = 25

VAL_SPLIT = 0.50             # contact frames: first half val (δ tuning), rest test
DELTA_GRID_PX = [4, 6, 8, 10, 12, 15, 18, 22, 28, 36]
MIN_SOURCES_GRID = [1, 2]    # cameras required to confirm an actor (2 = stock fusion)
EPSILON_GRID_PX = [3, 6, 10, 15, 20]   # cross-camera dedup / clustering radius

OUT_JSON = Path(__file__).resolve().parent.parent / "reports" / "waveshare_contact_results.json"
SSD_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "mobilenet_ssd_detector" / "Waveshare_26984.thalg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# SessionMetadata.load_frame() re-reads the whole .npz per call; cache the
# decompressed array per (session, channel) so each file is read once.
_FRAMES_CACHE: dict[tuple[str, int], np.ndarray] = {}


def get_frame(session, ch: int, fi: int) -> Frame:
    key = (str(session.root), ch)
    arr = _FRAMES_CACHE.get(key)
    if arr is None:
        arr = session.load_frames(ch)
        _FRAMES_CACHE[key] = arr
    return Frame(data=arr[fi].astype(np.float32), timestamp=fi / PROFILE.sample_rate_hz,
                 camera_id=ch, metadata={"session_id": session.session_id})


def _split_indices(indices: list[int]) -> tuple[list[int], list[int], list[int]]:
    n = len(indices)
    n_train = max(1, int(n * TRAIN_FRAC))
    n_val = max(0, int(n * (TRAIN_FRAC + VAL_FRAC)) - n_train)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def _load_contact_labels(path: Path) -> dict[str, dict[int, int]]:
    """Parse combined contact_labels.csv → {scene: {frame_idx: contact}}."""
    out: dict[str, dict[int, int]] = defaultdict(dict)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["session"]][int(row["frame_idx"])] = int(row["contact"])
    return dict(out)


def _print_scene_table(rows, title):
    w = 24
    print(f"\n  {title}")
    hdr = (f"  {'Scene':<{w}}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  "
           f"{'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}  {'N':>5}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep); print(hdr); print(sep)
    for r in rows:
        print(f"  {r['scene']:<{w}}  {r['acc']:>6.1%}  {r['prec']:>7.1%}  "
              f"{r['rec']:>7.1%}  {r['f1']:>7.1%}  {r['tp']:>4d}  {r['tn']:>4d}  "
              f"{r['fp']:>4d}  {r['fn']:>4d}  {r['total']:>5d}")
    print(sep)


def _print_cm(cm: BinaryConfusionMatrix, title: str):
    total = cm.total
    pct = lambda v: f"{v/total*100:5.1f}%" if total else "  N/A "
    print(f"\n  {title}")
    print(f"  {'':25s}  {'Pred: Contact':>18}  {'Pred: None':>18}")
    print(f"  {'Truth: Contact (Pos)':25s}  {'TP='+str(cm.tp):>8} {pct(cm.tp):>8}  {'FN='+str(cm.fn):>8} {pct(cm.fn):>8}")
    print(f"  {'Truth: None (Neg)':25s}  {'FP='+str(cm.fp):>8} {pct(cm.fp):>8}  {'TN='+str(cm.tn):>8} {pct(cm.tn):>8}")
    print(f"  Accuracy={cm.accuracy:.1%}  Precision={cm.precision:.1%}  Recall={cm.recall:.1%}  "
          f"F1={cm.f1:.1%}  FalseAlarm={cm.false_alarm_rate:.1%}  Correct={cm.correct}/{total}")


def actors_from_dets(dets_per_cam, H, epsilon: float, min_sources: int):
    """Project foot-points to the reference plane, greedy single-linkage
    cluster within ``epsilon``, keep clusters seen by >= ``min_sources``
    cameras. ``min_sources=2`` reproduces GeometricContactDetector's fusion
    (single-source discard); ``min_sources=1`` keeps single-camera actors."""
    pts = []
    for cam, dets in enumerate(dets_per_cam):
        for d in dets:
            pts.append((project_foot_point(d.foot_point, H[cam]), cam))
    clusters: list[list] = []
    for xy, cam in pts:
        placed = False
        for cl in clusters:
            if any(((xy[0]-m[0]) ** 2 + (xy[1]-m[1]) ** 2) ** 0.5 < epsilon
                   for m, _ in cl):
                cl.append((xy, cam)); placed = True; break
        if not placed:
            clusters.append([(xy, cam)])
    actors = []
    for cl in clusters:
        if len({c for _, c in cl}) < min_sources:
            continue
        xs = [p[0] for p, _ in cl]; ys = [p[1] for p, _ in cl]
        actors.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return actors


def _contact_pred(world_xy: list[tuple[float, float]], delta: float) -> int:
    """1 if any pair of actor positions is within ``delta``."""
    for i in range(len(world_xy)):
        for j in range(i + 1, len(world_xy)):
            dx = world_xy[i][0] - world_xy[j][0]
            dy = world_xy[i][1] - world_xy[j][1]
            if (dx * dx + dy * dy) ** 0.5 < delta:
                return 1
    return 0


# ---------------------------------------------------------------------------
# Stage 1 — self-calibrate homography from the calibrate_room track
# ---------------------------------------------------------------------------

def calibrate_homography(idx: DatasetIndex):
    calib = idx.find(CALIB_SCENE)
    eh, ew = PROFILE.height, PROFILE.width
    points_per_frame: list[dict[int, tuple[float, float]]] = []
    for fi in range(calib.n_frames):
        fp: dict[int, tuple[float, float]] = {}
        for ch in CHANNELS:
            dets = load_yolo_labels(
                calib.frames_dir(ch) / f"frame_{fi:05d}.txt",
                frame_shape=(eh, ew), camera_id=ch, class_filter=[PERSON_CLASS_ID],
            )
            if dets:
                # Single calibration subject; if several boxes, take the largest.
                d = max(dets, key=lambda d: d.area)
                fp[ch] = d.foot_point
        if len(fp) >= 2:
            points_per_frame.append(fp)

    H, info = calibrate_floor_homographies_from_tracks(
        points_per_frame, ref_camera=REF_CAMERA, threshold_px=2.5, iters=3000,
    )
    return H, info, len(points_per_frame)


# ---------------------------------------------------------------------------
# Stage 2 — Tateno calibration + SSD training (person boxes)
# ---------------------------------------------------------------------------

def fit_preprocessors(idx: DatasetIndex) -> dict[int, TatenoPipeline]:
    empty = idx.find(EMPTY_SCENE)
    pre: dict[int, TatenoPipeline] = {}
    for ch in CHANNELS:
        calib = [get_frame(empty, ch, i) for i in range(empty.n_frames) if i % 2 == 0]
        pre[ch] = TatenoPipeline(PROFILE).fit(calib)
    return pre


def build_ssd_examples(idx: DatasetIndex, pre: dict[int, TatenoPipeline]):
    """Person-box train examples on Tateno-preprocessed frames (train split)."""
    eh, ew = PROFILE.height, PROFILE.width
    train_ex = []
    empty = idx.find(EMPTY_SCENE)

    for session in idx.sessions:
        if session.scene in (EMPTY_SCENE, CALIB_SCENE):
            continue
        for ch in CHANNELS:
            if ch not in session.channels_with_labels:
                continue
            labeled = sorted(
                int(p.stem.split("_", 1)[1])
                for p in session.frames_dir(ch).glob("frame_*.txt")
            )
            if not labeled:
                continue
            tr, _, _ = _split_indices(labeled)
            for fi in tr:
                proc = pre[ch].predict(get_frame(session, ch, fi))
                dets = load_yolo_labels(
                    session.frames_dir(ch) / f"frame_{fi:05d}.txt",
                    frame_shape=proc.shape, camera_id=ch, class_filter=[PERSON_CLASS_ID],
                )
                train_ex.append((proc, dets))

    # Empty-room odd frames → negatives (train split only).
    for ch in CHANNELS:
        odd = [i for i in range(empty.n_frames) if i % 2 == 1]
        tr, _, _ = _split_indices(odd)
        for fi in tr:
            train_ex.append((pre[ch].predict(get_frame(empty, ch, fi)), []))
    return train_ex


# ---------------------------------------------------------------------------
# Stage 3 — build contact records (SSD + fusion, one pass)
# ---------------------------------------------------------------------------

def build_contact_records(idx, pre, ssd):
    """For every labeled contact frame: run SSD per camera, cache detections.
    Fusion / contact thresholds are applied later so they can be swept."""
    labels = _load_contact_labels(CONTACT_CSV)
    records = []
    t0 = time.time()
    n_done = 0

    for scene in sorted(labels):
        try:
            session = idx.find(scene)
        except Exception:
            continue
        if not all(ch in session.channels_with_data for ch in CHANNELS):
            continue
        frame_indices = sorted(
            fi for fi in labels[scene] if 0 <= fi < session.n_frames
        )
        # Sequential val/test split per scene.
        cut = int(len(frame_indices) * VAL_SPLIT)
        split_of = {fi: ("val" if k < cut else "test")
                    for k, fi in enumerate(frame_indices)}

        for fi in frame_indices:
            dets = []
            for ch in CHANNELS:
                proc = pre[ch].predict(get_frame(session, ch, fi))
                dets.append(ssd.predict(proc))
            records.append({
                "scene": scene,
                "split": split_of[fi],
                "label": int(labels[scene][fi]),
                "dets": tuple(dets),
                "ndets": [len(d) for d in dets],
                "ts": fi / PROFILE.sample_rate_hz,
            })
            n_done += 1
            if n_done % 300 == 0:
                print(f"      ... {n_done} frames  "
                      f"({(time.time()-t0)/n_done*1000:.0f} ms/frame)", flush=True)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("  CONTACT DETECTION — Waveshare 26984 (80×62)  —  Geometric multi-view")
    print("=" * 78)
    if not _TORCH_OK:
        print("  ERROR: torch unavailable — MobileNet-SSD is required for this loop.")
        sys.exit(1)

    idx = DatasetIndex(DATASET_ROOT, sensor_profile=PROFILE)
    print(f"  Root: {DATASET_ROOT}   Sessions: {len(idx.sessions)}   Device: {_DEVICE}")

    # ── Stage 1: self-calibrate homography ───────────────────────────────────
    print("\n  [1/4] Self-calibrating homography from calibrate_room track ...", flush=True)
    H, cal_info, n_track = calibrate_homography(idx)
    print(f"        Used {n_track} multi-view track frames. Reference cam = {REF_CAMERA}.")
    resids = []
    for cam, ci in cal_info.items():
        if ci.get("reference"):
            print(f"        cam{cam}: reference plane (H = I)")
        else:
            print(f"        cam{cam}: {ci['n_inliers']}/{ci['n_pairs']} inliers, "
                  f"median residual {ci['median_residual_px']:.2f} px")
            resids.append(ci["median_residual_px"])
    epsilon = max(3.0, 2.0 * (float(np.median(resids)) if resids else 1.5))
    print(f"        epsilon (cluster threshold) = {epsilon:.2f} px (from residuals)")

    # ── Stage 2: preprocessing + SSD (load checkpoint if present) ────────────
    print("\n  [2/4] Calibrating Tateno + obtaining MobileNet-SSD ...", flush=True)
    pre = fit_preprocessors(idx)
    if SSD_CKPT.is_file():
        ssd = MobileNetSSDDetector.load(SSD_CKPT)
        print(f"        Loaded SSD checkpoint → {SSD_CKPT}")
    else:
        train_ex = build_ssd_examples(idx, pre)
        n_pos = sum(1 for _, d in train_ex if d)
        print(f"        SSD train examples: {len(train_ex)} (pos={n_pos}, neg={len(train_ex)-n_pos})")
        t = time.time()
        ssd = MobileNetSSDDetector(PROFILE, n_epochs=SSD_EPOCHS, batch_size=16,
                                   learning_rate=1e-3, device=_DEVICE)
        ssd.fit(train_ex, verbose=True)
        ssd.save(SSD_CKPT)
        print(f"        SSD trained ({time.time()-t:.1f}s) → saved {SSD_CKPT}")

    # ── Stage 3: build contact records (single SSD pass) ─────────────────────
    print("\n  [3/4] Running SSD over all labeled contact frames ...", flush=True)
    records = build_contact_records(idx, pre, ssd)
    val = [r for r in records if r["split"] == "val"]
    test = [r for r in records if r["split"] == "test"]
    vp = sum(r["label"] for r in val); tp_ = sum(r["label"] for r in test)
    print(f"        Records: {len(records)}  val={len(val)} (pos={vp})  "
          f"test={len(test)} (pos={tp_})")

    # Diagnostic: how many people does SSD actually find, and how many actors
    # survive fusion, on TRUE-contact frames? This reveals whether recall is
    # lost at detection (people merge into one blob) or at fusion (cross-camera
    # validation discards everything).
    pos = [r for r in records if r["label"] == 1]
    if pos:
        mean_dets = np.mean([sum(r["ndets"]) for r in pos])
        max_cam = np.mean([max(r["ndets"]) for r in pos])
        ge2_any = np.mean([1.0 if max(r["ndets"]) >= 2 else 0.0 for r in pos])
        a1 = np.mean([len(actors_from_dets(r["dets"], H, epsilon, 1)) for r in pos])
        a2 = np.mean([len(actors_from_dets(r["dets"], H, epsilon, 2)) for r in pos])
        print(f"\n        Diagnostic on {len(pos)} TRUE-contact frames:")
        print(f"          mean detections (all 3 cams) : {mean_dets:.2f}")
        print(f"          mean max dets in any one cam : {max_cam:.2f}  "
              f"(>=2 in some cam: {ge2_any:.0%} of frames)")
        print(f"          mean actors  min_sources=1   : {a1:.2f}")
        print(f"          mean actors  min_sources=2   : {a2:.2f}  (stock fusion)")

    # ── Stage 4: sweep (min_sources, ε, δ) on val, report test ───────────────
    print("\n  [4/4] Tuning (min_sources, ε, δ) on val (max F1), evaluating on test ...", flush=True)
    yt = [r["label"] for r in val]
    sweep = []
    print(f"        {'min_src':>7}  {'ε px':>5}  {'best δ':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    for ms in MIN_SOURCES_GRID:
        for eps in EPSILON_GRID_PX:
            actors_val = [actors_from_dets(r["dets"], H, eps, ms) for r in val]
            row_best = None
            for delta in DELTA_GRID_PX:
                yp = [_contact_pred(a, delta) for a in actors_val]
                cm = binary_confusion_matrix(yt, yp)
                sweep.append((ms, eps, delta, cm))
                if row_best is None or (cm.f1, cm.recall) > (row_best[1].f1, row_best[1].recall):
                    row_best = (delta, cm)
            d, cm = row_best
            print(f"        {ms:>7}  {eps:>5}  {d:>6}  {cm.precision:>6.1%}  "
                  f"{cm.recall:>6.1%}  {cm.f1:>6.1%}")
    best_ms, best_eps, best_delta, best_cm = max(
        sweep, key=lambda x: (x[3].f1, x[3].recall))
    print(f"\n        Best: min_sources={best_ms}, ε={best_eps} px, δ={best_delta} px  "
          f"(val F1={best_cm.f1:.1%})")

    # Final test pass. min_sources=2 is exactly GeometricContactDetector; for
    # min_sources=1 we use the relaxed local fusion (same projection + ε clustering).
    by_scene: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    if best_ms == 2:
        detector = GeometricContactDetector(homography=H, epsilon_m=float(best_eps),
                                             delta_m=float(best_delta))
        eh, ew = PROFILE.height, PROFILE.width
        for r in test:
            stub = tuple(Frame(data=np.zeros((eh, ew), np.float32),
                               timestamp=r["ts"], camera_id=ch) for ch in CHANNELS)
            ev = detector.predict(stub, detections=r["dets"])
            by_scene[r["scene"]][0].append(r["label"])
            by_scene[r["scene"]][1].append(1 if ev.any_contact else 0)
    else:
        for r in test:
            actors = actors_from_dets(r["dets"], H, best_eps, best_ms)
            by_scene[r["scene"]][0].append(r["label"])
            by_scene[r["scene"]][1].append(_contact_pred(actors, best_delta))

    rows, agg = [], BinaryConfusionMatrix(0, 0, 0, 0)
    for scene in sorted(by_scene):
        yt, yp = by_scene[scene]
        cm = binary_confusion_matrix(yt, yp)
        agg = agg + cm
        rows.append({"scene": scene, "acc": cm.accuracy, "prec": cm.precision,
                     "rec": cm.recall, "f1": cm.f1, "far": cm.false_alarm_rate,
                     "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn, "total": cm.total})

    print("\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)
    _print_scene_table(rows, f"Per-scene [test]  (min_src={best_ms}, ε={best_eps}px, δ={best_delta}px)")
    _print_cm(agg, "Aggregate [test]")

    # ── Persist ──────────────────────────────────────────────────────────────
    results = {
        "profile": PROFILE.name,
        "dataset": DATASET_ROOT,
        "device": _DEVICE,
        "method": "GeometricContactDetector + MobileNet-SSD",
        "homography": {
            "mode": "self_calibration_shared_track",
            "ref_camera": REF_CAMERA,
            "track_frames": n_track,
            "epsilon_px": epsilon,
            "per_camera": {str(k): v for k, v in cal_info.items()},
        },
        "sweep_val": [
            {"min_sources": ms, "epsilon_px": eps, "delta_px": d, "prec": cm.precision,
             "rec": cm.recall, "f1": cm.f1, "tp": cm.tp, "fp": cm.fp, "fn": cm.fn, "tn": cm.tn}
            for ms, eps, d, cm in sweep
        ],
        "best_min_sources": best_ms,
        "best_epsilon_px": best_eps,
        "best_delta_px": best_delta,
        "test": {
            "aggregate": {"acc": agg.accuracy, "prec": agg.precision, "rec": agg.recall,
                          "f1": agg.f1, "far": agg.false_alarm_rate,
                          "tp": agg.tp, "tn": agg.tn, "fp": agg.fp, "fn": agg.fn,
                          "total": agg.total},
            "per_scene": rows,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n  JSON results → {OUT_JSON}")


if __name__ == "__main__":
    main()
