#!/usr/bin/env python3
"""ThermoX3D contact training on REAL data only, with augmentation + Optuna.

Supersedes the contact path of ``scripts/train_balanced_corpus.py``. Project
owner's directive (2026-08-09), after the synthetic-data experiment failed:

  * ignore the synthetic corpus entirely -- train on ``waveshare_work`` only;
  * augment instead (horizontal flip, +/-15 deg rotation, N(0, sigma) noise --
    see ``thermal_algorithms/training/augment.py``);
  * keep the training set balanced 50/50 positive/negative;
  * label a window positive when >=2 of its T=5 triplets are contact-positive
    (``ThermoX3DDetector(min_positive_frames=...)``);
  * initialise weights as identity matrices (``weight_init="identity"``);
  * tune hyperparameters with Optuna.

Why synthetic data was dropped: raw synth frames span ~1 degC (ambient ~29.9,
std ~0.11) against real data's ~23 degC (13-36, std ~2.39), leaving real data
~+7.9 sigma outside the synth training distribution. Per-source normalisation
recovered the numeric mismatch but not the missing background structure, and
the resulting model scored 41.1% F1 / 26.3% precision on the real test scenes
-- indistinguishable from the mixed-source run it replaced.

THE BINDING CONSTRAINT, stated plainly because it bounds everything below:
only TWO non-test scenes contain any contact-positive frames at all --
``3pp_surprise`` (66 positives, and flagged as block-labelled/suspect by the
label audit in reports/historical_investigations.md S5.1) and ``2men_clash``
(58 positives, freshly annotated). A leakage-free SCENE-level split therefore
puts exactly one positive scene on each side. ``2men_clash`` is held out for
validation (project owner's instruction, and it carries the more trustworthy
labels), which leaves a single positive training scene. Augmentation
multiplies frames; it cannot manufacture scenario diversity. Read every number
this script produces with that in mind.

Usage::

    python scripts/train_contact_real_optuna.py --n-trials 30
    python scripts/train_contact_real_optuna.py --n-trials 5 --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.core.checkpoints import CheckpointRegistry
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.preprocessing.global_norm_pipeline import GlobalNormPreprocessor
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.augment import AugmentParams, augment_examples
from thermal_algorithms.training.balance import (
    balance_windows,
    build_waveshare_contact_pools,
    size_and_cap_negative_runs,
)
from thermal_algorithms.training.datasets import ContactFrameDataset
from thermal_algorithms.training.metrics import binary_confusion_matrix
from thermal_algorithms.training.split import CONTACT_TEST_SCENES

DATA_ROOT = _REPO_ROOT.parent / "data"
DEFAULT_OUT = _REPO_ROOT / "checkpoints_balanced"
DEFAULT_VAL_SCENES = ["2men_clash"]
REPORTS = _REPO_ROOT / "reports"


# ---------------------------------------------------------------------------
# Data loading (done ONCE; every Optuna trial reuses these cached runs)
# ---------------------------------------------------------------------------

def _materialize(item) -> list:
    """balance.py item -> list[(triplet, ContactEvent)]."""
    _n, thunk = item
    _label, frames, events = thunk()
    return list(zip(frames, events))


def load_scene_runs(index: DatasetIndex, scene: str, *, T: int, neg_chunk: int):
    """Contiguous runs for ONE scene, as raw (un-preprocessed) example lists.

    Deliberately per-scene: ``build_waveshare_contact_pools`` finds runs by
    scanning a ContactFrameDataset's examples as one flat sequence, so handing
    it several scenes at once would let a "contiguous run" silently straddle a
    scene boundary and splice two unrelated recordings into one window.
    """
    pos_items, raw_neg = build_waveshare_contact_pools(index, {scene}, T=T)
    neg_items = size_and_cap_negative_runs(
        raw_neg, pos_items, T=T,
        target_negative_frames=10 ** 9,     # no volume cap here; balancing happens later
        max_run_frames=neg_chunk,
    )
    return [_materialize(i) for i in pos_items], [_materialize(i) for i in neg_items]


def load_all(index: DatasetIndex, *, T: int, neg_chunk: int, val_scenes: set[str], verbose=True):
    all_scenes = {s.scene for s in ContactFrameDataset(index).sessions}
    usable = sorted(all_scenes - set(CONTACT_TEST_SCENES))
    train_scenes = [s for s in usable if s not in val_scenes]
    val_list = [s for s in usable if s in val_scenes]

    data = {"train": {"pos": [], "neg": []}, "val": {"pos": [], "neg": []}}
    for split, scenes in (("train", train_scenes), ("val", val_list)):
        for sc in scenes:
            p, n = load_scene_runs(index, sc, T=T, neg_chunk=neg_chunk)
            data[split]["pos"] += p
            data[split]["neg"] += n
            if verbose and p:
                print(f"    {split:<5} {sc:<24} {len(p)} positive run(s), {len(n)} negative run(s)")
    if verbose:
        for split in ("train", "val"):
            nf = lambda rs: sum(len(r) for r in rs)  # noqa: E731
            print(f"  {split}: {len(data[split]['pos'])} pos runs ({nf(data[split]['pos'])} frames), "
                  f"{len(data[split]['neg'])} neg runs ({nf(data[split]['neg'])} frames)")
    return data, train_scenes, val_list


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def preprocess_run(run, preprocessor):
    return [(tuple(preprocessor.predict(f) for f in trip), ev) for trip, ev in run]


def windows_from_runs(det, runs, preprocessor, *, aug: AugmentParams | None = None,
                      n_copies: int = 0, seed: int = 0):
    """Build windows per run (never across runs, so no cross-clip 'seam'
    windows), optionally adding ``n_copies`` augmented variants of each run.
    Augmentation is applied to RAW frames, before preprocessing, so
    ``noise_sigma`` stays interpretable in degrees C."""
    out = []
    rng = np.random.default_rng(seed)
    for run in runs:
        out += det._build_training_windows(preprocess_run(run, preprocessor))
        if aug is not None and n_copies > 0:
            for _ in range(n_copies):
                out += det._build_training_windows(
                    preprocess_run(augment_examples(run, aug, rng=rng), preprocessor)
                )
    return out


def _norm_stats(runs, preprocessor, max_runs: int = 40):
    vals = []
    for run in runs[:max_runs]:
        for trip, _ev in run:
            for f in trip:
                vals.append(preprocessor.predict(f).data.ravel())
    if not vals:
        return 0.0, 1.0
    arr = np.concatenate(vals).astype(np.float32)
    return float(arr.mean()), float(arr.std()) + 1e-6


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def window_probs(det, windows, batch_size: int = 32):
    """P(contact) for each window, batched, no grad."""
    import torch
    import torch.nn.functional as F
    model, device = det._get_model()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            vol, _lbl = det._collate(windows[i:i + batch_size])
            logits = model(vol.to(device))
            out.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def best_f1(y_true, probs, *, min_precision: float = 0.0):
    """Sweep the decision threshold; return (threshold, confusion). Restricted
    to grid points clearing ``min_precision`` when any do -- same
    precision-floor convention as calibrate_otsu_thresholds.py."""
    grid = []
    for th in [round(0.05 * k, 2) for k in range(1, 20)]:
        cm = binary_confusion_matrix(list(y_true), [1 if p > th else 0 for p in probs])
        grid.append((th, cm))
    ok = [g for g in grid if g[1].precision >= min_precision]
    return max(ok or grid, key=lambda g: g[1].f1)


# ---------------------------------------------------------------------------
# One training run (shared by every Optuna trial and by the final refit)
# ---------------------------------------------------------------------------

def run_training(params: dict, data, preprocessor, norm, val_windows, *,
                 seed: int, trial=None, verbose: bool = False):
    """Train one detector under ``params``; return (detector, val_f1, threshold, cm)."""
    import optuna

    det = ThermoX3DDetector(
        sensor_profile=WAVESHARE_26984,
        T=params["T"],
        n_epochs=1,                       # epochs are driven here, for pruning
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        min_positive_frames=params["min_positive_frames"],
        weight_init=params["weight_init"],
        random_state=seed,
    )
    det.set_normalization(*norm)

    aug = AugmentParams(
        flip_prob=params["flip_prob"], rotate_prob=params["rotate_prob"],
        max_rotation_deg=params["max_rotation_deg"], noise_sigma=params["noise_sigma"],
    )
    raw = windows_from_runs(det, data["train"]["pos"] + data["train"]["neg"], preprocessor,
                            aug=aug, n_copies=params["n_aug_copies"], seed=seed)
    train_windows = balance_windows(raw, seed=seed)
    n_pos = sum(1 for w in train_windows if w[1] == 1)
    if not train_windows or n_pos == 0 or n_pos * 2 != len(train_windows):
        raise RuntimeError(
            f"training windows could not be balanced 50/50 "
            f"({n_pos} pos / {len(raw) - sum(1 for w in raw if w[1]==1)} neg available)"
        )
    if verbose:
        print(f"    train windows: {len(raw)} raw -> {len(train_windows)} balanced "
              f"(pos={n_pos}, neg={len(train_windows) - n_pos})")

    y_val = [w[1] for w in val_windows]
    best = (0.0, 0.5, None)
    for epoch in range(params["n_epochs"]):
        det._fit_windows(train_windows)
        th, cm = best_f1(y_val, window_probs(det, val_windows, params["batch_size"]))
        if cm.f1 > best[0]:
            best = (cm.f1, th, cm)
        if verbose:
            print(f"    epoch {epoch+1}/{params['n_epochs']}: train_loss="
                  f"{det._last_fit_loss:.4f} val_F1={cm.f1:.3f} (best {best[0]:.3f})", flush=True)
        if trial is not None:
            trial.report(cm.f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return det, best[0], best[1], best[2]


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--time-budget-hours", type=float, default=3.0)
    ap.add_argument("--val-scenes", nargs="*", default=DEFAULT_VAL_SCENES,
                    help="Scenes held out for validation. Default 2men_clash -- the only "
                         "non-test positive scene besides 3pp_surprise, and the one with "
                         "freshly-verified labels.")
    ap.add_argument("--neg-chunk", type=int, default=60,
                    help="Frames per negative run. Small chunks give many independent "
                         "negative runs, which matters because every augmented copy is drawn "
                         "per-run.")
    ap.add_argument("--min-precision", type=float, default=0.0,
                    help="Precision floor for the FINAL threshold calibration (the Optuna "
                         "objective itself is unconstrained best-F1).")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny search space + 1 epoch, for pipeline validation only.")
    ap.add_argument("--results-json", default=str(REPORTS / "contact_optuna_study.json"))
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data_root = Path(args.data_root)
    index = DatasetIndex(data_root / "waveshare_work", sensor_profile=WAVESHARE_26984, fps=8.0)
    preprocessor = GlobalNormPreprocessor(WAVESHARE_26984)
    preprocessor.fit([])

    T = 5
    print("=" * 78)
    print("  ThermoX3D contact training -- REAL data only, augmented, Optuna-tuned")
    print("=" * 78)
    print(f"  test scenes (never touched here): {sorted(CONTACT_TEST_SCENES)}")
    print("  loading real runs (cached once; every trial reuses them) ...", flush=True)
    t0 = time.time()
    data, train_scenes, val_list = load_all(
        index, T=T, neg_chunk=args.neg_chunk, val_scenes=set(args.val_scenes),
    )
    print(f"  loaded in {time.time() - t0:.0f}s | val scenes: {val_list}")

    # Normalisation from TRAIN only (val/test stats would be leakage), computed
    # on UN-augmented frames and frozen for every trial so validation windows
    # can be built once and reused.
    norm = _norm_stats(data["train"]["pos"] + data["train"]["neg"], preprocessor)
    print(f"  frozen normalisation (train only): mean={norm[0]:.4f} std={norm[1]:.4f}")

    probe = ThermoX3DDetector(sensor_profile=WAVESHARE_26984, T=T, n_epochs=1,
                              min_positive_frames=2)
    probe.set_normalization(*norm)
    val_windows_raw = windows_from_runs(probe, data["val"]["pos"] + data["val"]["neg"],
                                        preprocessor)
    val_windows = balance_windows(val_windows_raw, seed=args.seed)
    n_vp = sum(1 for w in val_windows if w[1] == 1)
    print(f"  validation windows: {len(val_windows_raw)} raw -> {len(val_windows)} balanced "
          f"(pos={n_vp}, neg={len(val_windows) - n_vp})")
    if not val_windows or n_vp == 0:
        raise SystemExit("validation set has no positive windows -- check --val-scenes.")

    # ---- Optuna study ----------------------------------------------------
    def objective(trial):
        if args.smoke:
            p = dict(T=T, min_positive_frames=2, n_epochs=1, batch_size=8,
                     learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
                     weight_decay=1e-4, weight_init="identity",
                     flip_prob=0.5, rotate_prob=0.5, max_rotation_deg=15.0,
                     noise_sigma=2.0, n_aug_copies=1)
        else:
            p = dict(
                T=T,
                min_positive_frames=2,          # fixed: the ">=2 of 5" directive
                max_rotation_deg=15.0,          # fixed: the "+/-15 deg" directive
                n_epochs=trial.suggest_int("n_epochs", 3, 15),
                batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
                learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                # Identity init is the directive; "default" is searched alongside
                # it purely so we can REPORT whether it actually helps here
                # rather than assert it (see _apply_identity_init's caveats).
                weight_init=trial.suggest_categorical("weight_init", ["identity", "default"]),
                flip_prob=trial.suggest_float("flip_prob", 0.0, 1.0),
                rotate_prob=trial.suggest_float("rotate_prob", 0.0, 1.0),
                # sigma=2 degC is the directive; the range brackets it so the
                # search can show whether it is too aggressive for this corpus.
                noise_sigma=trial.suggest_float("noise_sigma", 0.0, 3.0),
                n_aug_copies=trial.suggest_int("n_aug_copies", 1, 6),
            )
        _det, f1, _th, _cm = run_training(p, data, preprocessor, norm, val_windows,
                                          seed=args.seed, trial=trial)
        trial.set_user_attr("params_full", p)
        return f1

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    print(f"\n  running Optuna: {args.n_trials} trials, "
          f"{args.time_budget_hours}h budget ...", flush=True)
    t0 = time.time()
    done = {"n": 0}

    def _cb(st, tr):
        done["n"] += 1
        mark = "pruned" if tr.state.name == "PRUNED" else f"F1={tr.value:.3f}" if tr.value is not None else tr.state.name
        best = st.best_value if st.best_trial else float("nan")
        print(f"    trial {done['n']}/{args.n_trials}: {mark}  (best {best:.3f}, "
              f"{time.time() - t0:.0f}s)", flush=True)

    study.optimize(objective, n_trials=args.n_trials,
                   timeout=args.time_budget_hours * 3600, callbacks=[_cb],
                   catch=(RuntimeError,))

    print(f"\n  best val F1 = {study.best_value:.4f}")
    print(f"  best params = {json.dumps(study.best_params, indent=2)}")

    # ---- Refit the best configuration and save ---------------------------
    best_p = study.best_trial.user_attrs["params_full"]
    print("\n  refitting best configuration ...", flush=True)
    det, val_f1, val_th, val_cm = run_training(best_p, data, preprocessor, norm, val_windows,
                                               seed=args.seed, verbose=True)

    th, cm = best_f1([w[1] for w in val_windows], window_probs(det, val_windows),
                     min_precision=args.min_precision)
    det._conf_threshold = th
    det._persistence_frames = 1
    det.set_params(conf_threshold=th, persistence_frames=1)
    print(f"  calibrated: conf_threshold={th} -> val F1={cm.f1:.3f} "
          f"prec={cm.precision:.3f} rec={cm.recall:.3f}")

    registry = CheckpointRegistry(root=args.out)
    path = registry.register(det)
    print(f"  saved -> {path}")

    payload = {
        "best_val_f1": study.best_value,
        "best_params": study.best_params,
        "best_params_full": {k: v for k, v in best_p.items()},
        "final_threshold": th,
        "final_val": {"f1": cm.f1, "precision": cm.precision, "recall": cm.recall,
                      "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn},
        "train_scenes": train_scenes,
        "val_scenes": val_list,
        "test_scenes": sorted(CONTACT_TEST_SCENES),
        "normalisation": {"mean": norm[0], "std": norm[1]},
        "n_trials_run": len(study.trials),
        "n_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
        "checkpoint": str(path),
        "trials": [
            {"number": t.number, "value": t.value, "state": t.state.name, "params": t.params}
            for t in study.trials
        ],
    }
    Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_json).write_text(json.dumps(payload, indent=2, default=float))
    print(f"  study results -> {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
