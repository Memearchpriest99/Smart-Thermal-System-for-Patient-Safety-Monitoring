"""F1 comparison of Thermo-X3D T5v2 backends: torch / ONNX fp32 / ONNX int8.

Replicates the _retrain_x3d_T5_v2.py eval protocol exactly:
  - p25 session-background Tateno residuals, all labelled scenes
  - timeline 60/15/25 split; scored on the TEST frames at threshold 0.35
  - full replay of the held-out 2men_clash scene (tol=0 and +-2)

The confidence of frame i is softmax(logits)[1] of the T-frame window ending
at i (0.0 while the buffer fills) — identical to streaming predict() with
persistence_frames=1, which is how the reference numbers were produced.
Because each window is independent, only the windows that are actually
scored (test-split + held-out frames) are forwarded.

Phases (all resumable — cached under --cache-dir):
    python scripts/eval_x3d_backends.py build
    python scripts/eval_x3d_backends.py eval torch_fp32
    python scripts/eval_x3d_backends.py eval onnx_fp32
    python scripts/eval_x3d_backends.py eval onnx_int8
    python scripts/eval_x3d_backends.py report

torch runs on CUDA if available (fp32 CUDA-vs-CPU parity ~1e-7); the ONNX
backends run the CPU execution provider, batch 1, as they would on the Pi.

Outputs reports/x3d_backend_comparison.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CKPT_DIR = _ROOT / "checkpoints" / "thermo_x3d_detector"
TORCH_CKPT = CKPT_DIR / "Waveshare_26984_T5_v2.thalg"
ONNX_FP32 = CKPT_DIR / "Waveshare_26984_T5_v2.onnx"
ONNX_INT8 = CKPT_DIR / "Waveshare_26984_T5_v2.int8.onnx"
OUT_JSON = _ROOT / "reports" / "x3d_backend_comparison.json"
DEFAULT_CACHE = Path(__file__).resolve().parent / "_x3d_backend_cache"

HELD_OUT = "2men_clash"
BG_PCT = 25.0
THRESHOLD = 0.35
TRAIN_FRAC, VAL_FRAC = 0.60, 0.15
TORCH_BATCH = 64


# ---------------------------------------------------------------------------
# Phase 1: build residual stacks (cached per scene)
# ---------------------------------------------------------------------------

def build(cache: Path) -> None:
    import eval_waveshare_contact as geo
    from thermal_algorithms.core.types import Frame
    from thermal_algorithms.preprocessing import TatenoPipeline
    from thermal_algorithms.training import DatasetIndex

    cache.mkdir(parents=True, exist_ok=True)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    # T is needed for the min-length filter; read it from the checkpoint once.
    from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
    T = ThermoX3DDetector.load(TORCH_CKPT)._T

    for scene in sorted(labels):
        out = cache / f"{scene.replace('/', '_')}.npz"
        if out.exists():
            continue
        try:
            s = idx.find(scene)
        except Exception:
            continue
        if not all(ch in s.channels_with_data for ch in geo.CHANNELS):
            continue
        fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
        if scene != HELD_OUT and len(fis) < T + 2:
            continue
        t0 = time.time()
        pre = {}
        for c, ch in enumerate(geo.CHANNELS):
            stack = np.stack([geo.get_frame(s, ch, fi).data for fi in fis], 0)
            bg = np.percentile(stack, BG_PCT, axis=0)
            pre[c] = TatenoPipeline(geo.PROFILE).fit(
                [Frame(data=bg, timestamp=0.0, camera_id=ch)]
            )
        resid = np.stack(
            [np.stack([pre[c].predict(geo.get_frame(s, ch, fi)).data
                       for fi in fis], 0).astype(np.float32)
             for c, ch in enumerate(geo.CHANNELS)], 0)
        labs = np.array([labels[scene][fi] for fi in fis], dtype=np.int8)
        np.savez_compressed(out, resid=resid, labs=labs)
        print(f"  built {scene}: {len(fis)} frames ({time.time() - t0:.0f}s)",
              flush=True)
        # free the raw-frame cache between scenes to bound memory
        geo._FRAMES_CACHE.clear()
    print("build complete")


# ---------------------------------------------------------------------------
# Phase 2: per-backend confidences on scored frames (cached per scene)
# ---------------------------------------------------------------------------

def make_forward(backend: str, det):
    if backend == "torch_fp32":
        import torch

        model, device = det._get_model()
        model.eval()

        def forward(batch):
            with torch.no_grad():
                x = torch.from_numpy(batch).to(device)
                return torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        return forward, TORCH_BATCH

    import onnxruntime as ort

    path = ONNX_FP32 if backend == "onnx_fp32" else ONNX_INT8
    sess = ort.InferenceSession(str(path), ort.SessionOptions(),
                                providers=["CPUExecutionProvider"])

    def forward(batch):
        out = np.empty(len(batch), dtype=np.float64)
        for j, v in enumerate(batch):          # exported with batch=1
            logits = sess.run(None, {"volume": v[None]})[0][0]
            e = np.exp(logits - logits.max())
            out[j] = (e / e.sum())[1]
        return out
    return forward, 64


def eval_backend(backend: str, cache: Path) -> None:
    from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector

    det = ThermoX3DDetector.load(TORCH_CKPT)
    T = det._T
    forward, batch_size = make_forward(backend, det)

    scene_files = sorted(cache.glob("*.npz"))
    if not scene_files:
        raise SystemExit("no cached scenes — run the 'build' phase first")

    for f in scene_files:
        scene = f.stem
        out = cache / f"confs_{backend}_{scene}.npy"
        if out.exists():
            continue
        t0 = time.time()
        d = np.load(f)
        resid, labs = d["resid"], d["labs"]
        n = resid.shape[1]
        norm = det._normalise(resid).astype(np.float32)

        if scene == HELD_OUT:
            need = list(range(n))                       # full replay
        else:
            n_va = int(n * (TRAIN_FRAC + VAL_FRAC))
            need = list(range(n_va, n))                 # test split only

        confs = np.zeros(n, dtype=np.float64)
        wins = [i for i in need if i >= T - 1]
        for b in range(0, len(wins), batch_size):
            ids = wins[b:b + batch_size]
            batch = np.stack([norm[:, i - T + 1:i + 1] for i in ids], 0)
            confs[ids] = forward(batch)
        np.save(out, confs)
        print(f"  {backend} {scene}: {len(wins)} windows "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"{backend} eval complete")


# ---------------------------------------------------------------------------
# Phase 3: score + report
# ---------------------------------------------------------------------------

def _score_counts(labs, preds):
    tp = int(np.sum((preds == 1) & (labs == 1)))
    fn = int(np.sum((preds == 0) & (labs == 1)))
    fp = int(np.sum((preds == 1) & (labs == 0)))
    tn = int(np.sum((preds == 0) & (labs == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, prec=prec, rec=rec, f1=f1, far=far)


def _score_tol(labs, preds, tol):
    labs, preds = [int(x) for x in labs], [int(x) for x in preds]
    tp = fn = fp = tn = 0
    for i, (l, p) in enumerate(zip(labs, preds)):
        if tol and p != l:
            lo_, hi_ = max(0, i - tol), min(len(labs), i + tol + 1)
            if p in labs[lo_:hi_]:
                l = p
        tp += (p == 1 and l == 1); fn += (p == 0 and l == 1)
        fp += (p == 1 and l == 0); tn += (p == 0 and l == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return dict(prec=prec, rec=rec, f1=f1, far=far, tp=tp, fn=fn, fp=fp)


def report(cache: Path) -> None:
    backends = ["torch_fp32", "onnx_fp32", "onnx_int8"]
    scene_files = sorted(f for f in cache.glob("*.npz"))
    results = {}
    for backend in backends:
        yt, yp = [], []
        ho = None
        for f in scene_files:
            scene = f.stem
            conf_file = cache / f"confs_{backend}_{scene}.npy"
            if not conf_file.exists():
                print(f"WARNING: missing {conf_file.name}; run eval {backend}")
                break
            labs = np.load(f)["labs"].astype(np.int64)
            confs = np.load(conf_file)
            preds = (confs > THRESHOLD).astype(np.int64)
            if scene == HELD_OUT:
                ho = (labs, preds)
            else:
                n = len(labs)
                n_va = int(n * (TRAIN_FRAC + VAL_FRAC))
                yt.extend(labs[n_va:]); yp.extend(preds[n_va:])
        else:
            m = _score_counts(np.array(yt), np.array(yp))
            m0 = _score_tol(*ho, 0)
            m2 = _score_tol(*ho, 2)
            results[backend] = {"test": m, "held_out_tol0": m0,
                                "held_out_tol2": m2}
            print(f"\n{backend}")
            print(f"  test ({len(yt)} frames): P={m['prec']:.1%} R={m['rec']:.1%} "
                  f"F1={m['f1']:.1%} FAR={m['far']:.1%}")
            print(f"  {HELD_OUT} tol0: P={m0['prec']:.1%} R={m0['rec']:.1%} "
                  f"F1={m0['f1']:.1%}   tol±2: F1={m2['f1']:.1%}")

    OUT_JSON.write_text(json.dumps({
        "protocol": "T5_v2: p25 bg, test split @ th=0.35, held-out 2men_clash",
        "threshold": THRESHOLD,
        "reference_test_f1": 0.8777, "reference_heldout_tol0_f1": 0.337,
        "backends": results,
    }, indent=2))
    print(f"\nsaved -> {OUT_JSON.relative_to(_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "eval", "report", "all"])
    ap.add_argument("backend", nargs="?",
                    choices=["torch_fp32", "onnx_fp32", "onnx_int8"])
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()

    if args.phase == "build":
        build(args.cache_dir)
    elif args.phase == "eval":
        if not args.backend:
            raise SystemExit("eval requires a backend name")
        eval_backend(args.backend, args.cache_dir)
    elif args.phase == "report":
        report(args.cache_dir)
    else:
        build(args.cache_dir)
        for b in ["torch_fp32", "onnx_fp32", "onnx_int8"]:
            eval_backend(b, args.cache_dir)
        report(args.cache_dir)


if __name__ == "__main__":
    main()
