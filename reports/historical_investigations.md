# Historical Investigations & Findings

*Smart Thermal System for Patient Safety Monitoring — Beer Sheva Mental Health Center*

This document consolidates the substantive, non-obvious findings from eight standalone LaTeX
evaluation reports written between 2026-06-19 and 2026-07-05 (`checkpoint_report.tex`,
`fire_detection_report.tex`, `human_detection_report.tex`, `waveshare_contact_report.tex`,
`waveshare_detection_report.tex`, `touch_detection_detailed.tex`, `touch_detection_summary.tex`,
`contact_label_audit_report.tex` — deleted 2026-08-09 as part of a repo cleanup). Those reports
predate the current consolidated training/evaluation pipeline (`scripts/generate_full_report.py` →
`reports/Report_smoketest.pdf`, active since 2026-08-07) and used a different protocol in places
(different train/val/test splits, an earlier "T5v2" Thermo-X3D checkpoint, self-calibrated
homography) — their headline numbers are **not directly comparable** to the current report's
tables. What's preserved here is the *investigative content*: root causes, negative results, and
recommendations that remain relevant regardless of which specific numbers they were measured
against. Mathematical/architectural derivations shared with these reports already live in
`reports/algorithm_derivations.md`, so are not repeated here.

---

## 1. Fire detection

### 1.1 The hot-pixel bypass fix (historical — already shipped)

Two root causes were found for `OtsuFireDetector`'s original low recall:

1. A single erosion pass (3×3 kernel) in the morphological post-processing step deletes
   single-pixel fires outright — a 72.5°C pixel in one test scene was erased before
   classification ever saw it.
2. With people in frame, Otsu's global threshold anchors near ~25°C because a person's ~39°C
   body dominates as "the hot blob" — the much smaller, much hotter fire pixel is never isolated
   by a single frame-wide threshold.

Fix (now part of the shipped pipeline): direct thresholding at `t_ign` via connected components
(counting by pixel area, not blob bounding-box area) plus a despiking filter for dead pixels
reading 800–935°C. **Caveat:** the `t_ign=45°C` value this old report evaluated against is stale —
`t_ign` was later data-calibrated to 41.5°C (2026-08-08, `scripts/calibrate_otsu_thresholds.py`,
see `CLAUDE.md`'s Key Open Items). The architectural fix is current; the specific recall/F1 numbers
measured against the old threshold are not.

### 1.2 The recall ceiling is a labels/physics mismatch, not an algorithm defect

On the original MLX90640-era fire dataset, the median peak temperature across all 78 fire-labeled
frames was only **36.5°C** (indistinguishable from a human body); only 21/78 (26.9%) exceeded 45°C,
and only 9/78 (11.5%) exceeded 60°C. **73% of fire-labeled frames carried no above-threshold
thermal signature at all** — capping any absolute-temperature detector at ≈27% recall regardless
of algorithm. This is a general lesson (annotators box the ignition-source *object*, e.g. a
cigarette, whenever visible — not only when it's thermally glowing) that reappears in §1.3 below on
the current Waveshare corpus.

### 1.3 The recall ceiling on cigarette scenes reproduces on the current (better) sensor

Waveshare's higher resolution/sensitivity raised aggregate recall substantially (Otsu 9%→56.9%,
Fire-SVM 32%→75.5% vs. the old MLX90640 numbers), but the same physics-not-algorithm ceiling
persists per-scene, tracking measured peak temperature inside the annotated fire box directly:

| Scene | Median peak temp | % frames >45°C | Otsu recall | Fire-SVM recall |
|---|---|---|---|---|
| `man_light_cig_1` | 63.4°C | 100% | 100% | 100% |
| `heater_in_middle` | 54.5°C | 77% | 76.8% | 92.9% |
| `man_light_cig` | 42.7°C | 33% | 32.6% | 50.0% |
| `1person cig` | 39.9°C | 26% | 26.4% | 50.0% |

Recall tracks the ">45°C" column almost exactly — the annotator boxes the cigarette whenever
visible, not only when its tip is glowing hot enough to trip a thermal threshold.

---

## 2. Human detection

### 2.1 Two-criterion framework: presence vs. localization

Human detection was scored under two separate criteria — **A** (frame-level presence, no IoU
requirement) and **B** (requires IoU > 0.5 against the ground-truth box) — because a detector can
correctly report "a person is here" while still localizing badly. The gap between the two exposes
localization quality a single blended metric would hide:

| Detector | Criterion A F1 | Criterion B F1 | Drop |
|---|---|---|---|
| MobileNet-SSD | 98.8% | 94.9% | −3.9pp |
| Adaptive Threshold | 98.5% | 82.0% | −16.5pp |
| HOG+SVM | 97.8% | 57.4% | −40.4pp |

HOG+SVM's fixed 14×6px window structurally caps achievable IoU regardless of SVM confidence
(detection ≠ localization); Adaptive Threshold's morphological-closing bounding box is loose enough
to miss IoU 0.5 even when the underlying blob is correctly found (185 of its false negatives were
"detected but misplaced," not missed).

### 2.2 The pyramid-scale ablation: recall/false-alarm tradeoff

Adding a 4-scale pyramid to HOG+SVM raised localization recall from 40.5% to 61.8% (F1 57.4%→75.5%,
+18.1pp) — but false positives rose 3→12 and true negatives on the empty-room test split collapsed
10→1, i.e. a **92% false-alarm rate** on empty frames. Multi-scale search finds more real people at
a real, non-trivial false-alarm cost — a genuine tradeoff, not a free win.

### 2.3 Combined aggregate results (test split, Criterion B, IoU≥0.5)

| Detector | Acc | Prec | Rec | F1 | Mean IoU | False-alarm rate |
|---|---|---|---|---|---|---|
| Adaptive Threshold | 76.6% | 93.4% | 80.7% | 86.6% | 56.5% | 83.0% |
| HOG+SVM | 79.8% | 92.7% | 85.1% | 88.7% | 58.5% | 98.1% |
| MobileNet-SSD | 95.8% | 99.1% | 96.4% | 97.7% | 67.3% | 13.2% |

MobileNet-SSD was the production choice: 100% F1 on every crowded scene tested
(`3ppl_dance`, `3pp_surprise`, `3pplhedroncolider`, `2ppl_fight`). The shared hard case across all
three detectors was `1_man_run` — a mostly-empty-frame scene with residual motion heat, where even
MobileNet-SSD drops to 40% recall and the classical detectors fall below 22% accuracy.

---

## 3. Contact detection — the three "official" detectors (Geometric / MV-STGCN / Thermo-X3D)

### 3.1 Two failure modes, and why Thermo-X3D escapes both

**Failure mode 1 — homography noise.** With strict cross-camera validation (≥2 cameras must agree
within ε), the mean number of actors surviving per true-contact frame was only **0.16** — cameras
almost never agree at psychiatric-ward-relevant distances. Relaxing the agreement rule creates
phantom duplicate actors instead, which is why MV-STGCN false-alarms heavily on non-contact crowds
(64 FP on `1person cig`, 30 FP on `2ppl_walk`, 34 FP on `3ppl_dance`) — 82.9% recall bought at a
29.8% false-alarm rate. The Geometric detector is more conservative (4.6% overall FAR) but only
26.8% recall as a result.

**Failure mode 2 — blob merge at contact.** Touching bodies merge into one thermal blob in the
underlying person detector, so any box-based method has no second box left to measure a distance
between — this is intrinsic to any approach built on bounding boxes, not a tuning problem.

Thermo-X3D uses neither boxes nor homography (it's a pixel-based volumetric network), so it's
immune to both failure modes: FAR 2.4% vs. Geometric's 4.6% and MV-STGCN's 29.8%, with false alarms
confined to genuine multi-person scenes rather than lone/walking individuals.

### 3.2 Recall is data-limited, not method-limited

Only 188 positive contact frames existed in total at the time of this investigation (~90 usable
training windows). Thermo-X3D learned one scene (`3pp_surprise`) perfectly (100% recall) but missed
another (`2ppl_fight`) in test purely because too few fight frames existed to train on — not a model
capacity ceiling. The per-session (not held-out-scene) timeline split used at the time let train and
test share scenes, so the reported 48.4% precision figure was explicitly flagged as "a feasibility
signal, not a deployment number." **This same data-scarcity conclusion was independently
reconfirmed in 2026-08-09's ThermoX3D training-loop bug fix** (see memory: `thermox3d-training-bug`)
— after fixing an unrelated chunked-training bug, recall went from 0%→100% but precision stayed
poor (26.6%) because the real corpus (still only ~188 positive frames) is too small to calibrate a
reliable operating threshold. Two independent investigations, roughly a month apart, arrived at the
same root cause.

### 3.3 Aggregate results at the time (common 695-frame test set, 41 positive / 654 negative)

| Detector | Acc | Prec | Rec | F1 | FAR |
|---|---|---|---|---|---|
| Geometric | 91.4% | 26.8% | 26.8% | 26.8% | 4.6% |
| MV-STGCN | 70.9% | 14.8% | 82.9% | 25.2% | 29.8% |
| Thermo-X3D | 94.0% | 48.4% | 36.6% | 41.7% | 2.4% |

Self-calibrated homography (no known floor coordinates, derived from a 146-frame calibration walk)
had a RANSAC inlier ratio of only ~25–30% — foreshanding the homography-noise failure mode in §3.1.

### 3.4 Conclusion at the time

Thermo-X3D was recommended for deployment (best F1, best FAR); MobileNet-SSD-based human presence
was judged necessary but not sufficient on its own; MV-STGCN was flagged as recall-strong/
precision-poor, needing a real metric homography (not self-calibrated) to become viable; the
limiting resource was explicitly identified as labeled contact data — prioritized over further
algorithmic tuning. (§4 below found a second, independent reason MV-STGCN specifically remained
poor even after that.)

---

## 4. The image-plane voting family (V1–V10) — a parallel, homography-free investigation

A separate line of investigation explored whether a purely image-plane, homography-free voting
scheme could out-perform the three "official" detectors above. This was NOT continued into any
current pipeline, but its negative/positive results are informative and non-obvious:

- **V1–V3 (baseline):** each camera's person detector votes "touch" if two boxes have an edge-gap
  below a threshold τ. V1 = majority vote (≥2/3 cameras), V2 = unanimous (3/3), V3 = relaxed
  (≥2 votes, third camera corroborates but can't veto). **V3 was best (F1 28.6%)**; V2 (unanimous)
  was *worst* (F1 23.0%) — strict agreement is rare enough that tuning has to loosen τ to
  compensate, which floods false positives instead.
- **V3.1–V3.5 (ablations off the V3.0 baseline):** merge-aware quorum pushed recall to 92.7% but
  cost too much precision (F1 28.3%); temporal persistence alone hurt (F1 25.9%); size-normalized
  gap helped marginally (F1 30.2%); **merging over-segmented boxes (V3.4) was the one "free win"**
  (F1 32.0%, +3.4 over baseline); OR-ing with Thermo-X3D degenerated to Thermo-X3D alone once tuned.
- **V4/V5 (compound):** box-merge + merge-aware quorum + people-count routing (trust boxes at
  ≤2 people, defer to Thermo-X3D at ≥3). **F1 35.5%.**
- **V6 (thermal signal) — the breakthrough:** testing four thermal cues separately, treating two
  people who merge into one warm connected component as itself the "contact" signal
  (**blob-merge, V6.3**) reached **F1 44.6%** — the first homography-free method to beat Thermo-X3D
  (41.7% at the time). It fixed both remaining error categories at once (a cigarette-scene false
  positive count dropped 23→4; a dance-scene false-positive count dropped to 0). Notably, a
  **learned logistic combiner (V6.4) underperformed the simple hand-written rule** (F1 35.1% vs.
  44.6%) — the first of several signs that learning loses to a simple rule on this much data.
- **V7 (blob-merge on crowds):** extending blob-merge beyond the ≤2-person regime held F1 at 44.6%
  but recovered a specific scene's recall from 0%→86% (`3pplhedroncolider`), offset by new false
  positives in that same scene's negative frames.
- **V8 (temporal):** morphological open+close smoothing over the per-frame prediction stream (T1)
  reached **F1 53.8%** (+9 points) — again beating a learned temporal alternative (a logistic
  regression over the stream, T3, reached only F1 34.9%).
- **V9 (two-body merge + temporal):** added a spatial discriminator (fire only if a warm component
  contains *exactly two* person-centers, rejecting 3+-body clusters), then T1 smoothing. **F1 55.0%**
  (precision 56.4%, recall 53.7%, FAR 2.6%).
- **Part C — preprocessing ablation (key finding):** tested raw vs. Tateno-residual preprocessing
  *separately* for the person detector vs. the blob-merge segmentation. Both-raw collapsed to F1
  0.0%; both-residual reached F1 54.5%; but **raw-for-detector + residual-for-blob reached F1
  60.8%** — the best of all four combinations. **The two sub-tasks want opposite preprocessing**:
  blob segmentation needs the residual (bodies barely exceed background on raw frames), but the
  residual actively *hurts* the person detector (it removes the body's own thermal signature that
  the detector needs). This defines **"config-D"**: raw-input person detector + residual-input
  blob-merge + two-body merge + temporal smoothing, with no homography and no contact-specific
  training labels at all.

### 4.1 Part D — "Reaching 65%? The honest reality check"

- **Joint re-tuning overfits.** Jointly re-tuning ~6 knobs (V10) found a validation F1 of 55.4% that
  *dropped* to 54.4% on the test split — below config-D's already-tuned 60.8%. With only 40
  validation-positive frames, tuning 6 knobs at once fits noise, not signal.
- **Cross-validation reveals the true number.** A proper 4-fold timeline cross-validation over the
  full 1108-frame val+test pool (81 positives) gave per-fold F1 of 32% / 51% / 65% / 0% — mean
  36.9% ± 24.3%, **pooled F1 44.3%**. This is the honest number; the 60.8% single-split figure was
  inflated by which specific split happened to be drawn.
- **How much data would 65% actually need?** A bootstrap 95% CI on the frame-level split was
  [35.7%, 52.4%]; the scene-level CI (the real binding constraint, since only a handful of distinct
  contact scenes exist) was [12.2%, 66.3%]. Distinguishing a genuine 65% from a genuine 60%
  (±2.5% precision) would need roughly **900 positive frames (~11× what existed) and 3–4× more
  distinct contact scenes** — not more tuning.
- **Can old MLX90640 touch data substitute?** No — pretraining Thermo-X3D on upsampled 24×32
  MLX90640 data before fine-tuning on Waveshare *reduced* F1 by 6.2 points (35.9%→29.7%): the lower
  native resolution can't preserve the two-body structure that distinguishes contact from proximity,
  and the sensor-domain gap itself caused over-firing. Synthetic multi-view compositing was judged
  separately unviable (it would need multi-view geometric consistency this project doesn't have).

### 4.2 Conclusions from this investigation

1. Connected-warm-component ("blob") merge is the single most informative signal discovered for
   homography-free contact detection.
2. Preprocessing choice should differ *by sub-task* even within one detector (raw for detection,
   residual for segmentation) — a non-obvious result that would be easy to miss by tuning one
   preprocessing choice for the whole pipeline.
3. Temporal smoothing is a reliable, cheap ~9-F1-point win.
4. Every learned addition tried (V3.5, V6.4, T3, V10's joint tuning) matched or underperformed a
   simple hand-written rule — a robust negative result given how little labeled data exists.
5. The system was, and (per §3.2's 2026-08-09 reconfirmation) remains, data-limited rather than
   algorithm-limited: honest cross-validated F1 is ≈44% with a [12%, 66%] confidence interval on
   only ~4 distinct contact scenes.
6. Recommendation at the time: adopt config-D as a practical, explainable, homography-free
   detector; prioritize collecting substantially more contact recordings across more distinct
   scenarios over further algorithmic tuning — the same recommendation independently reached in
   §3.2/§5 below.

---

## 5. The label-quality audit and the MV-STGCN checkpoint-persistence bug

A later, follow-up investigation (2026-07-02) found that a chunk of the poor contact-detection
numbers above were a **labeling** problem, not a detection problem.

### 5.1 Three defective videos

- `the_more_the_merrier` (1050 frames, all labeled `touch=0`): **never actually annotated** —
  responsible for 81 false positives (36% of all false positives measured).
- `3ppl_dance` (399 frames, all labeled `touch=0`): also never annotated — 25 false positives (11%).
- `3pp_surprise` (207 frames, 198 labeled `touch=1`, 96% positive): block-labeled for the whole clip
  including frame 0 — responsible for 48 false negatives (63% of all false negatives measured).

No session's label sheet had ever been marked "reviewed" under the project's own green-tab
convention. **As of that report, this was diagnosed and excluded from clean-subset scoring, but not
yet fixed** (the ~550 frames were not re-annotated) — worth checking whether this remains true of
the current `data/waveshare_work/contact_labels.csv` before trusting any of these specific scenes'
per-frame contact labels.

Excluding these three videos and applying a ±2-frame tolerance for ordinary annotation imprecision
(66 of 298 measured errors sat within ±2 frames of a label transition) corrected config-D's F1 from
42.9% to **76.9%** (precision 63.3%, recall 97.8%). Two genuine (non-fixable-by-relabeling) failure
modes remained: hug-scenario blob-merge (two people merging into one box in all three cameras
simultaneously) and proximity-vs-touch ambiguity at native 80×62 resolution.

### 5.2 The MV-STGCN checkpoint-persistence bug

**Symptom:** MV-STGCN produced zero real predictions in this benchmark — flat confidence ≈0.03 on
every frame, even on obvious contact — despite the *same model* correctly showing ~30% false-alarm
rate when evaluated in-memory immediately after training (i.e. before any save/load round-trip).

**Root cause:** `ContactDetector.__init__` stored its `homography` argument on `self._homography`,
*outside* the base class's `_params` dict that `save()` actually persists (the exact same class of
bug — a constructor-time value that doesn't survive save/load because it lives outside the
persisted-params mechanism — later independently rediscovered and fixed for ThermoX3D's recalibrated
threshold on 2026-08-09; see memory `thermox3d-training-bug`). `MVSTGCNDetector._state_dict()` saved
only network weights, so every saved checkpoint reloaded with `homography=None` → the multi-view
fusion front-end produced zero actors on every frame → an empty graph → a constant network output.
**The learned weights were always correct; only the front-end was dead after save/load.** Existing
tests missed this because they called `predict()` with empty detections and no homography anyway, so
a dead front-end was indistinguishable from expected behavior.

**Fix applied at the time:** `_state_dict()`/`_load_state_dict()` extended to persist/restore the
homography alongside the weights, plus a new regression test
(`test_homography_survives_save_load`) to lock this in. Both existing checkpoints were repaired in
place (homography re-injected, re-saved) — no retraining was needed, and the repaired checkpoint's
behavior exactly matched the original in-memory evaluation. (`GeometricContactDetector` already
persisted its homography correctly; Thermo-X3D doesn't use a homography at all, so neither was
affected by this specific bug.)

### 5.3 Why MV-STGCN is structurally unsuited, independent of the bug

Even measured correctly (post-fix), MV-STGCN was the worst of the three official contact detectors
(pooled FAR 25.5%), for reasons independent of the checkpoint bug:

1. It inherits the Geometric detector's own representation (person detector → foot-point →
   homography → fusion) — the same pipeline whose homography accuracy (≈±15px) is nearly as large
   as the contact decision threshold δ (≈18px): measurement noise nearly as big as the signal.
2. Blob-merge (§3.1) deletes the defining evidence exactly when it matters: when two people
   actually touch, the person detector returns one box, so the two graph nodes it needs collapse
   into one — the model can only ever learn *approach* dynamics (proximity), never the geometry of
   contact itself.
3. It consequently fires on generic proximity — 60 false positives on one cigarette scene's test
   region alone, a ~25% false-alarm rate against a target on the order of ~2% for a psychiatric-ward
   alarm system.
4. No realistic amount of additional data fixes this — it would need orders of magnitude more
   positive training windows than the ~64–105 available at the time.

**This independently corroborates, with much deeper technical grounding, the 2026-08-08 project-owner
decision (unrelated session) to leave MV-STGCN out of the balanced-retraining pass as "useless."**
The recommendation from this investigation was explicit: drop MV-STGCN from the candidate set, but
keep the checkpoint-persistence fix (it's a correctness fix to the shared checkpoint format,
independent of whether MV-STGCN itself ships) and keep this architecture analysis on record as a
documented negative result rather than silently discarding the model.

### 5.4 Clean-label benchmark and a retraining lesson

Benchmarking all detector families on 14 videos / 2201 frames / 122 positive (clean labels only):
Thermo-X3D improved from 52.5%→78.5% F1 (±2-frame tolerance, FAR 1.5%) — the biggest beneficiary of
clean labels, and the only method that handled a hug scenario at all (54.5% F1 there specifically).
Config-D reached F1 56.6%/76.9% with the best recall (97.8%). The Geometric detector collapsed to
F1=7.7% — its earlier ~27% figure on noisier labels was found to be partly luck.

**However, retraining the learned models on the clean-label subset made things *worse*, not
better.** Excluding the three defective videos cut positive training windows from ~105 to 64;
Thermo-X3D's clean-retrained F1 dropped from 78.5% to 62.6% (pooled, ±2-frame tolerance), and an
honest (non-leaked) test split put it at only ≈21.4%. **Conclusion: at this project's data scale,
label *quantity* currently outweighs label *purity*** — a real, disclosed tension between "fix the
labels" and "don't shrink the already-scarce positive pool," worth remembering before re-annotating
the three defective videos identified in §5.1.

### 5.5 Recommended actions (at the time)

Re-annotate the three defective videos (identified as the single highest-leverage fix available);
lock in config-D (for recall) alongside Thermo-X3D (for precision) as the operating pair pending
that re-annotation; drop MV-STGCN and the Geometric detector from further candidate consideration
while keeping their negative results on record; institute an actual label-review process (no
session had ever been marked reviewed); continue prioritizing more contact recordings across more
distinct scenarios over further algorithmic tuning.

---

## 6. Deployed checkpoints and runtime (live demo, `demo-linux` branch)

*Historical snapshot as of `checkpoint_report.tex`, 2026-07-05 — the specific checkpoint filenames
below (e.g. the `T5_v2` Thermo-X3D variant) predate this project's later T=16→T=5 architecture fix
and balanced retrain (2026-08-09); check `checkpoints_balanced/`/`checkpoints_full_corpus/` for what
is actually current before assuming any filename below still exists.*

### 6.1 What was deployed and why

- **Fire:** `fire_svm_detector/_default.thalg` (sklearn pickle, 64 KB; RBF-SVC, C=1, γ=scale, 874
  support vectors).
- **Person:** `mobilenet_ssd_detector/Waveshare_26984_raw.onnx` (0.27 MB) — the **raw-input**
  variant specifically: a preprocessing ablation showed the SSD scores higher on raw frames and
  transfers better across rooms; background subtraction measurably hurt it.
- **Contact:** `thermo_x3d_detector/Waveshare_26984_T5_v2.ftz.onnx` (1.6 MB, denormal-flushed).
  Total deployed footprint ≈2 MB, torch-free via `onnxruntime`.
- **Deployment caveat (important, and consistent with §3.2/§5's later data-scarcity findings):** the
  original decision to deploy this specific Thermo-X3D checkpoint rested on an 87.8% in-domain F1
  that turned out to be a training/test scene-overlap artifact, not a real result. Honest
  assessment at the time: 2.8% recall on genuine home-contact scenarios, and a 54% false-alarm rate
  cross-room. The classical config-D rule (§4) was judged steadier where it mattered (44–54% F1)
  despite having room-tuned constants that conflict with the project's "no per-room calibration"
  goal. **Recommended interim posture: run config-D and Thermo-X3D in parallel, alarm if either
  fires.**

### 6.2 Runtime (PC benchmark, with Raspberry Pi 5 estimates)

| Stage | PC | Pi5 (estimated, unconfirmed on-device) |
|---|---|---|
| Tateno preprocessing | 0.02 ms | ~0.1 ms |
| FireSVM | 1.19 ms | 3.5–5 ms |
| MobileNet-SSD | 2.6 ms (1.8 ms @4 threads) | 8–10 ms |
| Thermo-X3D (deployed: ONNX fp32, denormals flushed) | 12.9 ms | 45–60 ms |
| **Full pipeline tick** | **~24 ms** | **~65–90 ms (est.)** |

Runtime budget at 8 Hz is 125 ms/tick — both PC and the Pi5 estimate fit comfortably, though the Pi
number was never confirmed on real hardware (PC-to-Pi scaling used a ~3–3.5× CPU-clock ratio
estimate, Cortex-A76 2.4 GHz vs. the dev PC's ~4.9 GHz).

**Thermo-X3D-specific optimization path** (torch fp32 → deployed): 93.7 ms (torch) → 34.5 ms (ONNX
export, bit-identical F1, 2.7× speedup) → **12.9 ms (ONNX + denormal-weight flushing, another 2.7×,
total 7.3×)**. The denormal-flushing win was caused by 19.3% of the trained weights being subnormal
(~1e-41, a weight-decay artifact) — triggering CPU microcode penalties on every multiply; zeroing
them out changed only the zero bits of the output. An fp16-flushed variant (13.3 ms) was kept as the
leading Raspberry Pi candidate specifically because the Cortex-A76 has native fp16 SIMD (ARMv8.2)
that x86 lacks — on x86, `onnxruntime` just casts fp16→fp32 internally anyway, so fp32-flushed was
chosen for the PC/x86 deployment. An int8-quantized variant reached F1 86.6%/FAR 5.2% but was
*slower* than fp32 on x86 (no optimized int8 3D-conv path in `onnxruntime` at the time) — kept only
as an untested Pi candidate.

The `2men_clash` scene was deliberately held out entirely from training, as a cross-room
generalization benchmark — see the earlier session note on locating this scene's source data
(it's no longer present in this repo; only its evaluation results are, in
`reports/eval_2men_clash.json` and `scripts/_eval_2men_clash.py`/`_integrate_2men_clash.py`, kept
specifically as a template for integrating whatever labeled replacement video arrives next).
