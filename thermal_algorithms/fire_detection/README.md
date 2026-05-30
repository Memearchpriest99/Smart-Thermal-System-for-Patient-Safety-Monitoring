# fire_detection

Two alternative fire detection algorithms (§ 4.4.4), both implementing `FireDetector` → `predict(frame) → FireAlert`.

`FireAlert.level` is one of four states:

| Level | Meaning |
|---|---|
| `SAFE` | No thermal anomaly |
| `POTENTIAL_FIRE` | Large hot blob detected — awaiting growth confirmation |
| `IGNITION_SOURCE` | Small concentrated heat source — **immediate alert** |
| `ACTIVE_COMBUSTION` | Growth confirmed — **immediate alert** |

`alert.is_alarm` is `True` for `IGNITION_SOURCE` and `ACTIVE_COMBUSTION`.

---

## OtsuFireDetector (§ 4.4.4 — primary)

Three-stage pipelined approach. Rule-based — no training required.

**Stage 1 — Adaptive segmentation:**
- Variance gate: skip frames where `ΔT < ΔT_nom` (static background, sensor noise)
- Otsu's method: find optimal threshold maximising between-class variance
- Morphological shaping: 1× erosion + 2× dilation (3×3 kernel)

**Stage 2 — Decision Tree Classifier:**
- `T_max > T_ign AND area < A_limit` → **IGNITION_SOURCE** (immediate alert)
- `T_max > T_fire AND area ≥ A_limit` → **POTENTIAL_FIRE** (start temporal tracking)

**Stage 3 — Temporal Mass Gradient:**
- Track blob area `A_n` over sliding window (default Δt = 1 s, k = 8 frames at 8 Hz)
- Gradient `G_n = (A_n - A_{n-k}) / Δt`; increment counter `S_n` if growing
- `S_n ≥ S_threshold` → **ACTIVE_COMBUSTION**; timeout without growth → **SAFE**

This detector is **stateful** — call `reset()` between independent sequences.

```python
from thermal_algorithms.fire_detection import OtsuFireDetector
detector = OtsuFireDetector(MLX90640, t_ign=45.0, t_fire=60.0).fit([])
alert = detector.predict(frame)
```

---

## FireSVMDetector (§ 4.4.4 — ML alternative)

Binary classifier (SAFE / ACTIVE_COMBUSTION) on a 6-feature vector extracted from the largest thermal blob: `max_temp`, `mean_temp`, `std_temp`, `area`, `skewness`, `kurtosis`.

Resolution-invariant — one checkpoint works for both sensor profiles.

```python
from thermal_algorithms.fire_detection import FireSVMDetector
from thermal_algorithms.training import FireFrameDataset, DatasetIndex

ds = FireFrameDataset(DatasetIndex("data/"))
detector = FireSVMDetector(kernel="rbf").fit(
    [f for f, _ in ds], [a for _, a in ds]
)
```
