# contact_detection

Three alternative contact detection algorithms (§ 4.4.3). All consume a 3-camera frame triplet and output a `ContactEvent`.

The shared multi-view utilities live in [`multi_view/`](multi_view/README.md).

---

## GeometricContactDetector (§ 4.4.3.1 — primary baseline)

Fully deterministic. No training required. Requires calibrated homography matrices.

**Pipeline:**
1. Extract foot-point `P_foot = (x + w/2, y + h)` from each bounding box
2. Project through `H_k` → world floor coordinates `(X_w, Y_w)` per camera
3. Cross-camera validation: discard single-source projections; outlier removal for N=3
4. Cluster within `epsilon_m` metres → unique `ActorPosition` nodes
5. Flag any pair with `D_{i,j} < delta_m` as contact

**Key parameters:**
- `epsilon_m` — cross-camera clustering threshold (reflects homography accuracy, typically 0.15–0.20 m)
- `delta_m` — contact proximity threshold (reflects physical person width, typically 0.5–0.6 m)

> ⚠️ These two parameters must be tuned independently. `epsilon_m` is about sensor accuracy; `delta_m` is about the clinical definition of contact.

**Calibration:** use `calibrate_homography()` or `examples/calibrate_homography.py`.

```python
from thermal_algorithms.contact_detection import GeometricContactDetector
detector = GeometricContactDetector(homography=H, epsilon_m=0.2, delta_m=0.5).fit([])
event = detector.predict((frame0, frame1, frame2), detections=(dets0, dets1, dets2))
```

---

## MVSTGCNDetector (§ 4.4.3.2)

Multi-View Spatiotemporal Graph CNN. Requires `torch`.

Three stages: Kalman-tracked detections → homographic fusion to unique actor nodes → adaptive ST-GCN over 16-frame sliding window. Handles the I2C phase shift (Camera 3 reads ~50 ms after Camera 1) via the Kalman tracker.

Node features: `[X_norm, Y_norm, vx, vy, max_temp_norm, mean_temp_norm, bbox_area]`

---

## ThermoX3DDetector (§ 4.4.3.3)

Volumetric pixel-based failsafe. Does not use bounding boxes. Requires `torch`.

Handles the failure mode where intimate contact causes bounding boxes to merge (detector sees one blob → GCN has no edges). Architecture: 3× Micro-X3D stream (factorised (2+1)D ResBlocks + thermal CBAM attention) with late feature fusion.

Input per camera: 16-frame rolling buffer at effective 8 Hz regardless of capture FPS.

---

## Calibration (all algorithms)

```python
from examples.calibrate_homography import main as calibrate
# or manually:
from examples.utils import make_synthetic_homographies  # synthetic (demos only)
detector.calibrate_homography(marker_correspondences)   # real deployment
```
