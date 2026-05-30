# multi_view

Shared utilities used by all three contact detection algorithms.

## homography.py

SVD-based homography solver (Hot-Point Calibration).

```python
from thermal_algorithms.contact_detection.multi_view import solve_homography_from_markers, project_foot_point

H_matrices = solve_homography_from_markers([
    (camera_id, [((u, v), (Xw, Yw)), ...]),  # ≥4 pairs per camera
    ...
])
Xw, Yw = project_foot_point((u, v), H_matrices[0])
```

Standard checkerboard calibration fails in thermal (paper has uniform emissivity). Use **heated markers** (hot water containers, hand warmers) at known floor positions. See `examples/calibrate_homography.py`.

## tracker.py

Constant-velocity Kalman filter for I2C phase-shift correction.

The TCA9548A reads sensors sequentially — Camera 3 is read ~50 ms after Camera 1. At walking speeds this creates "ghosting" in multi-view fusion. `PerCameraTracker` forward-predicts Camera-1 detections by Δt to align them with Camera-3's capture time. It also smooths 1-pixel quantisation jitter.

```python
from thermal_algorithms.contact_detection.multi_view import PerCameraTracker
tracker = PerCameraTracker(dt=1/8)
tracker.predict_all()
tracker.update([(u, v), ...])  # detection centroids this frame
```

## fusion.py

Foot-point projection + cross-camera validation + clustering → `ActorPosition` list.

Validation rules (§ 4.4.3.1):
- N = 1: discard (no cross-camera corroboration)
- N = 3: outlier removal (if one point is inconsistent with both others, drop it)
- N ≥ 2 consistent: centroid = actor position

```python
from thermal_algorithms.contact_detection.multi_view import fuse_detections
actors, next_id = fuse_detections(
    detections_per_camera,  # list of list[Detection], one per camera
    homographies,
    epsilon_m=0.2,
)
```
