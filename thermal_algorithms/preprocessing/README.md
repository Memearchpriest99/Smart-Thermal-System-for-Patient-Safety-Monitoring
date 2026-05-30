# preprocessing

## TatenoPipeline (§ 4.4.1)

Three-stage transformation applied to every raw thermal frame before any detection algorithm runs:

```
raw frame  →  Gaussian smooth  →  background subtract  →  L1 residual
```

1. **Spatial denoising** — 2D Gaussian blur suppresses per-pixel thermal noise
2. **Background subtraction** — subtract mean-field background `B(x,y)` learned from empty-room calibration frames
3. **Residual rectification** — take absolute value `|I_s - B|` so anomalies (humans, fires) stand out regardless of whether they are hotter or cooler than background

**Calibration:** call `fit(empty_room_frames)` before any inference. The pipeline achieves a 5.61× SBR improvement on the project dataset (1.21 raw → 6.80 processed, §5.3.1).

```python
from thermal_algorithms.preprocessing import TatenoPipeline
from thermal_algorithms.core.sensor_profile import MLX90640

pipeline = TatenoPipeline(sensor_profile=MLX90640)
pipeline.fit(calibration_frames)          # empty room
residual_frame = pipeline.predict(frame)  # one inference frame
```

The Gaussian σ defaults to a physical smoothing scale (5 cm at 2 m range) and auto-converts to pixels via the sensor profile.
