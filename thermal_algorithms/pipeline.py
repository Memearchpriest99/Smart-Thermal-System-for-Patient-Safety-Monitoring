"""ThermalPipeline — runtime integration of all detection paths (§ 4.2.1).

Matches the high-level algorithmic flow in Figure 5 of the Engineering Report:

    Thermal Sensors
        └─► Embedded Controller
                ├─► Is there fire ignition?
                │       YES ──► Fire Ignition Detection ──► Send Alert
                │
                ├─► Is there inappropriate activity?
                │       YES ──► Abnormal Activity Detection ──► Send Alert
                │
                └─► Is there a person in a restricted area?  [future]
                        YES ──► Restricted Area Breach Detection ──► Send Alert

All three detection paths are independent — each produces its own alert stream.

Usage
-----
::

    from thermal_algorithms.pipeline import ThermalPipeline, PipelineResult
    from thermal_algorithms.core.sensor_profile import MLX90640
    from thermal_algorithms.preprocessing import TatenoPipeline
    from thermal_algorithms.human_detection import AdaptiveThresholdDetector
    from thermal_algorithms.fire_detection import OtsuFireDetector
    from thermal_algorithms.contact_detection import GeometricContactDetector

    # Build components
    preprocessor = TatenoPipeline(MLX90640).fit(calibration_frames)
    human_det    = AdaptiveThresholdDetector(MLX90640).fit([])
    fire_det     = OtsuFireDetector(MLX90640).fit([])
    contact_det  = GeometricContactDetector(homography=H).fit([])

    pipeline = ThermalPipeline(
        preprocessor=preprocessor,
        human_detector=human_det,
        fire_detector=fire_det,
        contact_detector=contact_det,
    )

    # Inference loop (called once per sensor readout cycle, ~8 Hz)
    result = pipeline.process(frame_cam0, frame_cam1, frame_cam2)
    for alert in result.alerts:
        print(f"ALERT  type={alert.type.value}  confidence={alert.confidence:.2f}")

Notes
-----
* All detectors receive *preprocessed* frames by default.  If no preprocessor
  is provided, raw sensor frames are used for all paths.

* Stateful detectors (``OtsuFireDetector``, ``MVSTGCNDetector``,
  ``ThermoX3DDetector``) are reset automatically when ``reset()`` is called.

* Alert cooldown prevents spamming the same alert type.  Each type has an
  independent configurable silence window after it fires.

* The Restricted Area path is reserved — not implemented at this stage.
  A ``restricted_area_masks`` parameter accepts floor-plane polygons (one per
  camera) for future integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from thermal_algorithms.core.types import ContactEvent, Detection, FireAlert, FireLevel, Frame


# ---------------------------------------------------------------------------
# Alert types
# ---------------------------------------------------------------------------

class AlertType(Enum):
    """Enumeration of system-level alert categories (§ 2.3).

    FIRE             — §4.4.4: ignition source or active combustion detected on
                       any camera.  Triggers immediate staff notification.
    CONTACT          — §4.4.3: inappropriate physical contact between patients
                       (covers both violent acts and sexual behaviour — system
                       cannot distinguish the two at this resolution; both
                       require immediate response).
    RESTRICTED_AREA  — §2.3 scenario 4: a person detected in a zone where
                       patients are not permitted.  Only active when the
                       pipeline is constructed with ``restricted=True``.
                       In restricted mode the pipeline exclusively runs human
                       detection — fire and contact detection are skipped
                       (matching the §4.2.1 Figure 5 branch exactly).
    """
    FIRE             = "fire"
    CONTACT          = "contact"
    RESTRICTED_AREA  = "restricted_area"


@dataclass(frozen=True)
class Alert:
    """A single system-level alarm event.

    Attributes:
        type: Which scenario was detected.
        timestamp: Capture timestamp of the triggering frame (seconds).
        confidence: Detector confidence ∈ [0, 1].  Rule-based detectors
            (``GeometricContactDetector``, ``OtsuFireDetector``) yield 1.0;
            ML-based detectors yield a softmax probability.
        source_cameras: Which camera indices contributed to this alert.
        debug: Algorithm-specific diagnostic payload (level, n_actors, etc.).
    """
    type: AlertType
    timestamp: float
    confidence: float
    source_cameras: tuple[int, ...]
    debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineResult:
    """Full output of one ``ThermalPipeline.process()`` call.

    Carries both the actionable alerts and all intermediate algorithm outputs
    so the GUI, data-recorder, and post-hoc analysis tools can inspect any
    layer without re-running the pipeline.

    Attributes:
        timestamp: Maximum frame timestamp in the input triplet.
        alerts: Tuple of active alerts this cycle (empty = all-clear).
        processed_frames: Preprocessed frames (or raw if no preprocessor).
        detections: Per-camera human detections from the HumanDetector.
        fire_alerts: Per-camera FireAlert objects from the FireDetector.
        contact_event: ContactEvent from the ContactDetector, or None if no
            ContactDetector is configured.
    """
    timestamp: float
    alerts: tuple[Alert, ...]
    processed_frames: tuple[Frame, Frame, Frame]
    detections: tuple[list[Detection], list[Detection], list[Detection]]
    fire_alerts: tuple[FireAlert, FireAlert, FireAlert]
    contact_event: Optional[ContactEvent]

    @property
    def any_alarm(self) -> bool:
        """True if at least one alert fired this cycle."""
        return len(self.alerts) > 0

    @property
    def fire_alarm(self) -> bool:
        """True if a FIRE alert fired this cycle."""
        return any(a.type == AlertType.FIRE for a in self.alerts)

    @property
    def contact_alarm(self) -> bool:
        """True if a CONTACT alert fired this cycle."""
        return any(a.type == AlertType.CONTACT for a in self.alerts)

    @property
    def restricted_area_alarm(self) -> bool:
        """True if a RESTRICTED_AREA alert fired this cycle."""
        return any(a.type == AlertType.RESTRICTED_AREA for a in self.alerts)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ThermalPipeline:
    """Runtime integration of all algorithm layers (Figure 5 flow diagram).

    Args:
        preprocessor: Optional fitted ``Preprocessor``.  Applied to every
            frame before all downstream detectors.  If None, raw frames are
            passed to all detectors.
        human_detector: Optional fitted ``HumanDetector``.  Produces per-camera
            bounding boxes used as input to the ``contact_detector``.  Also
            used for the (future) restricted-area path.
        fire_detector: Optional fitted ``FireDetector``.  Run independently on
            each camera; FIRE alert fires if *any* camera reports an alarm.
        contact_detector: Optional fitted ``ContactDetector``.  Receives all
            three (preprocessed) frames and the per-camera human detections.
        fire_cooldown_s: Minimum seconds between successive FIRE alerts.
            Default 30 s — long enough to avoid alert fatigue while still
            responding within the 1-minute requirement from §2.1.
        contact_cooldown_s: Minimum seconds between successive CONTACT alerts.
            Default 30 s for the same reason.
    """

    def __init__(
        self,
        *,
        preprocessor=None,
        human_detector=None,
        fire_detector=None,
        contact_detector=None,
        restricted: bool = False,
        fire_cooldown_s: float = 30.0,
        contact_cooldown_s: float = 30.0,
        restricted_area_cooldown_s: float = 30.0,
    ) -> None:
        self._preprocessor = preprocessor
        self._human_detector = human_detector
        self._fire_detector = fire_detector
        self._contact_detector = contact_detector
        self._restricted = bool(restricted)

        self._cooldown: dict[AlertType, float] = {
            AlertType.FIRE:            float(fire_cooldown_s),
            AlertType.CONTACT:         float(contact_cooldown_s),
            AlertType.RESTRICTED_AREA: float(restricted_area_cooldown_s),
        }
        # Timestamps of the most recent alert of each type (None = never fired)
        self._last_alert: dict[AlertType, Optional[float]] = {t: None for t in AlertType}

    # ---- Core inference --------------------------------------------------

    def process(
        self,
        frame0: Frame,
        frame1: Frame,
        frame2: Frame,
    ) -> PipelineResult:
        """Process one synchronized frame triplet and return all results.

        Args:
            frame0: Frame from camera 0 (channel 0 on the TCA9548A).
            frame1: Frame from camera 1.
            frame2: Frame from camera 2 (read last; latest timestamp).

        Returns:
            ``PipelineResult`` containing any active alerts plus all
            intermediate algorithm outputs.
        """
        frames_raw = (frame0, frame1, frame2)
        timestamp = max(f.timestamp for f in frames_raw)

        # ── Stage 1: Preprocessing (both modes) ─────────────────────────
        if self._preprocessor is not None:
            processed: tuple[Frame, Frame, Frame] = tuple(
                self._preprocessor.predict(f) for f in frames_raw
            )  # type: ignore[assignment]
        else:
            processed = frames_raw

        # ── Restricted-area mode (Figure 5 — YES branch) ────────────────
        # Human detection only; fire and contact are deliberately skipped.
        if self._restricted:
            return self._process_restricted(timestamp, processed)

        # ── Full pipeline (Figure 5 — NO branch) ────────────────────────

        # ── Stage 2: Human detection (feeds contact detector) ───────────
        if self._human_detector is not None:
            detections: tuple[list[Detection], ...] = tuple(
                self._human_detector.predict(f) for f in processed
            )  # type: ignore[assignment]
        else:
            detections = ([], [], [])

        # ── Stage 3a: Fire detection (per camera, independent) ──────────
        if self._fire_detector is not None:
            fire_alerts: tuple[FireAlert, FireAlert, FireAlert] = tuple(
                self._fire_detector.predict(f) for f in processed
            )  # type: ignore[assignment]
        else:
            fire_alerts = (
                FireAlert(level=FireLevel.SAFE, timestamp=timestamp),
                FireAlert(level=FireLevel.SAFE, timestamp=timestamp),
                FireAlert(level=FireLevel.SAFE, timestamp=timestamp),
            )

        # ── Stage 3b: Contact detection (3-view) ────────────────────────
        contact_event: Optional[ContactEvent] = None
        if self._contact_detector is not None:
            contact_event = self._contact_detector.predict(
                processed,
                detections=detections,  # type: ignore[arg-type]
            )

        # ── Stage 4: Compose alerts with cooldown ───────────────────────
        alerts = self._compose_alerts(timestamp, fire_alerts, contact_event)

        return PipelineResult(
            timestamp=timestamp,
            alerts=tuple(alerts),
            processed_frames=processed,
            detections=detections,  # type: ignore[arg-type]
            fire_alerts=fire_alerts,
            contact_event=contact_event,
        )

    def _process_restricted(
        self,
        timestamp: float,
        processed: tuple[Frame, Frame, Frame],
    ) -> "PipelineResult":
        """Restricted-area branch: human detection only (§4.2.1 Figure 5)."""
        _safe_fa = FireAlert(level=FireLevel.SAFE, timestamp=timestamp)

        if self._human_detector is not None:
            detections: tuple[list[Detection], ...] = tuple(
                self._human_detector.predict(f) for f in processed
            )  # type: ignore[assignment]
        else:
            detections = ([], [], [])

        # Any person detected in a restricted zone → alert
        alerts: list[Alert] = []
        any_person = any(len(d) > 0 for d in detections)
        if any_person and self._cooldown_ok(AlertType.RESTRICTED_AREA, timestamp):
            cam_ids = tuple(
                cam_id
                for cam_id, d in enumerate(detections)
                if len(d) > 0
            )
            alerts.append(Alert(
                type=AlertType.RESTRICTED_AREA,
                timestamp=timestamp,
                confidence=1.0,
                source_cameras=cam_ids,
                debug={"n_detections": sum(len(d) for d in detections)},
            ))
            self._last_alert[AlertType.RESTRICTED_AREA] = timestamp

        return PipelineResult(
            timestamp=timestamp,
            alerts=tuple(alerts),
            processed_frames=processed,
            detections=detections,  # type: ignore[arg-type]
            fire_alerts=(_safe_fa, _safe_fa, _safe_fa),
            contact_event=None,
        )

    # ---- Alert composition -----------------------------------------------

    def _compose_alerts(
        self,
        timestamp: float,
        fire_alerts: tuple,
        contact_event: Optional[ContactEvent],
    ) -> list[Alert]:
        alerts: list[Alert] = []

        # ── Fire path ───────────────────────────────────────────────────
        alarming_fire = [
            (cam_id, fa)
            for cam_id, fa in enumerate(fire_alerts)
            if fa.is_alarm
        ]
        if alarming_fire and self._cooldown_ok(AlertType.FIRE, timestamp):
            # Pick the highest-confidence detection across cameras
            best_cam, best_fa = max(alarming_fire, key=lambda x: x[1].confidence)
            alerts.append(Alert(
                type=AlertType.FIRE,
                timestamp=timestamp,
                confidence=best_fa.confidence,
                source_cameras=(best_cam,),
                debug={
                    "level": best_fa.level.value,
                    "all_alarming_cameras": [c for c, _ in alarming_fire],
                },
            ))
            self._last_alert[AlertType.FIRE] = timestamp

        # ── Contact path ────────────────────────────────────────────────
        if (
            contact_event is not None
            and contact_event.any_contact
            and self._cooldown_ok(AlertType.CONTACT, timestamp)
        ):
            actor_cams: tuple[int, ...] = tuple(
                cam_id
                for actor in contact_event.actors
                for cam_id in actor.source_camera_ids
            )
            alerts.append(Alert(
                type=AlertType.CONTACT,
                timestamp=timestamp,
                confidence=contact_event.confidence,
                source_cameras=actor_cams or (0, 1, 2),
                debug={
                    "n_actors": len(contact_event.actors),
                    "n_contact_pairs": len(contact_event.pairs_in_contact),
                },
            ))
            self._last_alert[AlertType.CONTACT] = timestamp

        # Note: RESTRICTED_AREA alerts are generated in _process_restricted(),
        # which is called before this method when self._restricted is True.
        # This path (full pipeline) never produces RESTRICTED_AREA alerts.

        return alerts

    def _cooldown_ok(self, alert_type: AlertType, timestamp: float) -> bool:
        """Return True if enough time has passed since the last alert of this type."""
        last = self._last_alert.get(alert_type)
        if last is None:
            return True
        return (timestamp - last) >= self._cooldown[alert_type]

    # ---- State management ------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state.

        Resets:
        * Alert cooldown timers (all alert types become immediately fireable).
        * Temporal state of stateful detectors (``OtsuFireDetector`` growth
          tracker, ``MVSTGCNDetector`` / ``ThermoX3DDetector`` rolling buffers).
        """
        for key in self._last_alert:
            self._last_alert[key] = None
        for component in (
            self._preprocessor,
            self._human_detector,
            self._fire_detector,
            self._contact_detector,
        ):
            if component is not None and hasattr(component, "reset"):
                component.reset()

    def reset_cooldown(self, alert_type: Optional[AlertType] = None) -> None:
        """Clear cooldown timers without touching detector state.

        Args:
            alert_type: If given, clear only that type's timer.
                If None, clear all timers.
        """
        if alert_type is None:
            for key in self._last_alert:
                self._last_alert[key] = None
        else:
            self._last_alert[alert_type] = None

    # ---- Properties ------------------------------------------------------

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def human_detector(self):
        return self._human_detector

    @property
    def fire_detector(self):
        return self._fire_detector

    @property
    def contact_detector(self):
        return self._contact_detector

    @property
    def cooldown_s(self) -> dict[AlertType, float]:
        """Per-type cooldown durations in seconds."""
        return dict(self._cooldown)

    @property
    def restricted(self) -> bool:
        """True if the pipeline is in restricted-area mode."""
        return self._restricted

    def __repr__(self) -> str:
        parts = []
        if self._restricted:
            parts.append("restricted=True")
        if self._preprocessor:
            parts.append(f"preprocessor={type(self._preprocessor).__name__}")
        if self._human_detector:
            parts.append(f"human={type(self._human_detector).__name__}")
        if self._fire_detector:
            parts.append(f"fire={type(self._fire_detector).__name__}")
        if self._contact_detector:
            parts.append(f"contact={type(self._contact_detector).__name__}")
        return f"ThermalPipeline({', '.join(parts)})"
