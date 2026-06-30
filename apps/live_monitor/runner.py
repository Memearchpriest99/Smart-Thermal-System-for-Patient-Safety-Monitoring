"""Qt-free orchestration core.

``PipelineRunner`` owns the three frame sources, the swappable detectors, and a
``ThermalPipeline``. It assembles synchronized triplets, runs inference, times
each stage, and exposes live controls (algorithm swap, restricted mode,
homography, threshold tuning). The Qt worker is a thin wrapper that drives this
from a background thread; keeping the logic here makes it unit-testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Frame, HomographyMatrices
from thermal_algorithms.pipeline import PipelineResult, ThermalPipeline

from apps.live_monitor.capture.base import FrameSource
from apps.live_monitor.detectors import (
    BuildContext,
    DetectorOption,
    apply_thresholds,
    build_registry,
)

# pipeline attribute that backs each task's detector
_TASK_ATTR = {
    "fire": "_fire_detector",
    "person": "_human_detector",
    "touch": "_contact_detector",
}


@dataclass
class RunnerResult:
    """One processed triplet plus the raw frames and timing info for the UI."""

    result: PipelineResult
    frames: tuple[Frame, Frame, Frame]      # raw frames used (for display)
    timings_ms: dict[str, float] = field(default_factory=dict)


class PipelineRunner:
    def __init__(
        self,
        sources: Sequence[FrameSource],
        profile: SensorProfile,
        registry,  # CheckpointRegistry
        *,
        restricted: bool = False,
        epsilon_m: float = 0.5,
        delta_m: float = 0.5,
        homography: Optional[HomographyMatrices] = None,
        preprocessor=None,
    ) -> None:
        if len(sources) != 3:
            raise ValueError(f"need exactly 3 sources, got {len(sources)}")
        self._sources = list(sources)
        self._profile = profile
        self._registry = registry
        self._options = build_registry()
        self._epsilon_m = epsilon_m
        self._delta_m = delta_m
        self._homography = homography
        self._thresholds: dict[str, float] = {}

        self._pipeline = ThermalPipeline(preprocessor=preprocessor, restricted=restricted)
        self._selection: dict[str, str] = {}

        self._latest: list[Optional[Frame]] = [None, None, None]
        self._frames_processed = 0
        self._dropped = 0

        # Build the default (rule-based, always-available) detector per task.
        for task, opts in self._options.items():
            self.select(task, opts[0])

    # ---- properties -----------------------------------------------------

    @property
    def pipeline(self) -> ThermalPipeline:
        return self._pipeline

    @property
    def options(self) -> dict[str, list[DetectorOption]]:
        return self._options

    @property
    def selection(self) -> dict[str, str]:
        return dict(self._selection)

    @property
    def homography(self) -> Optional[HomographyMatrices]:
        return self._homography

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def dropped(self) -> int:
        return self._dropped

    def build_context(self) -> BuildContext:
        return BuildContext(
            profile=self._profile,
            registry=self._registry,
            homography=self._homography,
            epsilon_m=self._epsilon_m,
            delta_m=self._delta_m,
        )

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> "PipelineRunner":
        for s in self._sources:
            s.start()
        return self

    def stop(self) -> None:
        for s in self._sources:
            s.stop()

    # ---- controls -------------------------------------------------------

    def select(self, task: str, option: DetectorOption | str) -> DetectorOption:
        """Swap the active detector for a task. Accepts an option or its key.

        Raises ``DetectorUnavailable`` if the option can't be built; the caller
        (UI) should keep the previous selection in that case.
        """
        opts = self._options[task]
        if isinstance(option, str):
            option = next(o for o in opts if o.key == option)
        detector = option.build(self.build_context())
        if self._thresholds:
            apply_thresholds(detector, **self._thresholds)
        setattr(self._pipeline, _TASK_ATTR[task], detector)
        self._pipeline.reset()
        self._selection[task] = option.key
        return option

    def set_restricted(self, restricted: bool) -> None:
        # ThermalPipeline reads _restricted at process time.
        self._pipeline._restricted = bool(restricted)
        self._pipeline.reset()

    def set_homography(self, homography: Optional[HomographyMatrices]) -> None:
        self._homography = homography
        # Rebuild the touch detector so it picks up the new floor map.
        if "touch" in self._selection:
            try:
                self.select("touch", self._selection["touch"])
            except Exception:
                pass

    def set_thresholds(self, **knobs: float) -> dict[str, float]:
        self._thresholds.update({k: v for k, v in knobs.items() if v is not None})
        if "epsilon_m" in knobs and knobs["epsilon_m"] is not None:
            self._epsilon_m = knobs["epsilon_m"]
        if "delta_m" in knobs and knobs["delta_m"] is not None:
            self._delta_m = knobs["delta_m"]
        applied: dict[str, float] = {}
        for attr in _TASK_ATTR.values():
            det = getattr(self._pipeline, attr, None)
            if det is not None:
                applied.update(apply_thresholds(det, **knobs))
        return applied

    def is_geometric_touch(self) -> bool:
        return self._selection.get("touch") == "geometric_contact_detector"

    # ---- inference ------------------------------------------------------

    def poll(self) -> Optional[RunnerResult]:
        """Read the newest frame from each source; process once all 3 are ready.

        Returns ``None`` until a full triplet is available. Each source caches
        its latest frame, so a slow camera reuses its previous frame rather than
        stalling the others.
        """
        got_new = False
        for i, src in enumerate(self._sources):
            f = src.read()
            if f is not None:
                if self._latest[i] is not None:
                    # a frame arrived before we consumed the prior one
                    self._dropped += 0  # accounted per-camera in the worker
                self._latest[i] = f
                got_new = True
        if not got_new or any(f is None for f in self._latest):
            return None
        triplet = (self._latest[0], self._latest[1], self._latest[2])
        return self.process_triplet(triplet)  # type: ignore[arg-type]

    def process_triplet(self, frames: tuple[Frame, Frame, Frame]) -> RunnerResult:
        """Run the pipeline on a triplet and time the call."""
        t0 = time.perf_counter()
        result = self._pipeline.process(*frames)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._frames_processed += 1
        return RunnerResult(result=result, frames=frames, timings_ms={"pipeline": dt_ms})
