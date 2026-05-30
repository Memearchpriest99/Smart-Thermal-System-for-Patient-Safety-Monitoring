"""thermal_algorithms — algorithms for the Smart Thermal System for Patient Safety Monitoring.

Implements every algorithm described in Chapter 4.4 of the Engineering Report,
behind a uniform abstract interface so concrete implementations are interchangeable
at the call site.

Top-level structure:
    core/               — Universal ABC, dataclasses, sensor profiles, IO, checkpoints
    preprocessing/      — Tateno pre-processing pipeline (4.4.1)
    human_detection/    — Adaptive threshold, HOG+SVM, MobileNet-SSD (4.4.2)
    contact_detection/  — Geometric, MV-STGCN, Thermo-X3D (4.4.3)
    fire_detection/     — Otsu pipeline, Fire SVM (4.4.4)
    training/           — Reference trainer, datasets, evaluation metrics
    pipeline.py         — ThermalPipeline (runtime integration, Figure 5 flow)
"""

__version__ = "0.1.0"

from thermal_algorithms.pipeline import Alert, AlertType, PipelineResult, ThermalPipeline

__all__ = ["Alert", "AlertType", "PipelineResult", "ThermalPipeline"]
