"""Trainer — evaluation harness connecting datasets, algorithms, and metrics.

Reproduces the §5.3 Algorithmic Performance Evaluation tables:
  - §5.3.1  ``evaluate_preprocessing``  →  SBR improvement factor
  - §5.3.2  ``evaluate_human_detection`` → Table 3 (Acc / Prec / Rec / F1)
  - §5.3.4  ``evaluate_fire_detection``  → Table 4 (IoU / Acc / Prec / Rec / F1)
  - §5.3.3  ``evaluate_contact_detection`` (when data is available)

Typical workflow
----------------
::

    from thermal_algorithms.training import (
        DatasetIndex, FrameLevelDataset, FireFrameDataset,
        Trainer, PERSON_CLASS_ID, FIRE_CLASS_ID,
        format_scenario_table,
    )
    from thermal_algorithms.core.sensor_profile import MLX90640
    from thermal_algorithms.preprocessing import TatenoPipeline
    from thermal_algorithms.human_detection import AdaptiveThresholdDetector

    index = DatasetIndex("data/", sensor_profile=MLX90640)

    # 1. Calibrate preprocessor on empty-room sessions
    preprocessor = TatenoPipeline(MLX90640)
    bg = FrameLevelDataset(index, scenes=["empty_room"],
                           include_negative_frames=True,
                           class_filter=[])
    preprocessor.fit([f for f, _ in bg])

    # 2. Evaluate human detector (raw vs. processed)
    human_ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                                 include_negative_frames=True)

    detector = AdaptiveThresholdDetector(MLX90640).fit([])
    raw_results  = Trainer.evaluate_human_detection(detector, human_ds, mode="raw")
    proc_results = Trainer.evaluate_human_detection(detector, human_ds,
                                                    preprocessor=preprocessor,
                                                    mode="proc")
    print(format_scenario_table(raw_results + proc_results))

Notes
-----
* Stateful detectors (``OtsuFireDetector``, ``MVSTGCNDetector``,
  ``ThermoX3DDetector``) have ``reset()`` called at the start of every new
  session so temporal buffers do not bleed between scenes.

* The ``mode`` string ("raw" / "proc") is passed through to ``ScenarioResult``
  to label the table rows.

* All evaluation is frame-level binary classification: a frame is predicted
  positive if the algorithm outputs any detection / non-SAFE / any-contact.
"""

from __future__ import annotations

from typing import Optional

from thermal_algorithms.training.datasets import (
    ContactFrameDataset,
    FireFrameDataset,
    FrameLevelDataset,
)
from thermal_algorithms.training.metrics import (
    BinaryConfusionMatrix,
    ScenarioResult,
    best_iou,
    binary_confusion_matrix,
    mean_sbr,
    signal_to_background_ratio,
)
from thermal_algorithms.core.types import FireLevel


class Trainer:
    """Static evaluation harness — no instance state needed."""

    # ------------------------------------------------------------------
    # §5.3.1 — Preprocessing evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_preprocessing(
        dataset: FrameLevelDataset,
        preprocessor,
        *,
        verbose: bool = True,
    ) -> tuple[float, float]:
        """Compute mean SBR on raw vs. preprocessed frames.

        Args:
            dataset: Labeled dataset; ground-truth bboxes supply the signal
                region for SBR computation.  Must include negative frames
                (pass ``include_negative_frames=True``) if you want a
                full-dataset average.
            preprocessor: A fitted ``Preprocessor`` (e.g. ``TatenoPipeline``).
            verbose: Print a one-line summary.

        Returns:
            ``(mean_raw_sbr, mean_processed_sbr)`` — the improvement factor
            is ``processed / raw``.  The Engineering Report achieved 5.61×
            (1.21 → 6.80).
        """
        raw_sbrs: list[float] = []
        proc_sbrs: list[float] = []

        for frame, gt_dets in dataset:
            gt_bboxes = [d.bbox for d in gt_dets]
            raw_sbrs.append(signal_to_background_ratio(frame.data, gt_bboxes))
            proc_frame = preprocessor.predict(frame)
            proc_sbrs.append(signal_to_background_ratio(proc_frame.data, gt_bboxes))

        mean_raw = float(sum(raw_sbrs) / len(raw_sbrs)) if raw_sbrs else 1.0
        mean_proc = float(sum(proc_sbrs) / len(proc_sbrs)) if proc_sbrs else 1.0
        if verbose:
            factor = mean_proc / mean_raw if mean_raw > 0 else float("nan")
            print(
                f"SBR: raw={mean_raw:.2f}  processed={mean_proc:.2f}  "
                f"improvement={factor:.2f}×"
            )
        return mean_raw, mean_proc

    # ------------------------------------------------------------------
    # §5.3.2 — Human detection evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_human_detection(
        detector,
        dataset: FrameLevelDataset,
        *,
        preprocessor=None,
        mode: str,
        iou_threshold: float = 0.5,
        verbose: bool = True,
    ) -> list[ScenarioResult]:
        """Evaluate a HumanDetector per session (one ScenarioResult each).

        Args:
            detector: Any fitted ``HumanDetector``.
            dataset: ``FrameLevelDataset`` with ``class_filter=[PERSON_CLASS_ID]``
                and ``include_negative_frames=True``.
            preprocessor: Optional fitted ``Preprocessor``.  When given, each
                frame is preprocessed before detection (= "Proc" mode in Table 3).
            mode: Label string written into ``ScenarioResult.mode`` ("raw" / "proc").
            iou_threshold: Minimum IoU to count a detection as a true positive
                (not used for the binary frame-level metric, but exposed for
                future bbox-level evaluation).
            verbose: Print a one-line summary per session.
        """
        results: list[ScenarioResult] = []

        for session, examples in dataset.by_session():
            y_true: list[int] = []
            y_pred: list[int] = []
            gt_bboxes_per_frame: list[list] = []
            pred_bboxes_per_frame: list[list] = []

            for frame, gt_dets in examples:
                if preprocessor is not None:
                    frame = preprocessor.predict(frame)
                pred_dets = detector.predict(frame)

                y_true.append(1 if gt_dets else 0)
                y_pred.append(1 if pred_dets else 0)
                gt_bboxes_per_frame.append([d.bbox for d in gt_dets])
                # Only collect bboxes from proper Detection objects
                pred_bboxes_per_frame.append(
                    [d.bbox for d in pred_dets if hasattr(d, "bbox")]
                )

            cm = binary_confusion_matrix(y_true, y_pred)
            result = ScenarioResult(
                name=session.scene,
                mode=mode,
                confusion=cm,
                mean_iou=0.0,  # Table 3 does not include IoU
            )
            results.append(result)

            if verbose:
                print(f"  {session.scene:<40} {mode}  {result}")

        return results

    # ------------------------------------------------------------------
    # §5.3.4 — Fire detection evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_fire_detection(
        detector,
        dataset: FireFrameDataset,
        *,
        preprocessor=None,
        mode: str,
        verbose: bool = True,
    ) -> list[ScenarioResult]:
        """Evaluate a FireDetector per session with binary labels + IoU.

        The ``detector.reset()`` is called at each session boundary to clear
        the ``OtsuFireDetector``'s temporal mass-gradient tracker.

        Positive prediction: ``alert.level != FireLevel.SAFE``.
        IoU: computed from ``pred_alert.blob_features["bbox"]`` (set by
        ``OtsuFireDetector``) vs. ground-truth bboxes in
        ``gt_alert.blob_features["bboxes"]``.  Frames without gt bboxes are
        excluded from the IoU average (matching Table 4 methodology).
        """
        results: list[ScenarioResult] = []

        for session, examples in dataset.by_session():
            # Reset temporal state at the start of each scene
            if hasattr(detector, "reset"):
                detector.reset()

            y_true: list[int] = []
            y_pred: list[int] = []
            gt_bboxes_per_frame: list[list] = []
            pred_bboxes_per_frame: list[list] = []

            for frame, gt_alert in examples:
                if preprocessor is not None:
                    frame = preprocessor.predict(frame)
                pred_alert = detector.predict(frame)

                gt_positive = gt_alert.level != FireLevel.SAFE
                pred_positive = pred_alert.level != FireLevel.SAFE
                y_true.append(1 if gt_positive else 0)
                y_pred.append(1 if pred_positive else 0)

                gt_bboxes = gt_alert.blob_features.get("bboxes", [])
                gt_bboxes_per_frame.append(gt_bboxes)

                pred_bbox = pred_alert.blob_features.get("bbox")
                pred_bboxes_per_frame.append([pred_bbox] if pred_bbox else [])

            cm = binary_confusion_matrix(y_true, y_pred)

            # IoU: only over frames where ground truth is positive
            iou_scores: list[float] = []
            for gt_boxes, pred_boxes in zip(gt_bboxes_per_frame, pred_bboxes_per_frame):
                if gt_boxes:
                    iou_scores.append(best_iou(pred_boxes, gt_boxes))
            mean_iou = float(sum(iou_scores) / len(iou_scores)) if iou_scores else 0.0

            result = ScenarioResult(
                name=session.scene,
                mode=mode,
                confusion=cm,
                mean_iou=mean_iou,
            )
            results.append(result)

            if verbose:
                print(f"  {session.scene:<40} {mode}  IoU={mean_iou:.1%}  {result}")

        return results

    # ------------------------------------------------------------------
    # §5.3.3 — Contact detection evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_contact_detection(
        detector,
        dataset: ContactFrameDataset,
        *,
        mode: str = "proc",
        verbose: bool = True,
    ) -> list[ScenarioResult]:
        """Evaluate a ContactDetector per session.

        ``detector.reset()`` is called at each session boundary to clear
        rolling buffers (``MVSTGCNDetector``, ``ThermoX3DDetector``).

        Note: contact detection evaluation is deferred to the system-
        integration phase per §5.3.3 (requires on-site multi-view data).
        This method will produce meaningful results once a labeled
        ``ContactFrameDataset`` exists.
        """
        results: list[ScenarioResult] = []

        for session, examples in dataset.by_session():
            if hasattr(detector, "reset"):
                detector.reset()

            y_true: list[int] = []
            y_pred: list[int] = []

            for frames, gt_event in examples:
                pred_event = detector.predict(frames)
                y_true.append(1 if gt_event.any_contact else 0)
                y_pred.append(1 if pred_event.any_contact else 0)

            cm = binary_confusion_matrix(y_true, y_pred)
            result = ScenarioResult(name=session.scene, mode=mode, confusion=cm)
            results.append(result)

            if verbose:
                print(f"  {session.scene:<40} {mode}  {result}")

        return results

    # ------------------------------------------------------------------
    # Train + evaluate convenience wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def fit_and_evaluate(
        detector,
        train_dataset,
        test_dataset,
        *,
        preprocessor=None,
        mode: str = "proc",
        verbose: bool = True,
    ) -> tuple:
        """Train on ``train_dataset``, then evaluate on ``test_dataset``.

        Handles the three task types automatically by inspecting the dataset
        type:
          * ``FrameLevelDataset``  → human detection
          * ``FireFrameDataset``   → fire detection
          * ``ContactFrameDataset`` → contact detection

        Args:
            detector: Any fitted or unfitted algorithm matching the dataset type.
            train_dataset: Dataset used for ``detector.fit()``.  Pass ``None``
                to skip training (e.g. for rule-based detectors).
            test_dataset: Dataset used for evaluation.
            preprocessor: Optional ``Preprocessor`` applied before both
                training and evaluation.
            mode: Label string for ``ScenarioResult``.
            verbose: Print progress.

        Returns:
            ``(fitted_detector, results)`` — the trained detector and a list
            of ``ScenarioResult`` objects.
        """
        # --- Training ---
        if train_dataset is not None and detector.is_trainable:
            if verbose:
                print(f"Training {type(detector).__name__} ...")

            if isinstance(train_dataset, FireFrameDataset):
                if preprocessor:
                    frames = [preprocessor.predict(f) for f, _ in train_dataset]
                    alerts = [a for _, a in train_dataset]
                else:
                    frames = [f for f, _ in train_dataset]
                    alerts = [a for _, a in train_dataset]
                detector.fit(frames, alerts)

            elif isinstance(train_dataset, ContactFrameDataset):
                if preprocessor:
                    frame_seqs = [
                        tuple(preprocessor.predict(f) for f in triplet)
                        for triplet, _ in train_dataset
                    ]
                else:
                    frame_seqs = [triplet for triplet, _ in train_dataset]
                events = [e for _, e in train_dataset]
                detector.fit(frame_seqs, events)

            else:  # FrameLevelDataset — human detection
                if preprocessor:
                    examples = [
                        (preprocessor.predict(f), dets)
                        for f, dets in train_dataset
                    ]
                else:
                    examples = list(train_dataset)
                detector.fit(examples)

            if verbose:
                print(f"Training complete.")

        # --- Evaluation ---
        if isinstance(test_dataset, FireFrameDataset):
            results = Trainer.evaluate_fire_detection(
                detector, test_dataset, preprocessor=preprocessor,
                mode=mode, verbose=verbose,
            )
        elif isinstance(test_dataset, ContactFrameDataset):
            results = Trainer.evaluate_contact_detection(
                detector, test_dataset, mode=mode, verbose=verbose,
            )
        else:
            results = Trainer.evaluate_human_detection(
                detector, test_dataset, preprocessor=preprocessor,
                mode=mode, verbose=verbose,
            )

        return detector, results
