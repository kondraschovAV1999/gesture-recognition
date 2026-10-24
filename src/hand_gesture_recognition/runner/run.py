from __future__ import annotations

import os
import uuid
from datetime import datetime

import mlflow

from hand_gesture_recognition.configuration.schema import ExperimentConfig
from hand_gesture_recognition.evaluation.base import EvaluationResult
from hand_gesture_recognition.logging_setup.config import get_logger
from hand_gesture_recognition.runner.pipeline import PipelineBuilder
from hand_gesture_recognition.tracking.mlflow_tracker import MlflowTracker


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.cfg.run_id = cfg.run_id or (
            f"{cfg.backend}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        )
        self.output_root = os.environ.get("HAND_GESTURE_RECOGNITION_OUTPUT", "output")
        self.logger = get_logger(
            "hand_gesture_recognition.runner", run_id=self.cfg.run_id
        )
        self.tracker = MlflowTracker(cfg.tracking)

    def run(self) -> EvaluationResult:
        cfg = self.cfg
        self.tracker.start(cfg.run_id)
        if cfg.backend == "classical":
            mlflow.sklearn.autolog()
        else:
            mlflow.pytorch.autolog()
        try:
            pipeline = PipelineBuilder(
                cfg, self.logger, self.tracker, self.output_root
            ).build()
            result = pipeline.run()
            self.tracker.finish("FINISHED")
            return result
        except Exception:
            self.tracker.finish("FAILED")
            raise
