from __future__ import annotations

from typing import Any, Dict, Optional

import mlflow

from hand_gesture_recognition.configuration.schema import TrackingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.evaluation.base import EvaluationResult


class MlflowTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.run_id: Optional[str] = None
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    def start(self, run_id: str) -> None:
        mlflow.start_run(run_name=run_id)
        self.run_id = mlflow.active_run().info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_dataset(self, dataset: Dataset) -> None:
        mlflow.log_params(
            {
                "dataset_name": dataset.metadata.name,
                "dataset_version": dataset.metadata.version,
                "dataset_content_hash": dataset.metadata.content_hash,
                "sampling_rate": dataset.metadata.sampling_rate,
            }
        )

    def log_results(self, result: EvaluationResult) -> None:
        mlflow.log_metrics(result.metrics)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def finish(self, status: str) -> None:
        mlflow.end_run(status=status)
