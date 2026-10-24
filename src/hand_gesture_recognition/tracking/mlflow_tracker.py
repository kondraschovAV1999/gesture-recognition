from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Silence mlflow's per-run "agent hint" banner (must be set before `import mlflow`).
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow

from hand_gesture_recognition.configuration.schema import TrackingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.pipeline.evaluation.base import EvaluationResult


class MlflowTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.run_id: Optional[str] = None
        # mlflow 3.x gates the local file store (e.g. a bare `mlruns/` dir)
        # behind MLFLOW_ALLOW_FILE_STORE; the tracker owns MLflow setup, so it
        # enables the file store when a scheme-less (local path) URI is used.
        if "://" not in config.tracking_uri:
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    def start(self, run_id: str) -> None:
        # System metrics (CPU/memory/disk) require `psutil`; enable only when
        # available so a missing dependency never breaks the run.
        import importlib.util

        if importlib.util.find_spec("psutil") is not None:
            try:
                mlflow.enable_system_metrics_logging()
            except Exception:
                pass
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

    def log_results(
        self, result: EvaluationResult, partition: Optional[str] = None
    ) -> None:
        prefix = f"{partition}_" if partition is not None else ""
        if result.metrics:
            mlflow.log_metrics({f"{prefix}{k}": v for k, v in result.metrics.items()})
        # Per-class precision/recall/F1 + confusion-matrix components are only
        # logged for the primary (test) result; train/val rely on their CMs.
        if partition is None:
            for cls, cls_metrics in result.per_class.items():
                mlflow.log_metrics(
                    {
                        f"precision_{cls}": cls_metrics.get("precision", 0.0),
                        f"recall_{cls}": cls_metrics.get("recall", 0.0),
                        f"f1_{cls}": cls_metrics.get("f1", 0.0),
                        f"tp_{cls}": cls_metrics.get("tp", 0),
                        f"fp_{cls}": cls_metrics.get("fp", 0),
                        f"tn_{cls}": cls_metrics.get("tn", 0),
                        f"fn_{cls}": cls_metrics.get("fn", 0),
                    }
                )

    def log_confusion_matrix(
        self,
        result: EvaluationResult,
        class_names: Optional[List[str]] = None,
        partition: str = "test",
    ) -> None:
        """Log the confusion matrix as heatmap figures (raw counts + normalized)."""
        cm = result.confusion_matrix
        base = f"confusion_matrix_{partition}"
        self._log_cm_figure(cm, class_names, f"{base}.png", normalize=False)
        self._log_cm_figure(cm, class_names, f"{base}_normalized.png", normalize=True)

    @staticmethod
    def _log_cm_figure(cm, class_names, artifact_name: str, normalize: bool) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        if cm.size == 0:
            return

        n = cm.shape[0]
        labels = class_names if class_names is not None else [str(i) for i in range(n)]

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            display = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
            title = "Confusion matrix (row-normalized)"
            fmt = "{:.0%}"
        else:
            display = cm
            title = "Confusion matrix (counts)"
            fmt = "{:d}"

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(display, cmap="Blues", vmin=0, vmax=1 if normalize else None)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        vmax = display.max() if display.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                color = "white" if display[i, j] > vmax / 2 else "black"
                ax.text(
                    j,
                    i,
                    fmt.format(display[i, j]),
                    ha="center",
                    va="center",
                    color=color,
                )
        fig.colorbar(im)
        fig.tight_layout()
        mlflow.log_figure(fig, artifact_name)
        plt.close(fig)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def finish(self, status: str) -> None:
        mlflow.end_run(status=status)
