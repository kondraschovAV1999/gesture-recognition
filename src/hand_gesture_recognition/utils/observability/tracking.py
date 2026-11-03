from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

# Silence mlflow's per-run "agent hint" banner (must be set before `import mlflow`).
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow

from hand_gesture_recognition.configuration.schema import TrackingConfig
from hand_gesture_recognition.configuration.schema import Dataset
from hand_gesture_recognition.configuration.schema import EvaluationResult
from hand_gesture_recognition.utils.paths import resolve_tracking_uri


def is_primary_process() -> bool:
    """Return True when this process should write to the shared MLflow store.

    Multi-process launches (``torchrun``/``mpirun``/``srun``) run the same
    pipeline on every rank. Only one process should own the MLflow run so that
    identical metrics (e.g. the cross-validation aggregate) are not written
    several times in the same millisecond — which collides with the SQLite
    ``metrics`` unique index ``(key, timestamp, step, run_uuid, value, is_nan)``.
    """
    # The checks compose (AND) so a job that is BOTH a Slurm array task and a
    # multi-rank launch (an array of DDP jobs) resolves to exactly one primary:
    # the lowest array task *and* rank 0 within it.
    primary = True
    # Slurm job array: every task is its own job (each sees ``SLURM_PROCID=0``),
    # so ``SLURM_ARRAY_TASK_ID`` (compared to the array min index, so 1-based
    # arrays like ``--array=1-4`` also resolve to one primary) is the cross-task
    # discriminator. Checked first because torchrun-style rank vars are often
    # "0" in every array task and would otherwise mask this check.
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_min = os.environ.get("SLURM_ARRAY_TASK_MIN", "0")
        primary = primary and os.environ["SLURM_ARRAY_TASK_ID"] == task_min
    # torchrun / accelerate / Lightning set RANK (global) and LOCAL_RANK.
    for var in ("RANK", "LOCAL_RANK"):
        if var in os.environ:
            primary = primary and os.environ[var] == "0"
            break
    # OpenMPI.
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        primary = primary and os.environ["OMPI_COMM_WORLD_RANK"] == "0"
    # A plain Slurm multi-task step (``srun --ntasks=4``) is ranked by
    # ``SLURM_PROCID``.
    if "SLURM_PROCID" in os.environ:
        primary = primary and os.environ["SLURM_PROCID"] == "0"
    # torch.distributed (best-effort; torch may be absent in classical-only envs).
    try:
        import torch.distributed as dist
    except Exception:  # pragma: no cover - defensive against missing torch
        dist = None
    if dist is not None and dist.is_available() and dist.is_initialized():
        primary = primary and dist.get_rank() == 0
    # Single-process runs always write.
    return primary


class MlflowTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.run_id: Optional[str] = None
        # Only the primary process in a multi-rank launch owns the MLflow run;
        # non-primary processes skip all tracking writes.
        self._primary = is_primary_process()
        # Exact ``(key, value)`` pairs already logged this run. ``step`` is
        # always the MLflow default (0) in this tracker, so a re-log of an
        # identical value is a true duplicate and is dropped instead of
        # tripping SQLite's unique index on the metrics table.
        self._logged_metrics: Set[Tuple[str, float]] = set()
        # Anchor relative local stores (a bare `mlruns/` dir or a relative
        # `sqlite:///mlflow.db`) to the project root so results don't land in a
        # subdirectory depending on the process's current working directory.
        uri = resolve_tracking_uri(config.tracking_uri)
        # mlflow 3.x gates the local file store (e.g. a bare `mlruns/` dir)
        # behind MLFLOW_ALLOW_FILE_STORE; the tracker owns MLflow setup, so it
        # enables the file store when a scheme-less (local path) URI is used.
        if "://" not in uri:
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(uri)
        # `set_experiment` performs a get-or-create against the store; only the
        # primary process should touch the shared SQLite file to avoid
        # concurrent-writer races from the other ranks/tasks.
        if self._primary:
            mlflow.set_experiment(config.experiment_name)

    def start(self, run_id: str) -> None:
        if not self._primary:
            return
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
        if not self._primary:
            return
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if not self._primary:
            return
        deduped = self._dedupe_metrics(metrics)
        if deduped:
            mlflow.log_metrics(deduped)

    def _dedupe_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Drop ``(key, value)`` pairs already logged this run.

        MLflow's SQLite backend rejects a second insert of an identical
        ``(key, timestamp, step, value)`` within the same millisecond. Since
        ``step`` is always the default (0) here, an identical ``(key, value)``
        is a true duplicate (e.g. the same CV aggregate logged by several
        processes sharing a run) and is safely skipped.
        """
        if not metrics:
            return metrics
        fresh: Dict[str, float] = {}
        for key, value in metrics.items():
            # Coerce so numpy scalars hash/compare identically to Python floats
            # (the same logical value logged twice must be recognized as a dup).
            value = float(value)
            pair = (key, value)
            if pair in self._logged_metrics:
                continue
            self._logged_metrics.add(pair)
            fresh[key] = value
        return fresh

    def log_dataset(self, dataset: Dataset) -> None:
        if not self._primary:
            return
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
        if not self._primary:
            return
        prefix = f"{partition}_" if partition is not None else ""
        if result.metrics:
            self.log_metrics({f"{prefix}{k}": v for k, v in result.metrics.items()})
        # Per-class precision/recall/F1 + confusion-matrix components are only
        # logged for the primary (test) result; train/val rely on their CMs.
        if partition is None:
            for cls, cls_metrics in result.per_class.items():
                self.log_metrics(
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
        if not self._primary:
            return
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
        if not self._primary:
            return
        mlflow.log_artifact(local_path)

    def finish(self, status: str) -> None:
        if not self._primary:
            return
        mlflow.end_run(status=status)
