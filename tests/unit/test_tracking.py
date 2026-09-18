import mlflow
import numpy as np
import pytest

from hand_gesture_recognition.configuration.schema import TrackingConfig
from hand_gesture_recognition.evaluation.base import EvaluationResult
from hand_gesture_recognition.tracking.mlflow_tracker import MlflowTracker


def test_tracker_logs_params_and_metrics(tmp_path):
    cfg = TrackingConfig(tracking_uri=str(tmp_path / "mlruns"), experiment_name="t")
    t = MlflowTracker(cfg)
    t.start("run-1")
    t.log_params({"seed": 42, "window_size": 40})
    t.log_metrics({"accuracy": 0.9})
    t.finish("FINISHED")
    # assert the run was recorded
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    run = client.get_run(t.run_id)
    assert run.data.params["seed"] == "42"
    assert run.data.metrics["accuracy"] == 0.9
    assert run.info.status == "FINISHED"


def test_tracker_marks_failed(tmp_path):
    cfg = TrackingConfig(tracking_uri=str(tmp_path / "mlruns"), experiment_name="t")
    t = MlflowTracker(cfg)
    t.start("run-2")
    t.finish("FAILED")
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    assert client.get_run(t.run_id).info.status == "FAILED"


def test_log_results_partition_prefix(tmp_path):
    cfg = TrackingConfig(tracking_uri=str(tmp_path / "mlruns"), experiment_name="t")
    t = MlflowTracker(cfg)
    t.start("run-3")
    res = EvaluationResult(run_id="r", dataset_id="d", metrics={"accuracy": 0.8})
    t.log_results(res, partition="val")
    t.finish("FINISHED")
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    assert client.get_run(t.run_id).data.metrics["val_accuracy"] == 0.8


def test_log_results_no_partition_logs_per_class(tmp_path):
    cfg = TrackingConfig(tracking_uri=str(tmp_path / "mlruns"), experiment_name="t")
    t = MlflowTracker(cfg)
    t.start("run-3b")
    res = EvaluationResult(
        run_id="r",
        dataset_id="d",
        metrics={"accuracy": 0.9},
        per_class={
            "fist": {
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "tp": 1,
                "fp": 1,
                "tn": 1,
                "fn": 1,
            }
        },
    )
    t.log_results(res, partition=None)
    t.finish("FINISHED")
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    metrics = client.get_run(t.run_id).data.metrics
    assert metrics["accuracy"] == 0.9
    assert metrics["precision_fist"] == 0.5


def test_log_confusion_matrix_empty_skips(tmp_path):
    cfg = TrackingConfig(tracking_uri=str(tmp_path / "mlruns"), experiment_name="t")
    t = MlflowTracker(cfg)
    t.start("run-4")
    res = EvaluationResult(
        run_id="r", dataset_id="d", confusion_matrix=np.zeros((0, 0))
    )
    t.log_confusion_matrix(res, class_names=["a", "b"], partition="train")
    t.finish("FINISHED")
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    artifacts = [a.path for a in client.list_artifacts(t.run_id)]
    assert "confusion_matrix_train.png" not in artifacts
