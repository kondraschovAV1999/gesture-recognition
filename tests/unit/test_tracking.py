import mlflow
import pytest

from hand_gesture_recognition.configuration.schema import TrackingConfig
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
