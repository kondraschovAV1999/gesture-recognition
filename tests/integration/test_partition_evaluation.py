import os

import mlflow

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def _artifacts(runner):
    client = mlflow.tracking.MlflowClient(tracking_uri=runner.cfg.tracking.tracking_uri)
    return [a.path for a in client.list_artifacts(runner.tracker.run_id)]


def test_classical_partition_evaluation(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()

    assert "accuracy" in result.metrics  # returned result is the test result
    out = tmp_path / "output" / "evaluation" / cfg.run_id
    assert (out / "raw_signal_class_0.png").exists()
    assert (out / "predictions_on_signal_val.png").exists()
    assert (out / "predictions_on_signal_test.png").exists()
    assert (out / "signal_processing_comparison.png").exists()

    artifacts = _artifacts(runner)
    for name in (
        "confusion_matrix_train.png",
        "confusion_matrix_val.png",
        "confusion_matrix_test.png",
        "confusion_matrix_train_normalized.png",
        "confusion_matrix_val_normalized.png",
        "confusion_matrix_test_normalized.png",
    ):
        assert name in artifacts, name


def test_neural_partition_evaluation(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "backend=neural",
            "neural.trainer.max_epochs=1",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()

    assert "accuracy" in result.metrics
    artifacts = _artifacts(runner)
    assert "confusion_matrix_train.png" in artifacts
    assert "confusion_matrix_val.png" in artifacts
    assert "confusion_matrix_test.png" in artifacts
