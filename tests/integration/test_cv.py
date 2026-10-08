import mlflow

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_cv_kfold_smoke(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "baseline=none",
            "cv=kfold",
            "cv.n_splits=3",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()

    assert "accuracy" in result.metrics
    assert "accuracy_std" in result.metrics
    out = tmp_path / "output" / "evaluation" / cfg.run_id
    assert (out / "report.json").exists()

    client = mlflow.tracking.MlflowClient(tracking_uri=cfg.tracking.tracking_uri)
    artifacts = [a.path for a in client.list_artifacts(runner.tracker.run_id)]
    assert any(a.startswith("confusion_matrix_fold0_test") for a in artifacts)
