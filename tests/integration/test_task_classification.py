from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def _classification_cfg(tmp_path, monkeypatch, backend):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "task=classification",
            "windowing.window_size_samples=40",
            f"backend={backend}",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    return cfg


def test_classical_classification_smoke(tmp_path, monkeypatch):
    runner = ExperimentRunner(_classification_cfg(tmp_path, monkeypatch, "classical"))
    result = runner.run()
    assert "accuracy" in result.metrics
    assert result.confusion_matrix.shape == (3, 3)


def test_neural_classification_smoke(tmp_path, monkeypatch):
    cfg = _classification_cfg(tmp_path, monkeypatch, "neural")
    cfg.neural.trainer.max_epochs = 1
    runner = ExperimentRunner(cfg)
    result = runner.run()
    assert "accuracy" in result.metrics
    assert result.confusion_matrix.shape == (3, 3)
