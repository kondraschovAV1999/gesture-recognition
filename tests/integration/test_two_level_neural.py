from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_neural_two_level_smoke(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "backend=neural",
            "windowing=two_level",
            "windowing.big_window_size_samples=60",
            "windowing.big_stride_samples=30",
            "windowing.window_size_samples=20",
            "windowing.stride_samples=10",
            "neural.trainer.max_epochs=1",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()
    assert "accuracy" in result.metrics
    assert result.confusion_matrix.shape == (3, 3)
