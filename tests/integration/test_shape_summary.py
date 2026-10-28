from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_run_writes_shape_summary(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "baseline=none",
            "split.strategy=random",
            "backend=classical",
            "windowing=default",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    runner.run()
    out = tmp_path / "output" / "evaluation" / cfg.run_id
    assert (out / "shape_summary.json").exists()
