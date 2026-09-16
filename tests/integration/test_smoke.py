import os

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_classical_smoke(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "reporting.formats=[json]",
        ],
    )
    # filesystem paths are set directly on the typed config (avoids Hydra
    # override parsing of Windows backslash/colon path separators)
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    runner.run()
    out = tmp_path / "output"
    assert (out / "evaluation" / cfg.run_id / "config" / "resolved.yaml").exists()
    assert (out / "evaluation" / cfg.run_id / "report.json").exists()


def test_neural_smoke(tmp_path, monkeypatch):
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
    runner.run()
    out = tmp_path / "output"
    assert (out / "evaluation" / cfg.run_id / "config" / "resolved.yaml").exists()
    assert (out / "evaluation" / cfg.run_id / "report.json").exists()
