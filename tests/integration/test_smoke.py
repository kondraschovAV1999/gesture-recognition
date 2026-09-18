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


def test_neural_smoke_with_optimizer_and_early_stopping(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "backend=neural",
            "neural.trainer.max_epochs=3",
            "neural.optimizer._target_=torch.optim.SGD",
            "neural.optimizer.lr=0.1",
            "reporting.formats=[json]",
        ],
    )
    # early stopping is a nested dict, set directly (CLI override of a null
    # node is awkward)
    cfg.neural.early_stopping = {
        "_target_": "lightning.pytorch.callbacks.EarlyStopping",
        "monitor": "val_loss",
        "patience": 1,
        "mode": "min",
    }
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()
    assert "accuracy" in result.metrics
    out = tmp_path / "output"
    assert (out / "evaluation" / cfg.run_id / "config" / "resolved.yaml").exists()


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
