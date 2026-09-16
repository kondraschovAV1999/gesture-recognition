import numpy as np

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_replay_reproduces_predictions(tmp_path, monkeypatch):
    # compose mode
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "out1"))
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "reporting.formats=[json]",
        ],
    )
    cfg.run_id = "replay-test"
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns1")
    result1 = ExperimentRunner(cfg).run()

    # replay mode from the persisted resolved config
    resolved = (
        tmp_path / "out1" / "evaluation" / "replay-test" / "config" / "resolved.yaml"
    )
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "out2"))
    cfg2 = load_config(mode="replay", resolved_path=str(resolved))
    cfg2.tracking.tracking_uri = str(tmp_path / "mlruns2")
    result2 = ExperimentRunner(cfg2).run()

    assert result1.metrics == result2.metrics
    assert np.array_equal(result1.confusion_matrix, result2.confusion_matrix)
