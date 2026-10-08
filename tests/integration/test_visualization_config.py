from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_visualization_config_produces_selected_plots(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "baseline=none",
            "split.strategy=random",
            "reporting.formats=[json]",
            "visualization.selection.gestures=[class_0]",
            "visualization.selection.channels=[1,2]",
            "visualization.stage_progression=true",
            "visualization.feature_over_windows.features=[rms]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    runner.run()
    out = tmp_path / "output" / "evaluation" / cfg.run_id
    assert (out / "raw_signal_class_0.png").exists()
    assert (out / "stage_progression_class_0.png").exists()
    assert (out / "feature_over_windows_class_0.png").exists()
    # selection limited to class_0, so class_1 artifacts are absent
    assert not (out / "raw_signal_class_1.png").exists()
