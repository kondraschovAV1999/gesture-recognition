import json

import mlflow

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def _write_user(root, subdir, user_id, n_samples=3, length=120):
    """Minimal EMG-EPN612-like user JSON with a partial gesture segment."""
    d = root / subdir / f"user{user_id}"
    d.mkdir(parents=True, exist_ok=True)
    gestures = ["noGesture", "fist", "open"]

    def make_samples(n, labeled, with_gt):
        samples = {}
        for i in range(n):
            g = gestures[i % len(gestures)]
            s = {
                "emg": {
                    f"ch{c}": [float(c + j % 5) for j in range(length)]
                    for c in range(1, 9)
                }
            }
            if labeled:
                s["gestureName"] = g
            if with_gt and labeled and g != "noGesture":
                s["groundTruthIndex"] = [30, 90]
            samples[f"idx_{i + 1}"] = s
        return samples

    doc = {
        "generalInfo": {
            "samplingFrequencyInHertz": 200,
            "myoPredictionLabel": {
                "noGesture": 0,
                "fist": 1,
                "waveIn": 2,
                "waveOut": 3,
                "open": 4,
                "pinch": 5,
            },
        },
        "userInfo": {
            "name": f"user{user_id}",
            "age": 20 + user_id,
            "gender": "man",
            "occupation": "student",
            "ethnicGroup": "latin",
            "handedness": "right",
            "ArmDamage": "False",
            "distanceFromElbowToMyoInCm": 6,
            "distanceFromElbowToUlnaInCm": 24,
            "armPerimeterInCm": 23,
            "date": "30-Oct-2019",
        },
        "trainingSamples": make_samples(n_samples, True, True),
        "testingSamples": make_samples(n_samples, True, False),
    }
    (d / f"user{user_id}.json").write_text(json.dumps(doc))


def _artifacts(runner):
    client = mlflow.tracking.MlflowClient(tracking_uri=runner.cfg.tracking.tracking_uri)
    return [a.path for a in client.list_artifacts(runner.tracker.run_id)]


def test_gesture_crop_and_user_subset_integration(tmp_path, monkeypatch):
    for uid in range(1, 4):
        _write_user(tmp_path, "trainingJSON", uid)
    for uid in range(1, 4):
        _write_user(tmp_path, "testingJSON", uid)

    cfg = load_config(
        mode="compose",
        overrides=["split.strategy=random", "reporting.formats=[json]"],
    )
    cfg.dataset.path = str(tmp_path)
    cfg.dataset.loader["max_train_users"] = 2
    cfg.dataset.loader["max_test_users"] = 2
    cfg.gesture_crop.pad_samples = 10  # crop actually shortens these short trials
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))

    runner = ExperimentRunner(cfg)
    result = runner.run()

    assert "accuracy" in result.metrics
    # only 2 train + 2 test users were loaded
    out = tmp_path / "output" / "evaluation" / cfg.run_id
    assert (out / "signal_processing_comparison.png").exists()
    assert "confusion_matrix_test.png" in _artifacts(runner)
