import numpy as np

from hand_gesture_recognition.utils.analytics.qc import run_quality_checks
from hand_gesture_recognition.utils.analytics.stats import dataset_summary
from hand_gesture_recognition.utils.datasets import Dataset, DatasetMetadata, Trial


def _ds():
    meta = DatasetMetadata(
        name="x",
        version="0",
        sampling_rate=200.0,
        channels=["c1", "c2"],
        class_names=["a", "b"],
        content_hash="h",
    )
    trials = [
        Trial(
            trial_id="t1",
            user_id="s1",
            gesture_name="a",
            start=0,
            end=100,
            emg=np.ones((100, 2)),
        ),
        Trial(
            trial_id="t2",
            user_id="s2",
            gesture_name="b",
            start=0,
            end=100,
            emg=np.full((100, 2), np.nan),
        ),
    ]
    return Dataset.from_trials(trials, meta)


def test_dataset_summary_counts():
    s = dataset_summary(_ds())
    assert s["n_trials"] == 2
    assert s["n_subjects"] == 2
    assert s["n_channels"] == 2
    assert s["class_distribution"] == {"a": 1, "b": 1}


def test_qc_detects_nan():
    checks = run_quality_checks(_ds())
    msgs = " ".join(c["message"] for c in checks)
    assert any(c["check"] == "nan_inf" for c in checks)
    assert "NaN" in msgs
