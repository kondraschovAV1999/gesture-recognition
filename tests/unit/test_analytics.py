import numpy as np

from hand_gesture_recognition.analytics.qc import run_quality_checks
from hand_gesture_recognition.analytics.stats import dataset_summary
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata


def _ds():
    meta = DatasetMetadata(
        name="x",
        version="0",
        sampling_rate=200.0,
        channels=["c1", "c2"],
        class_names=["a", "b"],
        content_hash="h",
    )
    return Dataset(
        trials=[np.ones((100, 2)), np.full((100, 2), np.nan)],
        labels=np.array([0, 1]),
        subject_ids=np.array(["s1", "s2"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.array([0, 0]),
        gesture_end=np.array([100, 100]),
        metadata=meta,
    )


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
