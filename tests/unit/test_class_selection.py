import numpy as np
import pandas as pd

from hand_gesture_recognition.data.class_selection import remove_no_gesture
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata


def _dataset():
    # 4 trials: fist(1), waveIn(2), rest(0), open(4)
    return Dataset(
        trials=[
            np.zeros((50, 2), dtype=np.float32),
            np.zeros((60, 2), dtype=np.float32),
            np.zeros((40, 2), dtype=np.float32),
            np.zeros((70, 2), dtype=np.float32),
        ],
        labels=np.array([1, 2, 0, 4]),
        subject_ids=np.array(["s1", "s1", "s1", "s2"]),
        trial_ids=np.array(["t1", "t2", "t3", "t4"]),
        gesture_start=np.array([10, 20, -1, 30]),
        gesture_end=np.array([50, 60, -1, 70]),
        metadata=DatasetMetadata(
            name="epn612",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"],
            content_hash="h",
        ),
    )


def test_removes_rest_trials_and_remaps_labels():
    ds = remove_no_gesture(_dataset())
    assert ds.n_trials == 3
    assert ds.labels.tolist() == [0, 1, 2]  # fist->0, waveIn->1, open->2
    assert ds.subject_ids.tolist() == ["s1", "s1", "s2"]
    assert ds.trial_ids.tolist() == ["t1", "t2", "t4"]
    assert ds.gesture_start.tolist() == [10, 20, 30]
    assert ds.gesture_end.tolist() == [50, 60, 70]


def test_updates_class_names_and_signals_windowing():
    ds = remove_no_gesture(_dataset())
    assert ds.metadata.class_names == ["fist", "waveIn", "open"]
    assert ds.metadata.drop_rest_windows is True


def test_no_op_when_no_rest_trials():
    ds = _dataset()
    ds.labels = np.array([1, 2, 4, 4])
    ds.gesture_start = np.array([10, 20, 30, 30])
    ds.gesture_end = np.array([50, 60, 70, 70])
    out = remove_no_gesture(ds)
    assert out is ds


def test_filters_subject_info_to_remaining_subjects():
    ds = _dataset()
    ds.labels = np.array([1, 2, 0, 0])  # s2's only trial becomes rest
    ds.gesture_start = np.array([10, 20, -1, -1])
    ds.gesture_end = np.array([50, 60, -1, -1])
    ds.subject_info = pd.DataFrame({"age": [20, 21]}, index=pd.Index(["s1", "s2"]))
    out = remove_no_gesture(ds)
    assert list(out.subject_info.index) == ["s1"]
