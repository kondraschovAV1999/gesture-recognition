import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.base import window_dataset


def _ds(lengths=(100, 60), labels=(0, 1), subjects=("s1", "s2")):
    trials = [
        np.arange(l, dtype=np.float32)[:, None].repeat(2, axis=1) for l in lengths
    ]
    return Dataset(
        trials=trials,
        labels=np.array(labels),
        subject_ids=np.array(subjects),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.zeros(len(trials), dtype=int),
        gesture_end=np.array([t.shape[0] for t in trials], dtype=int),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["a", "b"],
            content_hash="h",
        ),
    )


def test_window_counts_and_shapes():
    ds = _ds(lengths=(100, 100), labels=(0, 1), subjects=("s1", "s2"))
    cfg = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_dataset(ds, cfg)
    # each 100-sample trial -> ceil((100-40)/10)+1 = 7 windows
    assert wd.X.shape == (14, 40, 2)
    assert wd.y.shape == (14,)
    assert len(wd.metadata.window_start) == 14


def test_window_labels_and_subject_metadata_carry_through():
    ds = _ds(lengths=(100, 100), labels=(0, 1), subjects=("s1", "s2"))
    cfg = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_dataset(ds, cfg)
    assert wd.y[:7].tolist() == [0] * 7
    assert wd.y[7:].tolist() == [1] * 7
    assert set(wd.metadata.subject_ids.tolist()) == {"s1", "s2"}


def test_window_seconds_derived():
    ds = _ds(lengths=(100, 100), labels=(0, 1))
    cfg = WindowingConfig(
        window_size_seconds=0.2, stride_seconds=0.05
    )  # 40 / 10 @200Hz
    wd = window_dataset(ds, cfg)
    assert wd.X.shape[1] == 40
