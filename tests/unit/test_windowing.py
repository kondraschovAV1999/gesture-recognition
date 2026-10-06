import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.base import (
    WindowedPartitions,
    window_dataset,
)


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


def test_per_window_labels_from_gesture_bounds():
    ds = Dataset(
        trials=[
            np.zeros((50, 2), dtype=np.float32),
            np.zeros((50, 2), dtype=np.float32),
        ],
        labels=np.array([1, 0]),
        subject_ids=np.array(["s1", "s1"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.array([10, -1]),
        gesture_end=np.array([30, -1]),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["noGesture", "fist"],
            content_hash="h",
        ),
    )
    wd = window_dataset(ds, WindowingConfig(window_size_samples=10, stride_samples=10))
    # trial 0 (gesture [10,30]) -> [0,1,1,0,0]; trial 1 (rest) -> [0,0,0,0,0]
    assert wd.y.tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]


def test_window_ids_are_unique_and_traceable():
    ds = _ds(lengths=(100, 100), labels=(0, 1), subjects=("s1", "s2"))
    cfg = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_dataset(ds, cfg)
    ids = wd.metadata.window_ids.tolist()
    assert len(ids) == 14
    assert len(set(ids)) == 14  # unique
    # traceable: first trial's windows embed the trial id "t1" and subject "s1"
    assert ids[0].startswith("s1_t1_")
    assert ids[7].startswith("s2_t2_")


def test_windowed_partitions_dataclass():
    ds = _ds(lengths=(100, 100), labels=(0, 1), subjects=("s1", "s2"))
    cfg = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_dataset(ds, cfg)
    wp = WindowedPartitions(train=wd, val=wd, test=wd)
    assert wp.train is wd
    assert wp.val.X.shape[0] == 14


def test_drop_rest_windows_skips_rest():
    ds = Dataset(
        trials=[
            np.zeros((50, 2), dtype=np.float32),
            np.zeros((50, 2), dtype=np.float32),
        ],
        labels=np.array([1, 0]),
        subject_ids=np.array(["s1", "s1"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.array([10, -1]),
        gesture_end=np.array([30, -1]),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["fist", "waveIn"],
            content_hash="h",
            drop_rest_windows=True,
        ),
    )
    wd = window_dataset(ds, WindowingConfig(window_size_samples=10, stride_samples=10))
    # trial 0 (gesture [10,30]): only the two overlapping windows survive
    # trial 1 (rest): fully dropped
    assert wd.y.tolist() == [1, 1]
    assert wd.X.shape == (2, 10, 2)
    assert wd.metadata.window_ids.tolist() == ["s1_t1_window1", "s1_t1_window2"]
