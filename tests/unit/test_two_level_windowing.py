import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.two_level import (
    window_big_windows,
    window_dataset_two_level,
)


def _dataset():
    # trial 0: gesture [0,100] label 1; trial 1: rest label 0
    return Dataset(
        trials=[
            np.arange(100, dtype=np.float32)[:, None].repeat(2, axis=1),
            np.zeros((100, 2), dtype=np.float32),
        ],
        labels=np.array([1, 0]),
        subject_ids=np.array(["s1", "s1"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.array([0, -1]),
        gesture_end=np.array([100, -1]),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["rest", "fist"],
            content_hash="h",
        ),
    )


def _config():
    return WindowingConfig(
        mode="two_level",
        window_size_samples=20,
        stride_samples=10,
        big_window_size_samples=60,
        big_stride_samples=30,
    )


def test_big_and_small_window_counts():
    twd = window_dataset_two_level(_dataset(), _config(), signal_steps=[])
    # trial 0 (len 100): big windows [0,60], [30,90] -> 2; trial 1 (rest): 2 -> 4 total
    assert len(twd.big_windows) == 4
    assert twd.y.tolist() == [1, 1, 0, 0]
    # each 60-sample big window -> (60-20)//10 + 1 = 5 small windows
    assert twd.small_windows[0].shape == (5, 20, 2)
    assert twd.big_windows.shape == (4, 60, 2)


def test_big_window_labels_and_metadata():
    twd = window_dataset_two_level(_dataset(), _config(), signal_steps=[])
    assert twd.metadata.window_start.tolist() == [0, 30, 0, 30]
    assert twd.metadata.window_end.tolist() == [60, 90, 60, 90]
    assert twd.metadata.window_ids.tolist() == [
        "s1_t1_big0",
        "s1_t1_big1",
        "s1_t2_big0",
        "s1_t2_big1",
    ]
    assert twd.metadata.trial_ids.tolist() == ["t1", "t1", "t2", "t2"]


def test_signal_steps_applied_per_big_window():
    class Scale:
        def transform(self, x):
            return x * 2.0

    twd = window_dataset_two_level(_dataset(), _config(), signal_steps=[Scale()])
    assert np.allclose(twd.big_windows[0], _dataset().trials[0][0:60] * 2.0)


def test_drop_rest_big_windows():
    ds = _dataset()
    ds.metadata.drop_rest_windows = True
    twd = window_dataset_two_level(ds, _config(), signal_steps=[])
    assert twd.y.tolist() == [1, 1]
    assert len(twd.big_windows) == 2


def test_window_big_windows_flat():
    wd = window_big_windows(_dataset(), _config(), signal_steps=[])
    assert wd.X.shape == (4, 60, 2)
    assert wd.y.tolist() == [1, 1, 0, 0]
    assert wd.metadata.window_ids[0] == "s1_t1_big0"


def test_empty_input_returns_empty():
    ds = _dataset()
    ds.gesture_start = np.array([-1, -1])
    ds.gesture_end = np.array([-1, -1])
    twd = window_dataset_two_level(ds, _config(), signal_steps=[])
    assert twd.big_windows.shape[0] == 0
    assert twd.y.shape == (0,)
