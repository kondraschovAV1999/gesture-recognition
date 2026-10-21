import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.gesture import window_gestures


def _dataset():
    # trial 0: gesture [10,30] in a 40-sample trial (label 1)
    # trial 1: gesture [0,60] in a 60-sample trial (label 2)
    # trial 2: rest (label 0)
    return Dataset(
        trials=[
            np.arange(40, dtype=np.float32)[:, None].repeat(2, axis=1),
            np.arange(60, dtype=np.float32)[:, None].repeat(2, axis=1),
            np.arange(30, dtype=np.float32)[:, None].repeat(2, axis=1),
        ],
        labels=np.array([1, 2, 0]),
        subject_ids=np.array(["s1", "s1", "s2"]),
        trial_ids=np.array(["t1", "t2", "t3"]),
        gesture_start=np.array([10, 0, -1]),
        gesture_end=np.array([30, 60, -1]),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["rest", "fist", "waveIn"],
            content_hash="h",
        ),
    )


def test_one_sample_per_gesture_trial():
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    # rest trial dropped; two gesture trials -> two samples
    assert wd.X.shape == (2, 25, 2)
    assert wd.y.tolist() == [1, 2]
    assert wd.metadata.subject_ids.tolist() == ["s1", "s1"]
    assert wd.metadata.trial_ids.tolist() == ["t1", "t2"]


def test_short_gesture_is_zero_padded():
    # trial 0 gesture [10,30] = 20 samples < ws 25 -> padded to 25
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    assert wd.X[0].shape[0] == 25
    assert np.all(wd.X[0, 20:] == 0)  # trailing padding is zeros


def test_long_gesture_is_truncated():
    # trial 1 gesture [0,60] = 60 samples > ws 25 -> truncated to first 25
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    assert wd.X[1].shape[0] == 25
    assert np.array_equal(wd.X[1], _dataset().trials[1][:25])


def test_exact_length_gesture_unchanged():
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=60, stride_samples=10)
    )
    assert wd.X[1].shape[0] == 60
    assert np.array_equal(wd.X[1], _dataset().trials[1])


def test_metadata_carries_gesture_bounds_and_ids():
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    assert wd.metadata.window_start.tolist() == [10, 0]
    assert wd.metadata.window_end.tolist() == [30, 60]
    assert wd.metadata.window_ids.tolist() == ["s1_t1_gesture", "s1_t2_gesture"]


def test_all_rest_trials_returns_empty_dataset():
    ds = _dataset()
    ds.gesture_start = np.array([-1, -1, -1])
    ds.gesture_end = np.array([-1, -1, -1])
    wd = window_gestures(ds, WindowingConfig(window_size_samples=25, stride_samples=10))
    assert wd.X.shape == (0, 25, 2)
    assert wd.y.shape == (0,)


from hand_gesture_recognition.data.windowing.dispatch import build_windowed_dataset


def test_dispatcher_routes_by_task():
    ds = _dataset()
    cfg = WindowingConfig(window_size_samples=25, stride_samples=10)
    cls = build_windowed_dataset(ds, cfg, "classification")
    rec = build_windowed_dataset(ds, cfg, "recognition")
    # classification: one window per gesture trial (2), rest dropped
    assert cls.X.shape[0] == 2
    assert cls.metadata.window_ids.tolist() == ["s1_t1_gesture", "s1_t2_gesture"]
    # recognition: sliding windows over all trials (2 + 4 + 1 = 7)
    assert rec.X.shape[0] == 7
    assert rec.metadata.window_ids[0].endswith("_window0")
