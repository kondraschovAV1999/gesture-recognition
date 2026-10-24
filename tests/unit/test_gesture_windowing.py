import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.utils.datasets import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.gesture import window_gestures


def _metadata():
    return DatasetMetadata(
        name="x",
        version="0",
        sampling_rate=200.0,
        channels=["c1", "c2"],
        class_names=["noGesture", "fist", "waveIn"],
        content_hash="h",
    )


def _trial(tid, uid, gesture, start, end, n):
    return Trial(
        tid,
        uid,
        gesture,
        start,
        end,
        np.arange(n, dtype=np.float32)[:, None].repeat(2, axis=1),
    )


def _dataset():
    # user s1: gesture trial t1 [10,30] in 40 samples (label "fist" -> 1),
    #          gesture trial t2 [0,60] in 60 samples (label "waveIn" -> 2)
    # user s2: rest trial t3 ("noGesture" -> 0), dropped in classification
    return Dataset(
        metadata=_metadata(),
        users=[
            User(
                "s1",
                [
                    _trial("t1", "s1", "fist", 10, 30, 40),
                    _trial("t2", "s1", "waveIn", 0, 60, 60),
                ],
            ),
            User("s2", [_trial("t3", "s2", "noGesture", -1, -1, 30)]),
        ],
    )


def _all_rest_dataset():
    return Dataset(
        metadata=_metadata(),
        users=[
            User(
                "s1",
                [
                    _trial("t1", "s1", "noGesture", -1, -1, 40),
                    _trial("t2", "s1", "noGesture", -1, -1, 60),
                    _trial("t3", "s2", "noGesture", -1, -1, 30),
                ],
            ),
        ],
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
    # trial t1 gesture [10,30] = 20 samples < ws 25 -> padded to 25
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    assert wd.X[0].shape[0] == 25
    assert np.all(wd.X[0, 20:] == 0)  # trailing padding is zeros


def test_long_gesture_is_truncated():
    # trial t2 gesture [0,60] = 60 samples > ws 25 -> truncated to first 25
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    assert wd.X[1].shape[0] == 25
    assert np.array_equal(wd.X[1], _dataset().trials[1].emg[:25])


def test_exact_length_gesture_unchanged():
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=60, stride_samples=10)
    )
    assert wd.X[1].shape[0] == 60
    assert np.array_equal(wd.X[1], _dataset().trials[1].emg)


def test_metadata_carries_gesture_bounds_and_ids():
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    assert wd.metadata.window_start.tolist() == [10, 0]
    assert wd.metadata.window_end.tolist() == [30, 60]
    assert wd.metadata.window_ids.tolist() == ["s1_t1_gesture", "s1_t2_gesture"]


def test_all_rest_trials_returns_empty_dataset():
    wd = window_gestures(
        _all_rest_dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    assert wd.X.shape == (0, 25, 2)
    assert wd.y.shape == (0,)


from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import get_windowing_strategy


def test_strategy_routes_by_name():
    ds = _dataset()
    cfg = WindowingConfig(window_size_samples=25, stride_samples=10)
    cls = get_windowing_strategy("classification").window(ds, cfg, signal_steps=None)
    rec = get_windowing_strategy("recognition").window(ds, cfg, signal_steps=None)
    # classification: one window per gesture trial (2), rest dropped
    assert cls.X.shape[0] == 2
    assert cls.metadata.window_ids.tolist() == ["s1_t1_gesture", "s1_t2_gesture"]
    # recognition: sliding windows over all trials (2 + 4 + 1 = 7)
    assert rec.X.shape[0] == 7
    assert rec.metadata.window_ids[0].endswith("_window0")
