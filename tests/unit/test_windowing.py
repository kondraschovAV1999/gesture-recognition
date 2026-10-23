import numpy as np
from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.windowing.base import window_emg


def _cfg(ws, stride):
    return WindowingConfig(
        window_size_samples=ws, stride_samples=stride, padding="none"
    )


def test_window_emg_dumb_slide():
    emg = np.arange(10, dtype=np.float32)[:, None]
    X, y, meta = window_emg(emg, 1, _cfg(4, 2), "s1", "t1", "s1_t1")
    assert X.shape == (4, 4, 1)  # windows at 0,2,4,6
    assert y.tolist() == [1, 1, 1, 1]
    assert meta.window_start.tolist() == [0, 2, 4, 6]
    assert meta.subject_ids.tolist() == ["s1"] * 4


def test_window_emg_rest_detection():
    emg = np.ones((10, 1), np.float32)
    X, y, meta = window_emg(
        emg, 5, _cfg(4, 2), "s1", "t1", "s1_t1", rest_bounds=(4, 10)
    )
    # window at 0 is rest (ends at 4, not overlapping [4,10]); window at 2 overlaps
    assert y.tolist() == [0, 5, 5, 5]
    assert meta.window_start.tolist() == [0, 2, 4, 6]


def test_window_emg_drop_rest():
    emg = np.ones((10, 1), np.float32)
    X, y, meta = window_emg(
        emg, 5, _cfg(4, 2), "s1", "t1", "s1_t1", rest_bounds=(4, 10), drop_rest=True
    )
    assert y.tolist() == [5, 5, 5]
    assert meta.window_start.tolist() == [2, 4, 6]
