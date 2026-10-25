import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.configuration.schema import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)
from hand_gesture_recognition.configuration.schema import GroupedWindowedPartitions
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import get_windowing_strategy


def _ds():
    meta = DatasetMetadata("x", "0", 200.0, ["c1", "c2"], ["noGesture", "fist"], "h")
    u = User(
        "s1",
        [
            Trial(
                "t1",
                "s1",
                "fist",
                0,
                100,
                np.arange(100, dtype=np.float32)[:, None].repeat(2, axis=1),
            ),
            Trial("t2", "s1", "noGesture", -1, -1, np.zeros((100, 2), np.float32)),
        ],
    )
    return Dataset(meta, [u])


def _cfg():
    return WindowingConfig(
        mode="two_level",
        window_size_samples=20,
        stride_samples=10,
        big_window_size_samples=60,
        big_stride_samples=30,
    )


def _short_ds():
    meta = DatasetMetadata("x", "0", 200.0, ["c1", "c2"], ["noGesture", "fist"], "h")
    u = User(
        "s1",
        [Trial("t1", "s1", "fist", 0, 10, np.zeros((10, 2), np.float32))],
    )
    return Dataset(meta, [u])


def test_two_level_compositional():
    s = get_windowing_strategy("two_level")
    g = s.window(
        _ds(), _cfg(), signal_steps=[]
    )  # Phase 1: returns GroupedWindowedDataset
    # big windows: t1 -> 2, t2 -> 2 => 4
    assert g.big.y.tolist() == [1, 1, 0, 0]
    assert g.small_windows[0].shape == (5, 20, 2)
    assert g.big.X.shape == (4, 60, 2)


def test_signal_steps_applied_per_big_window():
    class Scale:
        def transform(self, x):
            return x * 2.0

    ds = _ds()
    t1 = ds.trials[0]
    s = get_windowing_strategy("two_level")
    g = s.window(ds, _cfg(), signal_steps=[Scale()])
    assert np.allclose(g.big.X[0], t1.emg[0:60] * 2.0)


def test_big_window_metadata():
    s = get_windowing_strategy("two_level")
    g = s.window(_ds(), _cfg(), signal_steps=[])
    assert g.big.metadata.window_start.tolist() == [0, 30, 0, 30]
    assert g.big.metadata.window_end.tolist() == [60, 90, 60, 90]
    assert g.big.metadata.window_ids.tolist() == [
        "s1_t1_big0",
        "s1_t1_big1",
        "s1_t2_big0",
        "s1_t2_big1",
    ]
    assert g.big.metadata.trial_ids.tolist() == ["t1", "t1", "t2", "t2"]


def test_big_window_id_joinable():
    s = get_windowing_strategy("two_level")
    g = s.window(_ds(), _cfg(), signal_steps=[])
    assert g.big_window_id.tolist() == [
        "s1_t1_big0",
        "s1_t1_big1",
        "s1_t2_big0",
        "s1_t2_big1",
    ]
    # big_window_id must be joinable to big.metadata.window_ids exactly
    assert g.big_window_id.tolist() == g.big.metadata.window_ids.tolist()


def test_empty_input_returns_empty():
    cfg = WindowingConfig(
        mode="two_level",
        window_size_samples=20,
        stride_samples=10,
        big_window_size_samples=60,
        big_stride_samples=30,
        padding="none",
    )
    s = get_windowing_strategy("two_level")
    g = s.window(_short_ds(), cfg, signal_steps=[])
    assert g.big.y.shape == (0,)
    assert len(g.small_windows) == 0
