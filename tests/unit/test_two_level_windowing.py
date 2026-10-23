import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)
from hand_gesture_recognition.data.windowing.strategy import (
    GroupedWindowedPartitions,
    get_windowing_strategy,
)


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


def test_two_level_compositional():
    cfg = WindowingConfig(
        mode="two_level",
        window_size_samples=20,
        stride_samples=10,
        big_window_size_samples=60,
        big_stride_samples=30,
    )
    s = get_windowing_strategy("two_level")
    g = s.window(_ds(), cfg, signal_steps=[])  # Phase 1: returns GroupedWindowedDataset
    # big windows: t1 -> 2, t2 -> 2 => 4
    assert g.big.y.tolist() == [1, 1, 0, 0]
    assert g.small_windows[0].shape == (5, 20, 2)
    assert g.big.X.shape == (4, 60, 2)
