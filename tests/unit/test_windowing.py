import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)
from hand_gesture_recognition.data.windowing.base import window_dataset, window_emg


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


def test_window_emg_all_rest():
    emg = np.ones((10, 1), np.float32)
    X, y, meta = window_emg(
        emg, 5, _cfg(4, 2), "s1", "t1", "s1_t1", rest_bounds=(-1, -1)
    )
    assert y.tolist() == [0, 0, 0, 0]
    assert X.shape == (4, 4, 1)


def test_window_emg_drop_rest():
    emg = np.ones((10, 1), np.float32)
    X, y, meta = window_emg(
        emg, 5, _cfg(4, 2), "s1", "t1", "s1_t1", rest_bounds=(4, 10), drop_rest=True
    )
    assert y.tolist() == [5, 5, 5]
    assert meta.window_start.tolist() == [2, 4, 6]
    assert X.shape == (3, 4, 1)


def _dataset():
    return Dataset(
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1"],
            class_names=["noGesture", "fist"],
            content_hash="h",
        ),
        users=[
            User(
                "s1",
                [
                    Trial(
                        "t1",
                        "s1",
                        "fist",
                        0,
                        10,
                        np.arange(10, dtype=np.float32)[:, None],
                    ),
                    Trial(
                        "t2",
                        "s1",
                        "noGesture",
                        -1,
                        -1,
                        np.ones((10, 1), np.float32),
                    ),
                ],
            )
        ],
    )


def test_window_dataset_matches_flat_behavior():
    ds = _dataset()
    wd = window_dataset(ds, _cfg(4, 2))
    # gesture trial [0,10] -> 4 windows labeled 1; rest trial -> 4 windows labeled 0
    assert wd.X.shape == (8, 4, 1)
    assert wd.y.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert wd.metadata.subject_ids.tolist() == ["s1"] * 8
    assert wd.metadata.trial_ids.tolist() == [
        "t1",
        "t1",
        "t1",
        "t1",
        "t2",
        "t2",
        "t2",
        "t2",
    ]
    assert wd.metadata.window_start.tolist() == [0, 2, 4, 6, 0, 2, 4, 6]
    assert wd.metadata.window_ids[0] == "s1_t1_window0"


def test_window_dataset_drops_rest_windows_when_no_gesture_dropped():
    # A single gesture trial whose EMG extends beyond the gesture interval
    # [start, end]. With ws=40, stride=20 the trailing context (samples 80..139)
    # produces windows that fall entirely outside [start, end] (rest windows).
    # (An emg length of ~100 is too short to isolate a 40-sample rest window,
    # so use 140 to actually exercise the drop path.)
    emg_len = 140
    start, end = 20, 80
    trial = Trial(
        "t1",
        "s1",
        "fist",
        start,
        end,
        np.arange(emg_len, dtype=np.float32)[:, None],
    )

    # Dropped config: class_names=["fist"], so label 0 now means "fist" and any
    # rest window (which would be labeled 0) must be dropped entirely.
    dropped_meta = DatasetMetadata(
        name="x",
        version="0",
        sampling_rate=200.0,
        channels=["c1"],
        class_names=["fist"],
        content_hash="h",
        no_gesture_name="noGesture",
    )
    dropped = Dataset.from_trials([trial], dropped_meta)
    wd = window_dataset(dropped, _cfg(40, 20))
    assert wd.y.tolist() == [0, 0, 0, 0]  # all "fist"
    assert wd.metadata.window_start.tolist() == [0, 20, 40, 60]
    # No window lies fully outside the gesture interval [start, end]:
    assert (wd.metadata.window_start >= end).sum() == 0
    assert (wd.metadata.window_end <= start).sum() == 0

    # Recognition config (no-gesture kept) still emits rest windows labeled 0.
    kept_meta = DatasetMetadata(
        name="x",
        version="0",
        sampling_rate=200.0,
        channels=["c1"],
        class_names=["noGesture", "fist"],
        content_hash="h",
    )
    kept = Dataset.from_trials([trial], kept_meta)
    wd_kept = window_dataset(kept, _cfg(40, 20))
    assert wd_kept.y.tolist() == [1, 1, 1, 1, 0, 0]
    assert wd_kept.metadata.window_start.tolist() == [0, 20, 40, 60, 80, 100]
