import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.base import window_dataset


def _ds(lengths=(100, 100)):
    trials = [
        np.arange(l, dtype=np.float32)[:, None].repeat(2, axis=1) for l in lengths
    ]
    return Dataset(
        trials=trials,
        labels=np.array([0, 1]),
        subject_ids=np.array(["s1", "s2"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.zeros(2, dtype=int),
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


def test_window_count_matches_X():
    ds = _ds(lengths=(100, 100))
    wd = window_dataset(ds, WindowingConfig(window_size_samples=40, stride_samples=10))
    assert wd.X.shape[0] == 14
    assert len(wd.metadata.window_ids) == 14


def test_metadata_alignment_per_window():
    ds = _ds(lengths=(100, 100))
    wd = window_dataset(ds, WindowingConfig(window_size_samples=40, stride_samples=10))
    for i in range(wd.X.shape[0]):
        # window_start/end are per-trial (reset to 0 within each trial);
        # each trial yields 7 windows at stride 10.
        assert wd.metadata.window_start[i] == (i % 7) * 10
        assert wd.metadata.window_end[i] == (i % 7) * 10 + 40
        # window id embeds the trial + a per-trial window index
        assert wd.metadata.window_ids[i].endswith(f"_window{i % 7}")
