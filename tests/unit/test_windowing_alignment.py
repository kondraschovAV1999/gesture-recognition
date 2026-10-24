import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.utils.datasets import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.base import window_dataset


def _ds(lengths=(100, 100)):
    trials = [
        Trial(
            f"t{i + 1}",
            f"s{i + 1}",
            "a" if i == 0 else "b",
            0,
            l,
            np.arange(l, dtype=np.float32)[:, None].repeat(2, axis=1),
        )
        for i, l in enumerate(lengths)
    ]
    return Dataset(
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["a", "b"],
            content_hash="h",
        ),
        users=[User(t.user_id, [t]) for t in trials],
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
