import numpy as np

from hand_gesture_recognition.configuration.schema import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
    WindowingConfig,
)
from hand_gesture_recognition.utils.observability.shapes import compute_shape_summary

_TIME_SPEC = [
    {
        "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures"
    }
]


def _ds(n_channels=8, length=994):
    meta = DatasetMetadata(
        "x",
        "0",
        200.0,
        [f"ch{i}" for i in range(n_channels)],
        ["noGesture", "fist"],
        "h",
    )
    rest = Trial(
        "t_rest", "s1", "noGesture", -1, -1, np.zeros((length, n_channels), np.float32)
    )
    fist = Trial(
        "t_fist", "s1", "fist", 0, length, np.ones((length, n_channels), np.float32)
    )
    return Dataset(meta, [User("s1", [rest, fist])])


def test_two_level_shape_summary():
    ds = _ds()
    windowing = WindowingConfig(
        mode="two_level",
        window_size_samples=25,
        stride_samples=10,
        big_window_size_samples=250,
        big_stride_samples=50,
    )
    s = compute_shape_summary(ds, windowing, _TIME_SPEC)
    assert s["mode"] == "two_level"
    assert s["input"] == {"channels": 8, "datapoints_per_channel": 994}
    assert s["big_windows"]["channels"] == 8
    assert s["big_windows"]["big_window_size"] == 250
    assert s["big_windows"]["big_windows"] == (994 - 250) // 50 + 1
    assert s["small_windows"]["small_window_size"] == 25
    assert s["small_windows"]["small_windows"] == (250 - 25) // 10 + 1
    assert s["small_windows"]["big_windows"] == s["big_windows"]["big_windows"]
    assert s["features"] == {
        "channels": 8,
        "big_windows": s["big_windows"]["big_windows"],
        "small_windows": s["small_windows"]["small_windows"],
        "features_per_small_window": 11,
    }


def test_single_level_shape_summary():
    ds = _ds()
    windowing = WindowingConfig(window_size_samples=25, stride_samples=10)
    s = compute_shape_summary(ds, windowing, _TIME_SPEC, task_name="recognition")
    assert s["mode"] == "single"
    assert s["input"] == {"channels": 8, "datapoints_per_channel": 994}
    assert s["windows"]["window_size"] == 25
    assert s["windows"]["windows"] == (994 - 25) // 10 + 1
    assert s["features"]["features_per_window"] == 11


def test_subset_features_per_channel():
    ds = _ds(n_channels=2, length=100)
    windowing = WindowingConfig(window_size_samples=25, stride_samples=10)
    subset_spec = [
        {
            "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures",
            "features": ["rms", "mav"],
        }
    ]
    s = compute_shape_summary(ds, windowing, subset_spec)
    assert s["features"]["features_per_window"] == 2


def test_no_non_rest_trial_returns_empty():
    meta = DatasetMetadata("x", "0", 200.0, ["ch1"], ["noGesture", "fist"], "h")
    ds = Dataset(
        meta,
        [
            User(
                "s1",
                [Trial("t", "s1", "noGesture", -1, -1, np.zeros((50, 1), np.float32))],
            )
        ],
    )
    windowing = WindowingConfig(window_size_samples=10, stride_samples=5)
    assert compute_shape_summary(ds, windowing, _TIME_SPEC) == {}
