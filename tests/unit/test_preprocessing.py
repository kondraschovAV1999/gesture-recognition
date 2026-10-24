import numpy as np
import pytest

from hand_gesture_recognition.data.datasets.base import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)
from hand_gesture_recognition.data.splits.base import DataSplit
from hand_gesture_recognition.preprocessing.baseline import BaselineRemoval
from hand_gesture_recognition.preprocessing.learned import PerChannelStandardScaler
from hand_gesture_recognition.preprocessing.pipeline import PreprocessingPipeline
from hand_gesture_recognition.preprocessing.signal import (
    ButterworthFilter,
    EnvelopeExtractor,
    RectifyTransformer,
)


def test_rectify():
    X = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    out = RectifyTransformer().transform(X)
    assert np.array_equal(out, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_butterworth_lowpass_attenuates_high_frequency():
    fs = 200.0
    t = np.arange(500) / fs
    x = np.sin(2 * np.pi * 60 * t).astype(np.float32)  # pure 60 Hz
    f = ButterworthFilter(type="lowpass", highcut=20, fs=fs)
    y = f.transform(x.reshape(-1, 1)).ravel()
    assert y.shape == x.shape
    # 60 Hz through a 20 Hz lowpass -> nearly eliminated
    assert np.var(y) < np.var(x) * 0.1


def test_butterworth_notch_removes_60hz():
    fs = 200.0
    t = np.arange(500) / fs
    x = np.sin(2 * np.pi * 60 * t).astype(np.float32)
    f = ButterworthFilter(type="notch", frequency=60, fs=fs)
    y = f.transform(x.reshape(-1, 1)).ravel()
    assert np.var(y) < np.var(x) * 0.1


def test_envelope_is_nonnegative_and_preserves_shape():
    t = np.arange(200)
    x = np.sin(2 * np.pi * 0.1 * t).astype(np.float32)
    y = EnvelopeExtractor(window_size=25).transform(x.reshape(-1, 1)).ravel()
    assert y.shape == x.shape
    assert (y >= 0).all()  # envelope of |x| is nonnegative


def test_per_channel_scaler_fit_transform():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=10, scale=5, size=(100, 40, 8)).astype(np.float32)
    scaler = PerChannelStandardScaler()
    out = scaler.fit_transform(X)
    assert np.allclose(out.mean(axis=(0, 1)), 0.0, atol=1e-5)
    assert np.allclose(out.std(axis=(0, 1)), 1.0, atol=1e-5)


def test_per_channel_scaler_uses_train_stats_only():
    rng = np.random.default_rng(0)
    train = rng.normal(loc=0, scale=1, size=(50, 40, 8)).astype(np.float32)
    test = rng.normal(loc=5, scale=1, size=(50, 40, 8)).astype(np.float32)
    scaler = PerChannelStandardScaler().fit(train)
    out_train = scaler.transform(train)
    out_test = scaler.transform(test)
    # test transformed with TRAIN stats -> not zero mean (it has loc=5 offset)
    assert not np.allclose(out_test.mean(axis=(0, 1)), 0.0, atol=1e-2)
    assert np.allclose(out_train.mean(axis=(0, 1)), 0.0, atol=1e-5)


def test_pipeline_fits_baseline_on_train_then_rectifies():
    meta = DatasetMetadata("x", "0", 200.0, ["c"], ["noGesture", "fist"], "h")

    def trial(trial_id, gesture_name, start, end, level):
        return Trial(
            trial_id, "u1", gesture_name, start, end, np.full((4, 1), level, np.float32)
        )

    # Train: rest level 5, gesture level 10.
    train = Dataset(
        meta,
        [
            User("u1", [trial("t0", "noGesture", -1, -1, 5.0)]),
            User("u1", [trial("t1", "fist", 0, 4, 10.0)]),
        ],
    )
    # Val: a DIFFERENT rest level (20) so that fitting the global baseline on
    # train's rest (5) vs. val's rest (20) is distinguishable.
    val = Dataset(meta, [User("u1", [trial("v0", "noGesture", -1, -1, 20.0)])])
    split = DataSplit(train=train, val=val, test=val)

    p = PreprocessingPipeline(
        pre_steps=[], post_steps=[BaselineRemoval(mode="global"), RectifyTransformer()]
    )
    p.fit(split)
    out = p.transform(split)

    # Global baseline = mean of TRAIN's rest trials = 5.0 (not val's 20.0).
    assert np.allclose(
        out.train.trials[1].emg, 5.0
    )  # gesture: 10 - 5, rectified unchanged
    assert np.allclose(out.train.trials[0].emg, 0.0)  # rest: 5 - 5
    assert np.allclose(out.val.trials[0].emg, 15.0)  # val rest: 20 - 5 (train baseline)
    # Every partition/trial is rectified to be non-negative.
    for arr in (
        out.train.trials[0].emg,
        out.train.trials[1].emg,
        out.val.trials[0].emg,
        out.test.trials[0].emg,
    ):
        assert np.all(arr >= 0.0)
