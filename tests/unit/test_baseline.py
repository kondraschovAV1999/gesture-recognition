from dataclasses import replace

import numpy as np
import pytest

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.preprocessing.baseline import BaselineRemoval


def _dataset():
    # s1: gesture trial (1.0) + rest trial (5.0); s2: rest trial (9.0)
    return Dataset(
        trials=[
            np.full((40, 2), 1.0, dtype=np.float32),
            np.full((30, 2), 5.0, dtype=np.float32),
            np.full((20, 2), 9.0, dtype=np.float32),
        ],
        labels=np.array([1, 0, 0]),
        subject_ids=np.array(["s1", "s1", "s2"]),
        trial_ids=np.array(["t1", "t2", "t3"]),
        gesture_start=np.array([10, -1, -1]),
        gesture_end=np.array([30, -1, -1]),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["rest", "fist"],
            content_hash="h",
        ),
    )


def test_global_mode_unchanged():
    rem = BaselineRemoval(mode="global").fit(_dataset())
    # global = length-weighted mean of rest samples: (30*5 + 20*9) / 50 = 6.6
    assert np.allclose(rem.baseline_, np.array([6.6, 6.6]))
    out = rem.transform(_dataset())
    assert np.allclose(out.trials[0], -5.6)  # 1 - 6.6
    assert np.allclose(out.trials[2], 2.4)  # 9 - 6.6


def test_per_user_builds_subject_baselines():
    rem = BaselineRemoval(mode="per_user").fit(_dataset())
    assert set(rem.baseline_.keys()) == {"s1", "s2"}
    assert np.allclose(rem.baseline_["s1"], 5.0)
    assert np.allclose(rem.baseline_["s2"], 9.0)


def test_per_user_subtracts_own_subject_baseline():
    ds = _dataset()
    rem = BaselineRemoval(mode="per_user").fit(ds)
    out = rem.transform(ds)
    assert np.allclose(out.trials[0], -4.0)  # s1 gesture: 1 - 5
    assert np.allclose(out.trials[1], 0.0)  # s1 rest: 5 - 5
    assert np.allclose(out.trials[2], 0.0)  # s2 rest: 9 - 9


def test_per_user_raises_when_subject_missing_rest():
    ds = _dataset()
    ds.labels = np.array([1, 0, 1])
    ds.gesture_start = np.array([10, -1, 0])
    ds.gesture_end = np.array([30, -1, 20])
    with pytest.raises(ValueError, match="no rest trials"):
        BaselineRemoval(mode="per_user").fit(ds)


def test_per_user_raises_on_unseen_subject():
    rem = BaselineRemoval(mode="per_user").fit(_dataset())
    ds2 = replace(
        _dataset(),
        trials=_dataset().trials + [np.full((10, 2), 3.0, dtype=np.float32)],
        labels=np.append(_dataset().labels, 0),
        subject_ids=np.append(_dataset().subject_ids, "s3"),
        trial_ids=np.append(_dataset().trial_ids, "t4"),
        gesture_start=np.append(_dataset().gesture_start, -1),
        gesture_end=np.append(_dataset().gesture_end, -1),
    )
    with pytest.raises(ValueError, match="no baseline for subject"):
        rem.transform(ds2)


def test_transform_before_fit_raises():
    with pytest.raises(ValueError, match="must be fit"):
        BaselineRemoval(mode="per_user").transform(_dataset())
