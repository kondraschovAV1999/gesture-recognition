from typing import Dict, cast

import numpy as np
import pytest

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata, Trial
from hand_gesture_recognition.pipeline.preprocessing.baseline import BaselineRemoval


def _metadata():
    return DatasetMetadata(
        name="x",
        version="0",
        sampling_rate=200.0,
        channels=["c1", "c2"],
        class_names=["rest", "fist"],
        content_hash="h",
    )


def _trial(trial_id, user_id, gesture_name, start, end, emg):
    return Trial(
        trial_id=trial_id,
        user_id=user_id,
        gesture_name=gesture_name,
        start=start,
        end=end,
        emg=np.asarray(emg, dtype=np.float32),
    )


def _dataset():
    # s1: gesture trial (1.0) + rest trial (5.0); s2: rest trial (9.0)
    trials = [
        _trial("t1", "s1", "fist", 10, 30, np.full((40, 2), 1.0)),
        _trial("t2", "s1", "rest", -1, -1, np.full((30, 2), 5.0)),
        _trial("t3", "s2", "rest", -1, -1, np.full((20, 2), 9.0)),
    ]
    return Dataset.from_trials(trials, _metadata())


def test_global_mode_unchanged():
    rem = BaselineRemoval(mode="global").fit(_dataset())
    # global = length-weighted mean of rest samples: (30*5 + 20*9) / 50 = 6.6
    assert np.allclose(cast(np.ndarray, rem.baseline_), np.array([6.6, 6.6]))
    out = rem.transform(_dataset())
    assert np.allclose(out.trials[0].emg, -5.6)  # 1 - 6.6
    assert np.allclose(out.trials[2].emg, 2.4)  # 9 - 6.6


def test_per_user_builds_subject_baselines():
    rem = BaselineRemoval(mode="per_user").fit(_dataset())
    baseline = cast(Dict[str, np.ndarray], rem.baseline_)
    assert set(baseline.keys()) == {"s1", "s2"}
    assert np.allclose(baseline["s1"], 5.0)
    assert np.allclose(baseline["s2"], 9.0)


def test_per_user_subtracts_own_subject_baseline():
    ds = _dataset()
    rem = BaselineRemoval(mode="per_user").fit(ds)
    out = rem.transform(ds)
    assert np.allclose(out.trials[0].emg, -4.0)  # s1 gesture: 1 - 5
    assert np.allclose(out.trials[1].emg, 0.0)  # s1 rest: 5 - 5
    assert np.allclose(out.trials[2].emg, 0.0)  # s2 rest: 9 - 9


def test_per_user_raises_when_subject_missing_rest():
    # s2 now has a gesture trial (start >= 0) and no rest trial.
    trials = [
        _trial("t1", "s1", "fist", 10, 30, np.full((40, 2), 1.0)),
        _trial("t2", "s1", "rest", -1, -1, np.full((30, 2), 5.0)),
        _trial("t3", "s2", "fist", 0, 20, np.full((20, 2), 9.0)),
    ]
    ds = Dataset.from_trials(trials, _metadata())
    with pytest.raises(ValueError, match="no rest trials"):
        BaselineRemoval(mode="per_user").fit(ds)


def test_per_user_raises_on_unseen_subject():
    rem = BaselineRemoval(mode="per_user").fit(_dataset())
    ds2 = Dataset.from_trials(
        _dataset().trials + [_trial("t4", "s3", "rest", -1, -1, np.full((10, 2), 3.0))],
        _metadata(),
    )
    with pytest.raises(ValueError, match="no baseline for subject"):
        rem.transform(ds2)


def test_transform_before_fit_raises():
    with pytest.raises(ValueError, match="must be fit"):
        BaselineRemoval(mode="per_user").transform(_dataset())
