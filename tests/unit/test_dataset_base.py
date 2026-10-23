import numpy as np

from hand_gesture_recognition.data.datasets.base import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset


def _meta():
    return DatasetMetadata(
        name="x",
        version="0",
        sampling_rate=200.0,
        channels=["c1", "c2"],
        class_names=["noGesture", "fist", "open"],
        content_hash="h",
    )


def test_trial_is_rest():
    rest = Trial("t1", "u1", "noGesture", -1, -1, np.zeros((10, 2), np.float32))
    gesture = Trial("t2", "u1", "fist", 0, 10, np.zeros((10, 2), np.float32))
    assert rest.is_rest is True
    assert gesture.is_rest is False


def test_dataset_flatten_and_derived_labels():
    u = User(
        "u1",
        [
            Trial("u1:t1", "u1", "fist", 0, 10, np.zeros((10, 2), np.float32)),
            Trial("u1:t2", "u1", "noGesture", -1, -1, np.zeros((10, 2), np.float32)),
            Trial("u1:t3", "u1", "open", 0, 10, np.zeros((10, 2), np.float32)),
        ],
    )
    ds = Dataset(_meta(), [u])
    assert ds.n_trials == 3
    assert [t.trial_id for t in ds.trials] == ["u1:t1", "u1:t2", "u1:t3"]
    # labels derived from class_names ordering: fist=1, noGesture=0, open=2
    assert ds.labels.tolist() == [1, 0, 2]
    assert ds.subject_ids.tolist() == ["u1", "u1", "u1"]
    assert ds.trial_ids.tolist() == ["u1:t1", "u1:t2", "u1:t3"]


def test_dataset_trials_cache_respects_replace():
    u = User("u1", [Trial("u1:t1", "u1", "fist", 0, 10, np.zeros((10, 2), np.float32))])
    ds = Dataset(_meta(), [u])
    first = ds.trials
    assert ds.trials is first  # cached


def test_from_trials_groups_by_user():
    trials = [
        Trial("u1:t1", "u1", "fist", 0, 10, np.zeros((10, 2), np.float32)),
        Trial("u2:t1", "u2", "fist", 0, 10, np.zeros((10, 2), np.float32)),
        Trial("u1:t2", "u1", "open", 0, 10, np.zeros((10, 2), np.float32)),
    ]
    ds = Dataset.from_trials(trials, _meta())
    assert [u.user_id for u in ds.users] == ["u1", "u2"]
    assert len(ds.users[0].trials) == 2
    assert len(ds.users[1].trials) == 1


def test_synthetic_dataset_is_nested():
    ds = make_synthetic_dataset(seed=0)
    assert ds.n_trials == 4 * 6
    assert len(ds.users) == 4
    assert ds.trials[0].emg.shape == (100, 8)
    assert ds.trials[0].gesture_name == "class_0"  # seed 0, s=0,t=0 -> label 0
    assert ds.labels.tolist() == [(s + t) % 3 for s in range(4) for t in range(6)]


def test_replace_trials_preserves_demographics():
    meta = DatasetMetadata("x", "0", 200.0, ["c"], ["noGesture", "fist"], "h")
    u = User(
        "u1",
        [Trial("u1:t1", "u1", "fist", 0, 10, np.zeros((10, 1), np.float32))],
        demographics={"age": 30},
    )
    ds = Dataset(meta, [u])
    out = ds.replace_trials(
        [Trial("u1:t2", "u1", "fist", 0, 10, np.zeros((10, 1), np.float32))]
    )
    assert out.users[0].demographics == {"age": 30}
