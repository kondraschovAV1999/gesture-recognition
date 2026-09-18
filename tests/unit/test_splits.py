import numpy as np
import pytest

from hand_gesture_recognition.configuration.schema import SplitConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.data.splits.base import (
    assert_no_group_leakage,
    split_dataset,
)


def test_subject_wise_split_has_no_leakage():
    ds = make_synthetic_dataset(n_subjects=6, n_trials_per_subject=4, seed=0)
    cfg = SplitConfig(
        strategy="subject_wise",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=1,
    )
    split = split_dataset(ds, cfg)
    assert_no_group_leakage(split)  # raises on any subject overlap
    # every trial lands in exactly one partition
    total = split.train.n_trials + split.val.n_trials + split.test.n_trials
    assert total == ds.n_trials


def test_subject_wise_split_never_empty_partitions():
    ds = make_synthetic_dataset(n_subjects=3, n_trials_per_subject=3, seed=0)
    cfg = SplitConfig(
        strategy="subject_wise",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=0,
    )
    split = split_dataset(ds, cfg)
    assert split.train.n_trials > 0
    assert split.val.n_trials > 0
    assert split.test.n_trials > 0


def test_random_split_preserves_count():
    ds = make_synthetic_dataset(n_subjects=4, n_trials_per_subject=5, seed=0)
    cfg = SplitConfig(
        strategy="random", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=7
    )
    split = split_dataset(ds, cfg)
    assert (
        split.train.n_trials + split.val.n_trials + split.test.n_trials == ds.n_trials
    )


def test_leakage_detection_raises():
    ds = make_synthetic_dataset(n_subjects=2, n_trials_per_subject=3, seed=0)
    cfg = SplitConfig(
        strategy="random", train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    split = split_dataset(ds, cfg)
    # force overlap
    split.val = split.train
    with pytest.raises(ValueError):
        assert_no_group_leakage(split)


def _epn612_like_dataset():
    """Build a dataset mimicking EPN612: a train cohort (train:) with
    trainingSamples + testingSamples, and a test cohort (test:) with
    trainingSamples only."""
    trials, labels, subjects, tids = [], [], [], []
    for subj in ["train:s1", "train:s2"]:
        for sec in ["trainingSamples", "testingSamples"]:
            for i in range(2):
                trials.append(np.zeros((100, 8), dtype=np.float32))
                labels.append(1)
                subjects.append(subj)
                tids.append(f"{subj}:{sec}:idx_{i}")
    for i in range(2):
        trials.append(np.zeros((100, 8), dtype=np.float32))
        labels.append(2)
        subjects.append("test:s3")
        tids.append(f"test:s3:trainingSamples:idx_{i}")
    return Dataset(
        trials=trials,
        labels=np.asarray(labels, dtype=int),
        subject_ids=np.asarray(subjects),
        trial_ids=np.asarray(tids),
        gesture_start=np.zeros(len(trials), dtype=int),
        gesture_end=np.array([t.shape[0] for t in trials], dtype=int),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=[f"ch{i}" for i in range(1, 9)],
            class_names=["a", "b", "c"],
            content_hash="h",
        ),
    )


def test_predefined_split_is_subject_independent():
    ds = _epn612_like_dataset()
    split = split_dataset(ds, SplitConfig(strategy="predefined", seed=0))

    # train + val = subject-wise split of the train cohort (train:s1, train:s2);
    # test = the test cohort (test:s3) held out entirely.
    assert split.train.n_trials == 4  # one train-cohort subject (4 trials)
    assert split.val.n_trials == 4  # the other train-cohort subject
    assert split.test.n_trials == 2  # test:s3 (2 training samples)

    assert all(s.startswith("train:") for s in set(split.train.subject_ids))
    assert all(s.startswith("train:") for s in set(split.val.subject_ids))
    assert all(s.startswith("test:") for s in set(split.test.subject_ids))

    # genuinely independent: no subject appears in more than one partition
    assert_no_group_leakage(split)
