import pytest

from hand_gesture_recognition.configuration.schema import SplitConfig
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
