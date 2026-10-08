import numpy as np

from hand_gesture_recognition.configuration.schema import (
    CrossValidationConfig,
    SplitConfig,
)
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.data.splits.base import assert_no_group_leakage
from hand_gesture_recognition.data.splits.folds import generate_folds


def test_kfold_group_folds_no_leakage():
    ds = make_synthetic_dataset(n_subjects=6, n_trials_per_subject=4, seed=0)
    cv = CrossValidationConfig(
        enabled=True, method="kfold", n_splits=3, group_folds=True, seed=1
    )
    folds = generate_folds(ds, cv, SplitConfig(strategy="subject_wise", seed=1))
    assert len(folds) == 3
    for f in folds:
        assert_no_group_leakage(f)
        assert f.test.n_trials > 0
        assert f.train.n_trials > 0


def test_kfold_trial_folds_cover_all_trials_and_empty_val():
    ds = make_synthetic_dataset(n_subjects=3, n_trials_per_subject=6, seed=0)
    cv = CrossValidationConfig(
        enabled=True,
        method="kfold",
        n_splits=2,
        group_folds=False,
        include_val=False,
        seed=0,
    )
    folds = generate_folds(ds, cv, SplitConfig(strategy="random", seed=0))
    assert len(folds) == 2
    assert folds[0].val.n_trials == 0
    assert sum(f.test.n_trials for f in folds) == ds.n_trials


def test_loso_one_fold_per_subject():
    ds = make_synthetic_dataset(n_subjects=5, n_trials_per_subject=2, seed=0)
    cv = CrossValidationConfig(enabled=True, method="loso", include_val=False, seed=0)
    folds = generate_folds(ds, cv, SplitConfig(strategy="subject_wise", seed=0))
    assert len(folds) == 5
    for f in folds:
        assert len(set(f.test.subject_ids.tolist())) == 1
        assert_no_group_leakage(f)


def test_kfold_group_val_carved_from_train():
    ds = make_synthetic_dataset(n_subjects=8, n_trials_per_subject=3, seed=0)
    cv = CrossValidationConfig(
        enabled=True, method="kfold", n_splits=4, group_folds=True, seed=2
    )
    folds = generate_folds(ds, cv, SplitConfig(strategy="subject_wise", seed=2))
    for f in folds:
        assert f.val.n_trials > 0
        assert_no_group_leakage(f)
