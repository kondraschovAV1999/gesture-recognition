"""Full end-to-end classical pipeline on a SMALL subset of the real EMG-EPN612 dataset."""

from __future__ import annotations

import os

import pytest

from hand_gesture_recognition.classical.adapters.sklearn import SklearnPredictionAdapter
from hand_gesture_recognition.classical.models.random_forest import (
    RandomForestEstimator,
)
from hand_gesture_recognition.classical.pipelines.windowing import WindowingTransformer
from hand_gesture_recognition.configuration.schema import (
    EvaluationConfig,
    SplitConfig,
    WindowingConfig,
)
from hand_gesture_recognition.data.loaders.epn612 import EPN612Loader
from hand_gesture_recognition.data.splits.base import (
    assert_no_group_leakage,
    split_dataset,
)
from hand_gesture_recognition.evaluation.base import evaluate
from hand_gesture_recognition.features.time_domain import TimeDomainFeatures
from hand_gesture_recognition.paths import resolve_path

DATA_PATH = resolve_path("resources/data/EPN612")
HAS_REAL_DATA = os.path.isdir(DATA_PATH)

pytestmark = pytest.mark.skipif(
    not HAS_REAL_DATA,
    reason="EMG-EPN612 dataset not present under resources/data/EPN612",
)

SUBJECTS = [f"train:user{i}" for i in range(1, 7)]  # 6 subjects


def test_full_classical_pipeline_real_data():
    ds = EPN612Loader(subjects=SUBJECTS).load(DATA_PATH)
    assert ds.n_trials > 0

    split = split_dataset(
        ds,
        SplitConfig(
            strategy="group_wise",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=0,
        ),
    )
    assert_no_group_leakage(split)

    windower = WindowingTransformer(
        WindowingConfig(window_size_seconds=0.2, stride_seconds=0.05)
    )
    train_wd = windower.fit_transform(split.train)
    test_wd = windower.transform(split.test)

    # Per-window labels: both rest (0) and gesture (1-5) present.
    assert 0 in train_wd.y
    assert set(train_wd.y.tolist()) - {0}

    feat = TimeDomainFeatures()
    X_train = feat.fit_transform(train_wd.X)
    X_test = feat.transform(test_wd.X)

    model = RandomForestEstimator(n_estimators=50, random_state=0)
    model.fit(X_train, train_wd.y)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    adapter = SklearnPredictionAdapter()
    pred = adapter.to_prediction(
        windowed=test_wd,
        y_pred=y_pred,
        y_score=y_score,
        run_id="classical-full-test",
        duration_s=0.0,
    )
    assert pred.backend == "classical"

    result = evaluate(
        pred,
        EvaluationConfig(metrics=["accuracy", "balanced_accuracy", "macro_f1"]),
        dataset_id=ds.metadata.name,
        split_info={"strategy": "group_wise"},
    )
    assert result.metrics["accuracy"] > 1 / 6
    assert result.metrics["balanced_accuracy"] > 1 / 6
    assert result.confusion_matrix.shape == (6, 6)
