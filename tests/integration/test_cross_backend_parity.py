"""Cross-backend parity: classical and neural pipelines must produce Prediction
and EvaluationResult objects with identical schema (fields and shapes)."""

from __future__ import annotations

import numpy as np

from hand_gesture_recognition.configuration.schema import (
    EvaluationConfig,
    SplitConfig,
    WindowingConfig,
)
from hand_gesture_recognition.utils.fixtures import make_synthetic_dataset
from hand_gesture_recognition.pipeline.splits.base import split_dataset
from hand_gesture_recognition.pipeline.evaluation.base import evaluate


def _classical_prediction(split, windowing):
    from hand_gesture_recognition.pipeline.models.classical.adapter import (
        SklearnPredictionAdapter,
    )
    from hand_gesture_recognition.pipeline.models.classical.random_forest import (
        RandomForestEstimator,
    )
    from hand_gesture_recognition.pipeline.models.classical.windowing import (
        WindowingTransformer,
    )
    from hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain import TimeDomainFeatures

    windower = WindowingTransformer(windowing)
    train_wd = windower.fit_transform(split.train)
    test_wd = windower.transform(split.test)
    feat = TimeDomainFeatures()
    X_train = feat.fit_transform(train_wd.X)
    X_test = feat.transform(test_wd.X)
    model = RandomForestEstimator(n_estimators=20, random_state=0)
    model.fit(X_train, train_wd.y)
    return SklearnPredictionAdapter().to_prediction(
        windowed=test_wd,
        y_pred=model.predict(X_test),
        y_score=model.predict_proba(X_test),
        run_id="parity",
        duration_s=0.0,
    )


def _neural_prediction(split, windowing):
    import lightning.pytorch as pl

    from hand_gesture_recognition.pipeline.feature_extraction.windowing.base import (
        WindowedPartitions,
        window_dataset,
    )
    from hand_gesture_recognition.pipeline.models.neural.adapter import (
        LightningPredictionAdapter,
    )
    from hand_gesture_recognition.pipeline.models.neural.datamodule import EMGDataModule
    from hand_gesture_recognition.pipeline.models.neural.dnn import DNNModule

    pl.seed_everything(0, workers=True)
    n_channels = len(split.train.metadata.channels)
    n_classes = len(split.train.metadata.class_names)
    windowed = WindowedPartitions(
        train=window_dataset(split.train, windowing),
        val=window_dataset(split.val, windowing),
        test=window_dataset(split.test, windowing),
    )
    dm = EMGDataModule(windowed=windowed, batch_size=32)
    model = DNNModule(n_channels=n_channels, n_classes=n_classes, hidden_dims=(16,))
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, datamodule=dm)
    raw = trainer.predict(model, datamodule=dm)
    return LightningPredictionAdapter().to_prediction(
        raw_predictions=raw, run_id="parity", duration_s=0.0
    )


def test_cross_backend_prediction_schema_parity():
    ds = make_synthetic_dataset(
        n_subjects=4,
        n_trials_per_subject=4,
        length=100,
        n_channels=8,
        n_classes=3,
        seed=0,
    )
    split = split_dataset(
        ds,
        SplitConfig(
            strategy="random", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0
        ),
    )
    windowing = WindowingConfig(window_size_samples=40, stride_samples=20)

    classical = _classical_prediction(split, windowing)
    neural = _neural_prediction(split, windowing)

    assert classical.backend == "classical"
    assert neural.backend == "neural"
    # identical field set and shapes
    assert set(classical.__dict__.keys()) == set(neural.__dict__.keys())
    assert classical.y_true.shape == neural.y_true.shape
    assert classical.y_pred.shape == neural.y_pred.shape
    assert classical.y_score.shape == neural.y_score.shape
    # same windows: identical ground truth + sample ids (values may differ)
    assert np.array_equal(classical.y_true, neural.y_true)
    assert np.array_equal(classical.sample_ids, neural.sample_ids)

    # EvaluationResult schema parity too
    eval_cfg = EvaluationConfig(metrics=["accuracy"])
    res_c = evaluate(classical, eval_cfg, dataset_id="synthetic", split_info={})
    res_n = evaluate(neural, eval_cfg, dataset_id="synthetic", split_info={})
    assert set(res_c.__dict__.keys()) == set(res_n.__dict__.keys())
    assert res_c.confusion_matrix.shape == res_n.confusion_matrix.shape
