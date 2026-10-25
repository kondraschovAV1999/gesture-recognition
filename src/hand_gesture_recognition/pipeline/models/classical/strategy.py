from __future__ import annotations

import time

import numpy as np
from hydra.utils import instantiate

from hand_gesture_recognition.pipeline.models.classical.adapter import SklearnPredictionAdapter
from hand_gesture_recognition.configuration.schema import WindowedDataset, WindowedPartitions
from hand_gesture_recognition.configuration.schema import GroupedWindowedPartitions
from hand_gesture_recognition.configuration.schema import PartitionPredictions
from hand_gesture_recognition.pipeline.models.base import TrainingStrategy, register_training
from hand_gesture_recognition.pipeline.feature_extraction.features import FeatureExtractor


@register_training("classical")
class ClassicalTrainingStrategy(TrainingStrategy):
    """Classical (scikit-learn) train+prediction.

    Reproduces the old ``ExperimentRunner._train_and_predict_classical`` /
    ``_train_and_predict_classical_two_level`` behaviour: (1) flatten a
    two-level grouped windowing result to its big-window view, (2) apply the
    configured classical preprocessing steps (fit on train), (3) extract
    handcrafted features, (4) fit the model, (5) predict all partitions with
    timing, and (6) normalize through the sklearn prediction adapter.
    """

    def train_predict(self, windowed, class_names, cfg) -> PartitionPredictions:
        windowed = self._flatten_grouped(windowed)

        X_train, X_val, X_test = windowed.train.X, windowed.val.X, windowed.test.X

        # Classical preprocessing: fit each step on train, transform all.
        for step_cfg in cfg.classical.preprocessing:
            step = instantiate(step_cfg)
            X_train = step.fit_transform(X_train)
            X_val = step.transform(X_val)
            X_test = step.transform(X_test)

        # Feature extraction: compose extractors column-wise (FeatureUnion).
        feature_specs = cfg.feature_extraction.resolve_features(
            cfg.dataset.sampling_rate
        )
        features = FeatureExtractor(feature_specs)
        X_train = features.fit_transform(X_train)
        X_val = features.transform(X_val)
        X_test = features.transform(X_test)

        model = instantiate(cfg.classical.model)
        model.fit(X_train, windowed.train.y)

        adapter = SklearnPredictionAdapter()

        t0 = time.perf_counter()
        y_train_pred = model.predict(X_train)
        y_train_score = model.predict_proba(X_train)
        train_duration = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_val_pred = model.predict(X_val)
        y_val_score = model.predict_proba(X_val)
        val_duration = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_test_pred = model.predict(X_test)
        y_test_score = model.predict_proba(X_test)
        test_duration = time.perf_counter() - t0

        return PartitionPredictions(
            train=adapter.to_prediction(
                windowed=windowed.train,
                y_pred=y_train_pred,
                y_score=y_train_score,
                run_id=cfg.run_id,
                duration_s=train_duration,
            ),
            val=adapter.to_prediction(
                windowed=windowed.val,
                y_pred=y_val_pred,
                y_score=y_val_score,
                run_id=cfg.run_id,
                duration_s=val_duration,
            ),
            test=adapter.to_prediction(
                windowed=windowed.test,
                y_pred=y_test_pred,
                y_score=y_test_score,
                run_id=cfg.run_id,
                duration_s=test_duration,
            ),
        )

    @staticmethod
    def _flatten_grouped(windowed):
        """Collapse a two-level grouped result to its big-window view.

        Matches the old ``_train_and_predict_classical_two_level`` flatten: each
        partition is reduced to the grouped dataset's big windows, so the
        classical pipeline trains/predicts on big windows directly.
        """
        if not isinstance(windowed, GroupedWindowedPartitions):
            return windowed

        def flatten(twd) -> WindowedDataset:
            return WindowedDataset(X=twd.big.X, y=twd.big.y, metadata=twd.big.metadata)

        return WindowedPartitions(
            train=flatten(windowed.train),
            val=flatten(windowed.val),
            test=flatten(windowed.test),
        )
