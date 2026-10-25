from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.configuration.schema import Dataset
from hand_gesture_recognition.configuration.schema import WindowedDataset
from hand_gesture_recognition.pipeline.feature_extraction.windowing.base import window_dataset


class WindowingTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper around the shared windowing implementation."""

    def __init__(self, window_config: WindowingConfig):
        self.window_config = window_config

    def fit(self, X, y=None):
        return self

    def transform(self, X: Dataset) -> WindowedDataset:
        return window_dataset(X, self.window_config)

    def fit_transform(self, X: Dataset, y=None, **fit_params) -> WindowedDataset:
        return self.transform(X)
