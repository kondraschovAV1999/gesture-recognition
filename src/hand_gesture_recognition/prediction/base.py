from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Prediction:
    run_id: str
    sample_ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: Optional[np.ndarray]
    prediction_duration_s: float
    backend: str  # "classical" | "neural"


@dataclass
class PartitionPredictions:
    """Predictions for the train/val/test partitions of a single run."""

    train: Prediction
    val: Prediction
    test: Prediction


class PredictionAdapter(ABC):
    @abstractmethod
    def to_prediction(self, **kwargs) -> Prediction:
        """Normalize backend-native output into a framework Prediction."""
        raise NotImplementedError
