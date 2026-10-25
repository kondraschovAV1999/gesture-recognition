from __future__ import annotations

from abc import ABC, abstractmethod

from hand_gesture_recognition.configuration.schema import Prediction


class PredictionAdapter(ABC):
    @abstractmethod
    def to_prediction(self, **kwargs) -> Prediction:
        """Normalize backend-native output into a framework Prediction."""
        raise NotImplementedError
