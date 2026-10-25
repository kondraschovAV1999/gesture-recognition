from __future__ import annotations

from abc import ABC, abstractmethod

from hand_gesture_recognition.configuration.schema import Dataset


class DatasetLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Dataset:
        """Load a dataset from `path` into the framework Dataset representation."""
        raise NotImplementedError
