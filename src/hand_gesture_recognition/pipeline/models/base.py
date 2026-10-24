from __future__ import annotations

_REGISTRY: dict = {}


def register_training(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


def get_training_strategy(name):
    return _REGISTRY[name]()


class TrainingStrategy:
    """Backend-specific train+prediction. Implementations override train_predict."""

    def train_predict(self, windowed, class_names, cfg):
        """Return a PartitionPredictions for train/val/test."""
        raise NotImplementedError
