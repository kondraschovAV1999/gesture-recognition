from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


class RandomForestEstimator(RandomForestClassifier):
    """Random Forest classifier (thin wrapper for _target_ instantiation)."""
