from hydra.utils import instantiate
import numpy as np


class FeatureExtractor:
    """Apply a list of feature-extractor specs and concatenate column-wise.

    Each spec is a Hydra-style dict with a `_target_` pointing at a
    scikit-learn transformer whose `transform(X)` maps (n, time, ch) -> (n, f).
    Multiple extractors compose by concatenating their outputs along axis 1.
    """

    def __init__(self, feature_specs):
        self.feature_specs = list(feature_specs)
        self._steps = [instantiate(s) for s in self.feature_specs]

    def fit_transform(self, X):
        if not self._steps:
            return np.empty((X.shape[0], 0))
        return np.concatenate([s.fit_transform(X) for s in self._steps], axis=1)

    def transform(self, X):
        if not self._steps:
            return np.empty((X.shape[0], 0))
        return np.concatenate([s.transform(X) for s in self._steps], axis=1)

    def transform_windows(self, list_of_X):
        """Transform a list of window arrays (one array per item), used by the
        two-level grouped path where small windows are grouped per big window."""
        return [self.transform(X) for X in list_of_X]
