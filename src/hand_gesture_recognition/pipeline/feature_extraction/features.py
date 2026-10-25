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

    def transform_grouped(self, small_windows, max_small_per_big):
        """Transform each big window's small windows into a fixed-length feature
        sequence: apply the extractor to each (n_small, ws, ch) array, then
        zero-pad the tail / truncate to the head to ``max_small_per_big`` rows.
        Returns a list of (max_small_per_big, n_features) float32 arrays."""
        out = []
        for smalls in small_windows:
            feats = self.transform(smalls).astype(np.float32)  # (n_small, n_features)
            if feats.shape[0] > max_small_per_big:
                feats = feats[:max_small_per_big]
            elif feats.shape[0] < max_small_per_big:
                pad = np.zeros(
                    (max_small_per_big - feats.shape[0], feats.shape[1]),
                    dtype=np.float32,
                )
                feats = np.vstack([feats, pad])
            out.append(feats)
        return out
