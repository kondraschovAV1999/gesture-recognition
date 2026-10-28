import numpy as np

from hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain import (
    TimeDomainFeatures,
)
from hand_gesture_recognition.pipeline.feature_extraction.features import (
    FeatureExtractor,
)


class DoublingStub:
    """Stub feature extractor that doubles every value and flattens to 2D.

    Maps (n, time, ch) -> (n, time * ch) with each value doubled, so it acts
    as a second (independent) extractor for column-concatenation tests.
    """

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return (np.asarray(X) * 2.0).reshape(X.shape[0], -1)


def test_single_extractor_shape():
    n_windows, n_channels = 5, 8
    X = (
        np.random.default_rng(0)
        .normal(0, 1, size=(n_windows, 40, n_channels))
        .astype(np.float32)
    )
    fe = FeatureExtractor(
        [
            {
                "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures"
            }
        ]
    )
    assert fe.fit_transform(X).shape == (n_windows, 11 * n_channels)
    assert fe.transform(X).shape == (n_windows, 11 * n_channels)


def test_two_extractors_concatenate_along_axis1():
    n_windows, n_time, n_channels = 5, 40, 8
    X = (
        np.random.default_rng(1)
        .normal(0, 1, size=(n_windows, n_time, n_channels))
        .astype(np.float32)
    )
    fe = FeatureExtractor(
        [
            {
                "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures"
            },
            {"_target_": __name__ + ".DoublingStub"},
        ]
    )
    out = fe.fit_transform(X)
    expected_cols = 11 * n_channels + n_time * n_channels
    assert out.shape == (n_windows, expected_cols)
    # Left block = time-domain features; right block = doubled (flattened) values.
    assert np.allclose(out[:, : 11 * n_channels], TimeDomainFeatures().transform(X))
    assert np.allclose(out[:, 11 * n_channels :], (X * 2.0).reshape(n_windows, -1))


def test_empty_specs_return_zero_columns():
    X = np.random.default_rng(2).normal(0, 1, size=(4, 20, 3)).astype(np.float32)
    fe = FeatureExtractor([])
    assert fe.fit_transform(X).shape == (4, 0)
    assert fe.transform(X).shape == (4, 0)


def test_transform_windows():
    fe = FeatureExtractor(
        [
            {
                "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures"
            }
        ]
    )
    windows = [
        np.random.default_rng(i).normal(0, 1, size=(3, 40, 4)).astype(np.float32)
        for i in range(2)
    ]
    outs = fe.transform_windows(windows)
    assert len(outs) == 2
    for w, o in zip(windows, outs):
        assert o.shape == (w.shape[0], 11 * 4)
