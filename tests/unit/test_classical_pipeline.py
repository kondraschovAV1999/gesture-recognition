import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from hand_gesture_recognition.classical.models.random_forest import (
    RandomForestEstimator,
)
from hand_gesture_recognition.classical.pipelines.windowing import WindowingTransformer
from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.features.time_domain import TimeDomainFeatures
from hand_gesture_recognition.features.frequency_domain import FrequencyDomainFeatures


def test_windowing_transformer_calls_shared_windowing():
    ds = make_synthetic_dataset(
        n_subjects=2, n_trials_per_subject=2, length=100, seed=0
    )
    t = WindowingTransformer(WindowingConfig(window_size_samples=40, stride_samples=10))
    wd = t.fit_transform(ds)
    assert wd.X.ndim == 3  # (windows, time, channels) -> shared representation
    assert wd.X.shape[1] == 40


def test_full_classical_pipeline_fits_and_predicts():
    ds = make_synthetic_dataset(
        n_subjects=4, n_trials_per_subject=6, length=100, seed=0
    )
    win = WindowingTransformer(
        WindowingConfig(window_size_samples=40, stride_samples=20)
    )
    feat = TimeDomainFeatures()
    model = RandomForestEstimator(n_estimators=20, random_state=0)
    # manual wiring (the Runner wires these via hydra.utils.instantiate)
    wd = win.fit_transform(ds)
    X = feat.fit_transform(wd.X)
    model.fit(X, wd.y)
    preds = model.predict(X)
    assert preds.shape == wd.y.shape
    assert set(preds.tolist()).issubset(set(range(3)))


def test_time_domain_features_per_channel():
    # 1 window, 3 time steps, 1 channel: values [0, 2, 4]
    X = np.array([[[0.0], [2.0], [4.0]]], dtype=np.float32)  # (1, 3, 1)
    out = TimeDomainFeatures().transform(X)
    assert out.shape == (1, 7)  # 7 features x 1 channel
    rms, std, mn, mx, mean, mx_mean, mn_mx = out[0]
    assert rms == pytest.approx(np.sqrt(20 / 3))
    assert std == pytest.approx(np.std([0.0, 2.0, 4.0]))
    assert mn == 0.0
    assert mx == 4.0
    assert mean == 2.0
    assert mx_mean == pytest.approx(2.0)  # max / mean = 4 / 2
    assert mn_mx == 0.0  # min / max = 0 / 4


def test_time_domain_features_8_channels():
    X = np.zeros((5, 40, 8), dtype=np.float32)  # 5 windows, 40 time, 8 channels
    out = TimeDomainFeatures().transform(X)
    assert out.shape == (5, 56)  # 7 features x 8 channels
    # all-zero input must not produce NaN (division guards)
    assert np.isfinite(out).all()


def test_time_and_frequency_features_concatenate():
    X = np.random.default_rng(0).normal(0, 1, size=(5, 40, 8)).astype(np.float32)
    time_feat = TimeDomainFeatures().fit_transform(X)
    freq_feat = FrequencyDomainFeatures(sampling_rate=200.0).fit_transform(X)
    combined = np.concatenate([time_feat, freq_feat], axis=1)
    assert combined.shape == (5, 7 * 8 + 4 * 8)  # union of both extractors
    assert np.isfinite(combined).all()
