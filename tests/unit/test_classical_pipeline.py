import numpy as np
from sklearn.pipeline import Pipeline

from hand_gesture_recognition.classical.models.random_forest import (
    RandomForestEstimator,
)
from hand_gesture_recognition.classical.pipelines.windowing import WindowingTransformer
from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.features.time_domain import TimeDomainFeatures


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
