import os
from dataclasses import replace

import numpy as np

from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.reporting.visualization import (
    plot_fft_spectrum,
    plot_raw_signal_by_class,
    plot_signal_processing_comparison,
)


def test_plot_raw_signal_by_class(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2,
        n_trials_per_subject=3,
        n_classes=3,
        length=100,
        n_channels=4,
        seed=0,
    )
    paths = plot_raw_signal_by_class(
        ds, str(tmp_path), class_names=ds.metadata.class_names
    )
    assert len(paths) == 3  # classes 0,1,2 all present
    for p in paths:
        assert os.path.exists(p)
    assert os.path.exists(str(tmp_path / "raw_signal_class_0.png"))


def test_plot_signal_processing_comparison(tmp_path):
    raw = make_synthetic_dataset(
        n_subjects=2,
        n_trials_per_subject=3,
        n_classes=3,
        length=100,
        n_channels=4,
        seed=0,
    )
    proc = replace(raw, trials=[np.abs(t) for t in raw.trials])
    path = str(tmp_path / "signal_processing_comparison.png")
    fig = plot_signal_processing_comparison(raw, proc, path=path)
    assert fig is not None
    assert os.path.exists(path)


def test_plot_fft_spectrum(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2,
        n_trials_per_subject=3,
        n_classes=3,
        length=100,
        n_channels=4,
        seed=0,
    )
    windowing = WindowingConfig(window_size_samples=40, stride_samples=20)
    path = str(tmp_path / "fft_spectrum.png")
    fig = plot_fft_spectrum(ds, windowing, path=path)
    assert fig is not None
    assert os.path.exists(path)
