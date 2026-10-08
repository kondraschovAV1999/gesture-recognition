import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.configuration.schema import (
    VisualizationSelectionConfig,
    WindowingConfig,
)
from hand_gesture_recognition.features.time_domain import TimeDomainFeatures
from hand_gesture_recognition.features.frequency_domain import FrequencyDomainFeatures
from hand_gesture_recognition.reporting.visualization import (
    plot_fft_spectrum,
    plot_raw_signal,
    plot_raw_signal_by_class,
    plot_signal_processing_comparison,
    resolve_selection,
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


def test_feature_name_constants():
    assert TimeDomainFeatures.FEATURE_NAMES == [
        "rms",
        "std",
        "min",
        "max",
        "mean",
        "max_over_mean",
        "min_over_max",
    ]
    assert FrequencyDomainFeatures.FEATURE_NAMES == [
        "total_power",
        "peak_frequency",
        "mean_frequency",
        "median_frequency",
    ]


def test_resolve_selection_defaults_to_all():
    sel = VisualizationSelectionConfig()
    assert resolve_selection(sel, ["a", "b"], 3) == {"a": [0, 1, 2], "b": [0, 1, 2]}


def test_resolve_selection_channels_and_by_gesture():
    sel = VisualizationSelectionConfig(
        gestures=["a", "b"], channels=[1, 3], by_gesture={"b": [2]}
    )
    assert resolve_selection(sel, ["a", "b", "c"], 4) == {"a": [0, 2], "b": [1]}


def test_plot_raw_signal_by_class_respects_selection(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2,
        n_trials_per_subject=3,
        n_classes=2,
        length=100,
        n_channels=4,
        seed=0,
    )
    sel = VisualizationSelectionConfig(gestures=["class_0"], channels=[1, 2])
    selection = resolve_selection(sel, ds.metadata.class_names, 4)
    paths = plot_raw_signal_by_class(
        ds,
        str(tmp_path),
        class_names=ds.metadata.class_names,
        selection=selection,
        consistent_scale=True,
    )
    assert len(paths) == 1
    assert os.path.exists(str(tmp_path / "raw_signal_class_0.png"))


def test_consistent_scale_shares_ylim():
    ds = make_synthetic_dataset(
        n_subjects=1,
        n_trials_per_subject=1,
        n_classes=1,
        length=100,
        n_channels=3,
        seed=0,
    )
    ds.trials[0][:, 1] *= 100.0
    ds.trials[0][:, 2] *= 0.01
    fig = plot_raw_signal(ds, channels=[0, 1, 2], consistent_scale=True)
    ylims = [tuple(ax.get_ylim()) for ax in fig.axes]
    assert ylims[0] == ylims[1] == ylims[2]
    plt.close(fig)
