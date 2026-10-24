import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.utils.fixtures import make_synthetic_dataset
from hand_gesture_recognition.configuration.schema import (
    VisualizationSelectionConfig,
    WindowingConfig,
)
from hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain import (
    TimeDomainFeatures,
)
from hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.frequency_domain import (
    FrequencyDomainFeatures,
)
from hand_gesture_recognition.utils.observability.visualization import (
    compute_stages,
    plot_feature_over_windows,
    plot_fft_spectrum,
    plot_raw_signal,
    plot_raw_signal_by_class,
    plot_signal_processing_comparison,
    plot_stage_progression,
    resolve_selection,
    stage_label,
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
    proc = raw.replace_trials([replace(t, emg=np.abs(t.emg)) for t in raw.trials])
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
    ds.trials[0].emg[:, 1] *= 100.0
    ds.trials[0].emg[:, 2] *= 0.01
    fig = plot_raw_signal(ds, channels=[0, 1, 2], consistent_scale=True)
    ylims = [tuple(ax.get_ylim()) for ax in fig.axes]
    assert ylims[0] == ylims[1] == ylims[2]
    plt.close(fig)


def test_stage_label_and_compute_stages_order():
    ds = make_synthetic_dataset(
        n_subjects=1,
        n_trials_per_subject=1,
        n_classes=1,
        length=100,
        n_channels=2,
        seed=0,
    )
    steps = [
        {
            "_target_": "hand_gesture_recognition.pipeline.preprocessing.signal.RectifyTransformer"
        }
    ]
    stages = compute_stages(ds, steps, None)
    labels = [s[0] for s in stages]
    assert labels == ["raw", "rectified"]
    assert stage_label(steps[0]) == "rectified"


def test_plot_stage_progression_grid(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    ds = make_synthetic_dataset(
        n_subjects=2,
        n_trials_per_subject=3,
        n_classes=2,
        length=100,
        n_channels=4,
        seed=0,
    )
    steps = [
        {
            "_target_": "hand_gesture_recognition.pipeline.preprocessing.signal.RectifyTransformer"
        }
    ]
    stages = compute_stages(ds, steps, None)  # [("raw", ds), ("rectified", ds)]
    selection = {"class_0": [0, 1]}

    shapes = []
    orig_subplots = plt.subplots

    def spy(*args, **kwargs):
        shapes.append((args[0], args[1]))
        return orig_subplots(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", spy)

    paths = plot_stage_progression(
        stages, selection, ds.metadata.class_names, True, str(tmp_path)
    )
    assert len(paths) == 1  # one figure per gesture (not per stage)
    assert os.path.exists(str(tmp_path / "stage_progression_class_0.png"))
    assert shapes == [(2, 2)]  # rows = channels (2), columns = stages (2)


def test_plot_feature_over_windows(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2,
        n_trials_per_subject=3,
        n_classes=2,
        length=100,
        n_channels=4,
        seed=0,
    )
    windowing = WindowingConfig(window_size_samples=40, stride_samples=20)
    selection = {"class_0": [0, 1]}
    paths = plot_feature_over_windows(
        ds, windowing, selection, ["rms"], 0, ds.metadata.class_names, str(tmp_path)
    )
    assert len(paths) == 1
    assert os.path.exists(str(tmp_path / "feature_over_windows_class_0.png"))


def test_plot_raw_signal_by_class_maps_original_labels(monkeypatch, tmp_path):
    import numpy as np

    from hand_gesture_recognition.configuration.schema import (
        VisualizationSelectionConfig,
    )
    from hand_gesture_recognition.utils.datasets import (
        Dataset,
        DatasetMetadata,
        Trial,
        User,
    )
    from hand_gesture_recognition.utils.observability import visualization as viz
    from hand_gesture_recognition.utils.observability.visualization import (
        plot_raw_signal_by_class,
        resolve_selection,
    )

    # trial 0 = noGesture (label 0), trial 1 = fist (label 1)
    ds = Dataset(
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["ch1", "ch2"],
            class_names=["noGesture", "fist"],
            content_hash="h",
        ),
        users=[
            User(
                user_id="s0",
                trials=[
                    Trial(
                        trial_id="s0_t0",
                        user_id="s0",
                        gesture_name="noGesture",
                        start=-1,
                        end=-1,
                        emg=np.zeros((50, 2), dtype=np.float32),
                    ),
                    Trial(
                        trial_id="s0_t1",
                        user_id="s0",
                        gesture_name="fist",
                        start=0,
                        end=50,
                        emg=np.ones((50, 2), dtype=np.float32),
                    ),
                ],
            )
        ],
    )

    captured = {}

    def fake_plot_raw_signal(dataset, example_tid=None, **kwargs):
        captured["tid"] = example_tid
        return None

    monkeypatch.setattr(viz, "plot_raw_signal", fake_plot_raw_signal)

    sel = VisualizationSelectionConfig(gestures=["fist"])
    selection = resolve_selection(sel, ["noGesture", "fist"], 2)
    plot_raw_signal_by_class(
        ds, str(tmp_path), class_names=["noGesture", "fist"], selection=selection
    )
    assert captured["tid"] == "s0_t1"  # fist trial, NOT the noGesture trial s0_t0


import numpy as np

from hand_gesture_recognition.pipeline.models.classical.adapter import (
    SklearnPredictionAdapter,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.gesture import (
    window_gestures,
)
from hand_gesture_recognition.utils.observability.visualization import (
    plot_predictions_on_signal,
    plot_signal_data_flow,
)


def test_plot_predictions_on_signal_classification_task(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2,
        n_trials_per_subject=2,
        n_classes=2,
        length=50,
        n_channels=2,
        seed=0,
    )
    windowing = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_gestures(ds, windowing)
    pred = SklearnPredictionAdapter().to_prediction(
        windowed=wd,
        y_pred=wd.y.copy(),
        y_score=None,
        run_id="test",
        duration_s=0.0,
    )
    path = str(tmp_path / "pred_on_signal.png")
    fig = plot_predictions_on_signal(
        ds, pred, windowing, ds.metadata.class_names, path=path, task="classification"
    )
    assert fig is not None
    assert os.path.exists(path)


def test_plot_signal_data_flow_classification_task(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2,
        n_trials_per_subject=2,
        n_classes=2,
        length=50,
        n_channels=2,
        seed=0,
    )
    windowing = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_gestures(ds, windowing)
    pred = SklearnPredictionAdapter().to_prediction(
        windowed=wd,
        y_pred=wd.y.copy(),
        y_score=np.eye(2)[wd.y],
        run_id="test",
        duration_s=0.0,
    )
    path = str(tmp_path / "data_flow.png")
    fig = plot_signal_data_flow(
        ds,
        ds,
        ds,
        pred,
        windowing,
        ds.metadata.class_names,
        path=path,
        task="classification",
    )
    assert fig is not None
    assert os.path.exists(path)
