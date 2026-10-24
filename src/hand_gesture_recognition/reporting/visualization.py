from __future__ import annotations

import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate

from hand_gesture_recognition.configuration.schema import VisualizationSelectionConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import window_dataset
from hand_gesture_recognition.data.windowing.strategy import get_windowing_strategy
from hand_gesture_recognition.features.frequency_domain import FrequencyDomainFeatures
from hand_gesture_recognition.features.time_domain import TimeDomainFeatures
from hand_gesture_recognition.prediction.base import Prediction


def _channel_axes(n_channels: int):
    fig, axes = plt.subplots(
        n_channels, 1, figsize=(12, 1.9 * n_channels), squeeze=False
    )
    return fig, axes[:, 0]


def _trial_index(dataset: Dataset, tid: str) -> int:
    for i, t in enumerate(dataset.trial_ids.tolist()):
        if str(t) == tid:
            return i
    raise KeyError(tid)


def _shade_gesture(axes, g_start: int, g_end: int, fs: float) -> None:
    if g_start >= 0:
        for ax in axes:
            ax.axvspan(g_start / fs, g_end / fs, color="green", alpha=0.2)


def _shared_ylim(axes, traces, pad=0.05):
    """Set a single shared y-range across all axes from the given traces."""
    lo, hi = np.inf, -np.inf
    for arr in traces:
        if arr is None or arr.size == 0:
            continue
        lo = min(lo, float(np.nanmin(arr)))
        hi = max(hi, float(np.nanmax(arr)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return
    span = hi - lo
    for ax in axes:
        ax.set_ylim(lo - pad * span, hi + pad * span)


def resolve_selection(
    selection: VisualizationSelectionConfig, class_names: List[str], n_channels: int
) -> Dict[str, List[int]]:
    """Gesture name -> 0-based channel indices. Empty lists mean 'all'."""
    gestures = selection.gestures or list(class_names)
    all_channels = list(range(n_channels))
    out: Dict[str, List[int]] = {}
    for g in gestures:
        ch = selection.by_gesture.get(g)
        if ch is None:
            ch = selection.channels
        idx = [int(c) - 1 for c in ch] if ch else all_channels
        out[g] = sorted(set(idx))
    return out


def plot_raw_signal(
    dataset: Dataset,
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
    channels: Optional[List[int]] = None,
    consistent_scale: bool = False,
):
    """One trial's raw signal for the given channels (0-based), with the
    ground-truth gesture segment shaded. When ``consistent_scale`` is true, all
    channel subplots share one y-range."""
    if dataset.n_trials == 0:
        return None
    tid = example_tid or str(dataset.trial_ids[0])
    i = _trial_index(dataset, tid)
    trial = dataset.trials[i]
    fs = dataset.metadata.sampling_rate
    n_ch = trial.shape[1]
    if channels is None:
        channels = list(range(n_ch))

    fig, axes = plt.subplots(
        len(channels), 1, figsize=(12, 1.9 * len(channels)), squeeze=False
    )
    axes = axes[:, 0]
    t = np.arange(trial.shape[0]) / fs
    plotted = []
    for row, c in enumerate(channels):
        axes[row].plot(t, trial[:, c], lw=0.6)
        axes[row].set_ylabel(f"ch{c + 1}")
        plotted.append(trial[:, c])
    _shade_gesture(axes, int(dataset.gesture_start[i]), int(dataset.gesture_end[i]), fs)
    if consistent_scale:
        _shared_ylim(axes, plotted)

    axes[0].set_title(f"raw signal — {tid}")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_raw_signal_by_class(
    dataset: Dataset,
    run_dir: str,
    class_names: Optional[List[str]] = None,
    selection: Optional[Dict[str, List[int]]] = None,
    consistent_scale: bool = False,
) -> List[str]:
    """Write one raw-signal figure per class. When ``selection`` is provided,
    only its gestures (with their channels) are plotted; otherwise every class
    with all channels (legacy behavior)."""
    paths: List[str] = []
    if dataset.n_trials == 0:
        return paths
    names = class_names if class_names is not None else dataset.metadata.class_names
    if selection is not None:
        present = set(int(l) for l in np.unique(dataset.labels))
        for g_name, channels in selection.items():
            try:
                g_idx = names.index(g_name)
            except ValueError:
                continue
            if g_idx not in present:
                continue
            idx = np.where(dataset.labels == g_idx)[0]
            tid = str(dataset.trial_ids[idx[0]])
            path = os.path.join(run_dir, f"raw_signal_{g_name}.png")
            plot_raw_signal(
                dataset,
                example_tid=tid,
                path=path,
                channels=channels,
                consistent_scale=consistent_scale,
            )
            paths.append(path)
        return paths
    for cls in np.unique(dataset.labels):
        cls = int(cls)
        idx = np.where(dataset.labels == cls)[0]
        tid = str(dataset.trial_ids[idx[0]])
        name = names[cls] if cls < len(names) else str(cls)
        path = os.path.join(run_dir, f"raw_signal_{name}.png")
        plot_raw_signal(dataset, example_tid=tid, path=path)
        paths.append(path)
    return paths


def stage_label(step_cfg: Dict) -> str:
    target = str(step_cfg.get("_target_", ""))
    cls = target.rsplit(".", 1)[-1]
    if cls == "ButterworthFilter":
        t = step_cfg.get("type")
        if t == "bandpass":
            return f"bandpass_{step_cfg.get('lowcut')}_{step_cfg.get('highcut')}"
        if t == "notch":
            return f"notch_{step_cfg.get('frequency')}"
        if t == "lowpass":
            return f"lowpass_{step_cfg.get('highcut')}"
        if t == "highpass":
            return f"highpass_{step_cfg.get('lowcut')}"
        return str(t or "butterworth")
    if cls == "RectifyTransformer":
        return "rectified"
    if cls == "EnvelopeExtractor":
        return "envelope"
    return cls or "step"


def compute_stages(
    dataset: Dataset, signal_processing_steps: List[Dict], baseline_remover=None
) -> List[Tuple[str, Dataset]]:
    """Snapshot the dataset at each stage, in training-path order:
    raw -> baseline_removed (if any) -> each signal-processing step."""
    stages: List[Tuple[str, Dataset]] = [("raw", dataset)]
    cur = dataset
    if baseline_remover is not None:
        cur = baseline_remover.transform(cur)
        stages.append(("baseline_removed", cur))
    for step_cfg in signal_processing_steps:
        step = instantiate(step_cfg)
        cur = replace(cur, trials=[step.transform(t) for t in cur.trials])
        stages.append((stage_label(step_cfg), cur))
    return stages


def plot_stage_progression(
    stages: List[Tuple[str, Dataset]],
    selection: Dict[str, List[int]],
    class_names: List[str],
    consistent_scale: bool,
    run_dir: str,
) -> List[str]:
    """One figure per selected gesture: rows = channels, columns = stages,
    sharing one y-scale across the whole figure."""
    paths: List[str] = []
    if not stages:
        return paths
    raw = stages[0][1]
    if raw.n_trials == 0:
        return paths
    names = class_names if class_names is not None else raw.metadata.class_names
    present = set(int(l) for l in np.unique(raw.labels))
    stage_labels = [label for label, _ in stages]
    for g_name, channels in selection.items():
        try:
            g_idx = names.index(g_name)
        except ValueError:
            continue
        if g_idx not in present:
            continue
        idx = np.where(raw.labels == g_idx)[0]
        tid = str(raw.trial_ids[idx[0]])
        path = os.path.join(run_dir, f"stage_progression_{g_name}.png")
        _plot_stage_grid(
            stages, tid, channels, stage_labels, g_name, consistent_scale, path
        )
        paths.append(path)
    return paths


def _plot_stage_grid(
    stages: List[Tuple[str, Dataset]],
    tid: str,
    channels: List[int],
    stage_labels: List[str],
    gesture: str,
    consistent_scale: bool,
    path: str,
):
    """Grid of the example trial at every stage: rows = channels, columns =
    stages, one shared y-scale across all cells when consistent_scale."""
    fs = stages[0][1].metadata.sampling_rate
    trial_indices = [_trial_index(ds, tid) for _, ds in stages]
    n_ch = len(channels)
    n_st = len(stages)
    fig, axes = plt.subplots(
        n_ch, n_st, figsize=(3.0 * n_st, 1.8 * n_ch), squeeze=False
    )
    traces = []
    for r, ch in enumerate(channels):
        for c, (_, ds) in enumerate(stages):
            trial = ds.trials[trial_indices[c]]
            t = np.arange(trial.shape[0]) / fs
            axes[r, c].plot(t, trial[:, ch], lw=0.6)
            traces.append(trial[:, ch])
            if r == 0:
                axes[r, c].set_title(stage_labels[c], fontsize=8)
            if c == 0:
                axes[r, c].set_ylabel(f"ch{ch + 1}")
    if consistent_scale:
        _shared_ylim(axes.flat, traces)
    for c in range(n_st):
        axes[n_ch - 1, c].set_xlabel("time (s)")
    fig.suptitle(f"stage progression — {gesture}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def _feature_matrix(X: np.ndarray, sampling_rate: float):
    """(n_windows, 11*n_channels) matrix + flattened per-column names."""
    td = TimeDomainFeatures().transform(X)
    fd = FrequencyDomainFeatures(sampling_rate=sampling_rate).transform(X)
    n_ch = X.shape[-1]
    td_names = [
        f"{n}_ch{c + 1}" for n in TimeDomainFeatures.FEATURE_NAMES for c in range(n_ch)
    ]
    fd_names = [
        f"{n}_ch{c + 1}"
        for n in FrequencyDomainFeatures.FEATURE_NAMES
        for c in range(n_ch)
    ]
    return np.concatenate([td, fd], axis=1), td_names + fd_names


def plot_feature_over_windows(
    processed_dataset: Dataset,
    windowing,
    selection: Dict[str, List[int]],
    features: List[str],
    max_windows: int,
    class_names: List[str],
    run_dir: str,
) -> List[str]:
    """One figure per selected gesture: rows = features, cols = channels,
    value vs window index."""
    paths: List[str] = []
    if processed_dataset.n_trials == 0:
        return paths
    names = (
        class_names
        if class_names is not None
        else processed_dataset.metadata.class_names
    )
    present = set(int(l) for l in np.unique(processed_dataset.labels))
    if getattr(windowing, "mode", "single") == "two_level":
        # Two-level mode filters per big window (small windows are carved from
        # the filtered big window), so features belong to the big windows.
        wd = (
            get_windowing_strategy("two_level")
            .window(processed_dataset, windowing, [])
            .big
        )
    else:
        wd = window_dataset(processed_dataset, windowing)
    if wd.X.shape[0] == 0:
        return paths
    fs = processed_dataset.metadata.sampling_rate
    X_feat, feat_names = _feature_matrix(wd.X, fs)

    for g_name, channels in selection.items():
        try:
            g_idx = names.index(g_name)
        except ValueError:
            continue
        if g_idx not in present:
            continue
        idx = np.where(processed_dataset.labels == g_idx)[0]
        tid = str(processed_dataset.trial_ids[idx[0]])
        mask = wd.metadata.trial_ids == tid
        if not mask.any():
            continue
        path = os.path.join(run_dir, f"feature_over_windows_{g_name}.png")
        _plot_features(
            X_feat[mask], feat_names, channels, features, max_windows, g_name, path
        )
        paths.append(path)
    return paths


def _plot_features(X_feat, feat_names, channels, features, max_windows, g_name, path):
    wanted = features or list(dict.fromkeys(n.rsplit("_ch", 1)[0] for n in feat_names))
    if max_windows and X_feat.shape[0] > max_windows:
        X_feat = X_feat[:max_windows]
    fig, axes = plt.subplots(
        len(wanted),
        len(channels),
        figsize=(2.5 * len(channels), 1.8 * len(wanted)),
        squeeze=False,
    )
    w = np.arange(X_feat.shape[0])
    for r, feat in enumerate(wanted):
        row_traces = []
        for c_idx, ch in enumerate(channels):
            col = f"{feat}_ch{ch + 1}"
            ax = axes[r, c_idx]
            if col in feat_names:
                j = feat_names.index(col)
                ax.plot(w, X_feat[:, j], lw=0.8)
                row_traces.append(X_feat[:, j])
            else:
                ax.set_visible(False)
            ax.set_xlabel("window")
            if c_idx == 0:
                ax.set_ylabel(feat)
        if row_traces:
            _shared_ylim(axes[r, :], row_traces)
    axes[0, 0].set_title(f"features over windows — {g_name}")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_predictions_on_signal(
    dataset: Dataset,
    prediction: Prediction,
    windowing,
    class_names: List[str],
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
    task: str = "recognition",
):
    """One example trial's signal, every channel on its own subplot, with the
    ground-truth gesture segment shaded and per-window predicted labels
    annotated on channel 1 (green = correct, red = wrong)."""
    if dataset.n_trials == 0:
        return None
    test_wd = get_windowing_strategy(task).window(dataset, windowing, None)
    fs = dataset.metadata.sampling_rate
    ws = test_wd.X.shape[1]

    tid = example_tid
    if tid is None:
        # pick the first gesture trial as a meaningful example (rest trials
        # have gesture_start < 0)
        for i in range(dataset.n_trials):
            if int(dataset.gesture_start[i]) >= 0:
                tid = str(dataset.trial_ids[i])
                break
        tid = tid or str(dataset.trial_ids[0])

    i = _trial_index(dataset, tid)
    trial = dataset.trials[i]
    g_start, g_end = int(dataset.gesture_start[i]), int(dataset.gesture_end[i])
    true_label = int(dataset.labels[i])
    n_ch = trial.shape[1]

    fig, axes = _channel_axes(n_ch)
    t = np.arange(trial.shape[0]) / fs
    for c in range(n_ch):
        axes[c].plot(t, trial[:, c], lw=0.6)
        axes[c].set_ylabel(f"ch{c + 1}")
    _shade_gesture(axes, g_start, g_end, fs)

    # per-window predicted labels on channel 1
    mask = test_wd.metadata.trial_ids == tid
    starts = test_wd.metadata.window_start[mask]
    y_pred = prediction.y_pred[mask]
    y_true = test_wd.y[mask]
    ylim = axes[0].get_ylim()
    for s, yp, yt in zip(starts, y_pred, y_true):
        color = "green" if yp == yt else "red"
        x = (s + ws / 2) / fs
        axes[0].axvline(s / fs, color=color, alpha=0.3, lw=0.5)
        axes[0].annotate(
            class_names[int(yp)],
            xy=(x, ylim[1] * 0.9),
            color=color,
            fontsize=6,
            ha="center",
            va="top",
            rotation=90,
        )
    axes[0].set_ylim(ylim)

    axes[0].set_title(
        f"{tid} — expected: {class_names[true_label]} (green=correct, red=wrong)"
    )
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_signal_data_flow(
    raw_dataset: Dataset,
    processed_dataset: Dataset,
    test_dataset: Dataset,
    prediction: Prediction,
    windowing,
    class_names: List[str],
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
    task: str = "recognition",
):
    """4-panel data flow: raw ch1, processed ch1, per-window prediction
    probabilities, and expected-vs-predicted labels."""
    if test_dataset.n_trials == 0:
        return None
    tid = example_tid or str(test_dataset.trial_ids[0])
    fs = processed_dataset.metadata.sampling_rate

    def find_trial(ds, tid):
        try:
            return ds.trials[_trial_index(ds, tid)]
        except KeyError:
            return None

    raw_trial = find_trial(raw_dataset, tid)
    proc_trial = find_trial(processed_dataset, tid)
    test_wd = get_windowing_strategy(task).window(test_dataset, windowing, None)
    mask = test_wd.metadata.trial_ids == tid

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if raw_trial is not None:
        axes[0, 0].plot(np.arange(raw_trial.shape[0]) / fs, raw_trial[:, 0], lw=0.7)
    axes[0, 0].set_title("raw signal (ch1)")
    axes[0, 0].set_xlabel("time (s)")

    if proc_trial is not None:
        axes[0, 1].plot(np.arange(proc_trial.shape[0]) / fs, proc_trial[:, 0], lw=0.7)
    axes[0, 1].set_title("signal-processed (ch1)")
    axes[0, 1].set_xlabel("time (s)")

    y_score = prediction.y_score[mask] if prediction.y_score is not None else None
    if y_score is not None and y_score.shape[0] > 0:
        im = axes[1, 0].imshow(y_score.T, aspect="auto", cmap="viridis")
        axes[1, 0].set_yticks(range(len(class_names)))
        axes[1, 0].set_yticklabels(class_names)
        axes[1, 0].set_title("prediction probabilities (window × class)")
        axes[1, 0].set_xlabel("window index")
        fig.colorbar(im, ax=axes[1, 0])

    y_true = test_wd.y[mask]
    y_pred = prediction.y_pred[mask]
    w = np.arange(len(y_true))
    axes[1, 1].plot(w, y_true, "o-", label="expected", lw=1)
    axes[1, 1].plot(w, y_pred, "x--", label="predicted", lw=1)
    axes[1, 1].set_yticks(range(len(class_names)))
    axes[1, 1].set_yticklabels(class_names)
    axes[1, 1].legend()
    axes[1, 1].set_title("expected vs predicted per window")
    axes[1, 1].set_xlabel("window index")

    fig.suptitle(f"data flow for {tid}")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_fft_spectrum(
    dataset: Dataset,
    windowing,
    sampling_rate: Optional[float] = None,
    example_tid: Optional[str] = None,
    window_index: int = 0,
    max_frequency: Optional[float] = None,
    use_hann: bool = True,
    remove_dc: bool = True,
    path: Optional[str] = None,
):
    """FFT power spectrum for one window: one subplot per channel, with the
    extracted peak/mean/median frequency marked. Mirrors the logic in
    ``FrequencyDomainFeatures`` (DC removal + Hann + real FFT)."""
    fs = sampling_rate if sampling_rate is not None else dataset.metadata.sampling_rate
    if getattr(windowing, "mode", "single") == "two_level":
        # Two-level mode filters per big window, so the FFT is over big windows.
        wd = get_windowing_strategy("two_level").window(dataset, windowing, []).big
    else:
        wd = window_dataset(dataset, windowing)
    if wd.X.shape[0] == 0:
        return None
    if window_index >= wd.X.shape[0]:
        window_index = wd.X.shape[0] - 1
    signal = wd.X[window_index].astype(np.float32)  # (n_samples, n_channels)
    n_samples, n_channels = signal.shape

    if remove_dc:
        signal = signal - np.mean(signal, axis=0, keepdims=True)
    if use_hann:
        signal = signal * np.hanning(n_samples).astype(np.float32)[:, None]

    fft = np.fft.rfft(signal, axis=0)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    power = np.abs(fft) ** 2
    power = power[1:, :]
    frequencies = frequencies[1:]

    fig, axes = _channel_axes(n_channels)
    for c in range(n_channels):
        p = power[:, c]
        total = p.sum()
        peak_freq = frequencies[np.argmax(p)]
        weighted = float(np.sum(p * frequencies))
        mean_freq = weighted / total if total > 0 else 0.0
        cumulative = np.cumsum(p)
        median_freq = frequencies[np.argmax(cumulative >= total / 2.0)]

        axes[c].plot(frequencies, p, lw=0.8)
        axes[c].axvline(peak_freq, color="tab:red", lw=1, label="peak")
        axes[c].axvline(mean_freq, color="tab:green", lw=1, label="mean")
        axes[c].axvline(median_freq, color="tab:blue", lw=1, label="median")
        axes[c].set_ylabel(f"ch{c + 1}")
        if max_frequency is not None:
            axes[c].set_xlim(0, max_frequency)

    axes[0].legend(fontsize=7)
    axes[0].set_title(f"FFT power spectrum — window {window_index}")
    axes[-1].set_xlabel("frequency (Hz)")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_signal_processing_comparison(
    raw_dataset: Dataset,
    processed_dataset: Dataset,
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
):
    """Before/after comparison of signal processing: raw vs processed signal,
    one row per channel (2 columns), for a representative non-noGesture trial."""
    if raw_dataset.n_trials == 0:
        return None
    tid = example_tid
    if tid is None:
        for i in range(raw_dataset.n_trials):
            if int(raw_dataset.gesture_start[i]) >= 0:
                tid = str(raw_dataset.trial_ids[i])
                break
        tid = tid or str(raw_dataset.trial_ids[0])

    i = _trial_index(raw_dataset, tid)
    raw_trial = raw_dataset.trials[i]
    proc_trial = processed_dataset.trials[_trial_index(processed_dataset, tid)]
    fs = raw_dataset.metadata.sampling_rate
    n_ch = raw_trial.shape[1]

    fig, axes = plt.subplots(n_ch, 2, figsize=(12, 1.8 * n_ch), squeeze=False)
    t_raw = np.arange(raw_trial.shape[0]) / fs
    t_proc = np.arange(proc_trial.shape[0]) / fs
    for c in range(n_ch):
        axes[c, 0].plot(t_raw, raw_trial[:, c], lw=0.6)
        axes[c, 1].plot(t_proc, proc_trial[:, c], lw=0.6, color="tab:orange")
        axes[c, 0].set_ylabel(f"ch{c + 1}")
    axes[0, 0].set_title("raw")
    axes[0, 1].set_title("signal-processed")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"signal processing — {tid}")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
