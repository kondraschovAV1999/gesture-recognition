from __future__ import annotations

import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import window_dataset
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


def plot_raw_signal(
    dataset: Dataset,
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
):
    """One trial's raw signal: every channel on its own subplot, with the
    ground-truth gesture segment shaded."""
    if dataset.n_trials == 0:
        return None
    tid = example_tid or str(dataset.trial_ids[0])
    i = _trial_index(dataset, tid)
    trial = dataset.trials[i]
    fs = dataset.metadata.sampling_rate
    n_ch = trial.shape[1]

    fig, axes = _channel_axes(n_ch)
    t = np.arange(trial.shape[0]) / fs
    for c in range(n_ch):
        axes[c].plot(t, trial[:, c], lw=0.6)
        axes[c].set_ylabel(f"ch{c + 1}")
    _shade_gesture(axes, int(dataset.gesture_start[i]), int(dataset.gesture_end[i]), fs)

    axes[0].set_title(f"raw signal — {tid}")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_raw_signal_by_class(
    dataset: Dataset, run_dir: str, class_names: Optional[List[str]] = None
) -> List[str]:
    """Write one raw-signal figure per class present in the dataset.

    Returns the list of file paths written (for artifact logging). Classes with
    no trials in this dataset are naturally absent (skipped by iterating over
    the unique labels actually present).
    """
    paths: List[str] = []
    if dataset.n_trials == 0:
        return paths
    names = class_names if class_names is not None else dataset.metadata.class_names
    for cls in np.unique(dataset.labels):
        cls = int(cls)
        idx = np.where(dataset.labels == cls)[0]
        tid = str(dataset.trial_ids[idx[0]])
        name = names[cls] if cls < len(names) else str(cls)
        path = os.path.join(run_dir, f"raw_signal_{name}.png")
        plot_raw_signal(dataset, example_tid=tid, path=path)
        paths.append(path)
    return paths


def plot_predictions_on_signal(
    dataset: Dataset,
    prediction: Prediction,
    windowing,
    class_names: List[str],
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
):
    """One example trial's signal, every channel on its own subplot, with the
    ground-truth gesture segment shaded and per-window predicted labels
    annotated on channel 1 (green = correct, red = wrong)."""
    if dataset.n_trials == 0:
        return None
    test_wd = window_dataset(dataset, windowing)
    fs = dataset.metadata.sampling_rate
    ws = test_wd.X.shape[1]

    tid = example_tid
    if tid is None:
        # pick the first non-noGesture trial as a meaningful example
        for i in range(dataset.n_trials):
            if int(dataset.labels[i]) != 0:
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
    test_wd = window_dataset(test_dataset, windowing)
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
            if int(raw_dataset.labels[i]) != 0:
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
