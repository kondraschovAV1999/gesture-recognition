from __future__ import annotations

from ._helpers import _shared_ylim, _trial_index

import os
from dataclasses import replace
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate

from hand_gesture_recognition.configuration.schema import Dataset


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
        cur = cur.replace_trials(
            [replace(t, emg=step.transform(t.emg)) for t in cur.trials]
        )
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
            t = np.arange(trial.emg.shape[0]) / fs
            axes[r, c].plot(t, trial.emg[:, ch], lw=0.6)
            traces.append(trial.emg[:, ch])
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
