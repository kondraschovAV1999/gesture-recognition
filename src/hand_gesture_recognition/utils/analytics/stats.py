from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import numpy as np

from hand_gesture_recognition.utils.datasets import Dataset


def dataset_summary(dataset: Dataset) -> Dict[str, Any]:
    label_names = dataset.metadata.class_names
    class_counts = Counter(int(l) for l in dataset.labels)
    all_signals = np.vstack([t.emg for t in dataset.trials])
    per_channel = {
        ch: {
            "mean": float(all_signals[:, i].mean())
            if np.isfinite(all_signals[:, i].mean())
            else None,
            "std": float(np.nanstd(all_signals[:, i])),
            "min": float(np.nanmin(all_signals[:, i])),
            "max": float(np.nanmax(all_signals[:, i])),
            "rms": float(np.sqrt(np.nanmean(all_signals[:, i] ** 2))),
        }
        for i, ch in enumerate(dataset.metadata.channels)
    }
    return {
        "name": dataset.metadata.name,
        "content_hash": dataset.metadata.content_hash,
        "sampling_rate": dataset.metadata.sampling_rate,
        "n_trials": dataset.n_trials,
        "n_subjects": int(np.unique(dataset.subject_ids).size),
        "n_channels": len(dataset.metadata.channels),
        "class_distribution": {
            label_names[int(k)]: v for k, v in sorted(class_counts.items())
        },
        "per_channel": per_channel,
    }
