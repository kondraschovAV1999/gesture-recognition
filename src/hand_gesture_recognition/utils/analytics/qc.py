from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from hand_gesture_recognition.configuration.schema import Dataset


def run_quality_checks(dataset: Dataset) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for i, trial in enumerate(dataset.trials):
        emg = trial.emg
        if np.isnan(emg).any() or np.isinf(emg).any():
            checks.append(
                {
                    "level": "error",
                    "check": "nan_inf",
                    "message": f"trial {dataset.trial_ids[i]} contains NaN/Inf values",
                }
            )
    counts = np.bincount(dataset.labels.astype(int))
    if counts.size and counts.min() == 0:
        checks.append(
            {
                "level": "warning",
                "check": "empty_class",
                "message": "one or more classes have zero trials",
            }
        )
    if not checks:
        checks.append({"level": "info", "check": "ok", "message": "no issues detected"})
    return checks
