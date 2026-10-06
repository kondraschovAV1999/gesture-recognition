from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DatasetMetadata:
    name: str
    version: str
    sampling_rate: float
    channels: List[str]
    class_names: List[str]
    content_hash: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    drop_rest_windows: bool = False  # windowing drops (not labels) rest windows


@dataclass
class Dataset:
    """Raw, unwindowed dataset: one trial = one continuous signal.

    trials[i] has shape (time_steps, channels) and may vary in length across i.
    labels, subject_ids, trial_ids, gesture_start, gesture_end are aligned with
    trials. gesture_start/gesture_end are EMG sample indices delimiting the
    gesture segment within the trial; both are -1 for rest (no-gesture) trials.
    """

    trials: List[np.ndarray]
    labels: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    gesture_start: np.ndarray
    gesture_end: np.ndarray
    metadata: DatasetMetadata
    subject_info: Optional[pd.DataFrame] = None  # index=subject_id, cols=demographics

    @property
    def n_trials(self) -> int:
        return len(self.trials)
