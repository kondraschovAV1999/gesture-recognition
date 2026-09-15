from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class DatasetMetadata:
    name: str
    version: str
    sampling_rate: float
    channels: List[str]
    class_names: List[str]
    content_hash: str
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """Raw, unwindowed dataset: one trial = one continuous signal.

    trials[i] has shape (time_steps, channels) and may vary in length across i.
    labels, subject_ids, trial_ids are aligned with trials.
    """

    trials: List[np.ndarray]
    labels: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    metadata: DatasetMetadata

    @property
    def n_trials(self) -> int:
        return len(self.trials)
