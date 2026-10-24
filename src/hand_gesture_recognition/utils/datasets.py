from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional

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
    no_gesture_name: str = "noGesture"


@dataclass
class Trial:
    trial_id: str
    user_id: str
    gesture_name: str
    start: int
    end: int
    emg: np.ndarray  # (time_steps, channels)

    @property
    def is_rest(self) -> bool:
        return self.start < 0


@dataclass
class User:
    user_id: str
    trials: List[Trial]
    demographics: Optional[Dict[str, Any]] = None


@dataclass
class Dataset:
    """Nested dataset: users own trials; trials own the raw EMG signal.

    ``trials``/``labels``/``subject_ids``/``trial_ids`` are derived, cached
    flat views. Treat instances as immutable (transform via ``replace``).
    """

    metadata: DatasetMetadata
    users: List[User]

    @classmethod
    def from_trials(
        cls,
        trials: List[Trial],
        metadata: DatasetMetadata,
        demographics: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    ) -> "Dataset":
        by_user: Dict[str, List[Trial]] = {}
        order: List[str] = []
        for t in trials:
            if t.user_id not in by_user:
                by_user[t.user_id] = []
                order.append(t.user_id)
            by_user[t.user_id].append(t)
        return cls(
            metadata=metadata,
            users=[User(u, by_user[u], (demographics or {}).get(u)) for u in order],
        )

    def replace_trials(
        self, trials: List[Trial], metadata: Optional[DatasetMetadata] = None
    ) -> "Dataset":
        """Rebuild with a new trial list, preserving per-user demographics."""
        demo = {u.user_id: u.demographics for u in self.users}
        return Dataset.from_trials(
            trials,
            metadata if metadata is not None else self.metadata,
            demographics=demo,
        )

    @classmethod
    def from_users(cls, users: List[User], metadata: DatasetMetadata) -> "Dataset":
        return cls(metadata=metadata, users=list(users))

    @cached_property
    def trials(self) -> List[Trial]:
        return [t for u in self.users for t in u.trials]

    @cached_property
    def labels(self) -> np.ndarray:
        lut = {n: i for i, n in enumerate(self.metadata.class_names)}
        return np.asarray([lut[t.gesture_name] for t in self.trials], dtype=int)

    @cached_property
    def subject_ids(self) -> np.ndarray:
        return np.asarray([t.user_id for t in self.trials])

    @cached_property
    def trial_ids(self) -> np.ndarray:
        return np.asarray([t.trial_id for t in self.trials])

    @property
    def n_trials(self) -> int:
        return len(self.trials)
