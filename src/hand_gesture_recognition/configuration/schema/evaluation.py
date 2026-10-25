from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "balanced_accuracy", "macro_f1"]
    )
    average: str = "macro"
    exclude_classes: List[int] = field(default_factory=list)
    breakdown_by: List[str] = field(
        default_factory=list
    )  # subset of {"subject_id","trial_id","y_true"}


@dataclass
class EvaluationResult:
    run_id: str
    dataset_id: str
    split: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
