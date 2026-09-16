from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from hand_gesture_recognition.configuration.schema import EvaluationConfig
from hand_gesture_recognition.prediction.base import Prediction

_METRIC_FNS = {
    "accuracy": lambda yt, yp, **k: accuracy_score(yt, yp),
    "balanced_accuracy": lambda yt, yp, **k: balanced_accuracy_score(yt, yp),
    "macro_f1": lambda yt, yp, **k: f1_score(yt, yp, average="macro"),
    "weighted_f1": lambda yt, yp, **k: f1_score(yt, yp, average="weighted"),
}


@dataclass
class EvaluationResult:
    run_id: str
    dataset_id: str
    split: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def evaluate(
    prediction: Prediction,
    config: EvaluationConfig,
    dataset_id: str,
    split_info: Dict[str, Any],
    class_names: Optional[List[str]] = None,
) -> EvaluationResult:
    y_true = np.asarray(prediction.y_true)
    y_pred = np.asarray(prediction.y_pred)
    metrics = {
        m: _METRIC_FNS[m](y_true, y_pred) for m in config.metrics if m in _METRIC_FNS
    }
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    # Per-class TP/FP/TN/FN derived from the confusion matrix.
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    per_class: Dict[str, Dict[str, float]] = {}
    for i, c in enumerate(classes):
        name = (
            class_names[int(c)]
            if class_names is not None and int(c) < len(class_names)
            else str(c)
        )
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "tn": int(tn[i]),
            "fn": int(fn[i]),
        }
    return EvaluationResult(
        run_id=prediction.run_id,
        dataset_id=dataset_id,
        split=split_info,
        metrics=metrics,
        per_class=per_class,
        confusion_matrix=cm,
    )
