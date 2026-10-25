from __future__ import annotations

from typing import List

import numpy as np

from hand_gesture_recognition.configuration.schema import EvaluationResult


def aggregate_folds(
    results: List[EvaluationResult], run_id: str, dataset_id: str
) -> EvaluationResult:
    """Mean and std of each metric across folds, as a single EvaluationResult."""
    if not results:
        return EvaluationResult(
            run_id=run_id, dataset_id=dataset_id, split={"cv": {"n_folds": 0}}
        )
    metrics: dict = {}
    for m in results[0].metrics:
        vals = [r.metrics[m] for r in results]
        metrics[m] = float(np.mean(vals))
        metrics[f"{m}_std"] = float(np.std(vals))
    return EvaluationResult(
        run_id=run_id,
        dataset_id=dataset_id,
        split={"cv": {"n_folds": len(results)}},
        metrics=metrics,
    )
