import logging

from hand_gesture_recognition.evaluation.base import evaluate

logger = logging.getLogger("hand_gesture_recognition")


def evaluate_grouped(prediction, by, config, dataset_id, class_names=None):
    """Split a Prediction by a lineage key and evaluate each group.

    `by` is one of "subject_id", "trial_id", or "y_true" (per-gesture). Returns
    a dict {group_key: EvaluationResult}. Empty inputs are logged and return an
    empty dict.
    """
    import numpy as np
    from dataclasses import replace

    if by == "subject_id":
        keys = np.asarray(prediction.subject_ids)
    elif by == "trial_id":
        keys = np.asarray(prediction.trial_ids)
    elif by == "y_true":
        keys = np.asarray(prediction.y_true, dtype=int)
    else:
        raise ValueError(f"unknown grouping key: {by!r}")

    if keys.size == 0:
        logger.warning("evaluate_grouped: empty prediction for by=%r", by)
        return {}

    out = {}
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        sub = replace(
            prediction,
            sample_ids=np.asarray(prediction.sample_ids)[mask],
            subject_ids=np.asarray(prediction.subject_ids)[mask],
            trial_ids=np.asarray(prediction.trial_ids)[mask],
            y_true=np.asarray(prediction.y_true)[mask],
            y_pred=np.asarray(prediction.y_pred)[mask],
            y_score=prediction.y_score[mask]
            if prediction.y_score is not None
            else None,
        )
        out[key] = evaluate(sub, config, dataset_id, {"group": by}, class_names)
    return out
