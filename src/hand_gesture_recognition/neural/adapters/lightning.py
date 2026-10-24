from __future__ import annotations

import numpy as np
import torch

from hand_gesture_recognition.prediction.base import Prediction, PredictionAdapter


class LightningPredictionAdapter(PredictionAdapter):
    """Normalize `Trainer.predict()` output into the shared `Prediction` schema.

    `raw_predictions` is a list of per-batch dicts (from the model's
    `predict_step`), each carrying logits and metadata; softmax yields
    `y_score`, argmax yields `y_pred`.
    """

    def to_prediction(
        self, raw_predictions, run_id: str, duration_s: float
    ) -> Prediction:
        if raw_predictions:
            logits = torch.cat([p["logits"] for p in raw_predictions], dim=0)
            y_true = (
                torch.cat([p["y_true"] for p in raw_predictions], dim=0).cpu().numpy()
            )
            window_ids = np.concatenate(
                [np.asarray(p["window_id"], dtype=object) for p in raw_predictions]
            )
            subject_ids = np.concatenate(
                [np.asarray(p["subject_id"], dtype=object) for p in raw_predictions]
            )
            trial_ids = np.concatenate(
                [np.asarray(p["trial_id"], dtype=object) for p in raw_predictions]
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = logits.argmax(dim=1).cpu().numpy()
        else:
            y_true = np.empty(0, dtype=int)
            window_ids = np.empty(0, dtype=object)
            subject_ids = np.empty(0, dtype=object)
            trial_ids = np.empty(0, dtype=object)
            probs = np.empty((0, 0), dtype=np.float32)
            y_pred = np.empty(0, dtype=int)

        return Prediction(
            run_id=run_id,
            sample_ids=window_ids,
            subject_ids=subject_ids,
            trial_ids=trial_ids,
            y_true=y_true,
            y_pred=y_pred,
            y_score=probs,
            prediction_duration_s=duration_s,
            backend="neural",
        )
