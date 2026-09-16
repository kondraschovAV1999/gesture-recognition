from __future__ import annotations

from typing import List

import numpy as np
import torch

from hand_gesture_recognition.data.windowing.base import WindowedDataset
from hand_gesture_recognition.prediction.base import Prediction, PredictionAdapter


class LightningPredictionAdapter(PredictionAdapter):
    """Normalize `Trainer.predict()` output into the shared `Prediction` schema.

    `raw_predictions` is a list of batched logit tensors (from the model's
    `predict_step`); softmax yields `y_score`, argmax yields `y_pred`.
    """

    def to_prediction(
        self,
        windowed: WindowedDataset,
        raw_predictions: List[torch.Tensor],
        run_id: str,
        duration_s: float,
    ) -> Prediction:
        if raw_predictions:
            logits = torch.cat(raw_predictions, dim=0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = logits.argmax(dim=1).cpu().numpy()
        else:
            probs = np.empty((0, 0), dtype=np.float32)
            y_pred = np.empty(0, dtype=int)

        return Prediction(
            run_id=run_id,
            sample_ids=windowed.metadata.trial_ids,
            y_true=windowed.y,
            y_pred=y_pred,
            y_score=probs,
            prediction_duration_s=duration_s,
            backend="neural",
        )
