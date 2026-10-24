from __future__ import annotations

from typing import Optional

import numpy as np

from hand_gesture_recognition.pipeline.feature_extraction.windowing.base import WindowedDataset
from hand_gesture_recognition.pipeline.evaluation.prediction import Prediction, PredictionAdapter


class SklearnPredictionAdapter(PredictionAdapter):
    def to_prediction(
        self,
        windowed: WindowedDataset,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray],
        run_id: str,
        duration_s: float,
    ) -> Prediction:
        return Prediction(
            run_id=run_id,
            sample_ids=windowed.metadata.window_ids,
            subject_ids=windowed.metadata.subject_ids,
            trial_ids=windowed.metadata.trial_ids,
            y_true=windowed.y,
            y_pred=np.asarray(y_pred),
            y_score=y_score,
            prediction_duration_s=duration_s,
            backend="classical",
        )
