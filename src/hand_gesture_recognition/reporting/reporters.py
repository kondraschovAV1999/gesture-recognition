from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, Optional

import numpy as np

from hand_gesture_recognition.configuration.schema import ExperimentConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.pipeline.evaluation.base import EvaluationResult


def render_report(
    result: EvaluationResult,
    config: ExperimentConfig,
    dataset: Optional[Dataset],
) -> Dict[str, Any]:
    return {
        "run_id": result.run_id,
        "dataset_id": result.dataset_id,
        "backend": config.backend,
        "seed": config.seed,
        "dataset": {
            "name": dataset.metadata.name,
            "content_hash": dataset.metadata.content_hash,
        }
        if dataset
        else {},
        "split": result.split,
        "metrics": result.metrics,
        "confusion_matrix": result.confusion_matrix.tolist(),
    }


class JsonReporter:
    def write(self, report: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


class HtmlReporter:
    def write(self, report: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        metrics = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
            for k, v in report["metrics"].items()
        )
        doc = (
            "<html><head><title>Experiment Report</title></head><body>"
            f"<h1>Run {html.escape(report['run_id'])}</h1>"
            f"<table border='1'><tr><th>Metric</th><th>Value</th></tr>{metrics}</table>"
            "</body></html>"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
