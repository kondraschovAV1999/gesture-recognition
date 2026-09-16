import json

import numpy as np

from hand_gesture_recognition.configuration.schema import ExperimentConfig
from hand_gesture_recognition.evaluation.base import EvaluationResult
from hand_gesture_recognition.reporting.reporters import JsonReporter, render_report


def test_render_report_serializable():
    res = EvaluationResult(
        run_id="r1",
        dataset_id="d1",
        metrics={"accuracy": 0.9},
        confusion_matrix=np.array([[5, 0], [0, 5]]),
    )
    report = render_report(res, ExperimentConfig(run_id="r1"), dataset=None)
    s = json.dumps(report)
    assert "accuracy" in s
    assert report["run_id"] == "r1"


def test_json_reporter_writes_file(tmp_path):
    res = EvaluationResult(run_id="r1", dataset_id="d1", metrics={"accuracy": 0.9})
    report = render_report(res, ExperimentConfig(run_id="r1"), dataset=None)
    p = tmp_path / "report.json"
    JsonReporter().write(report, str(p))
    assert json.loads(p.read_text())["run_id"] == "r1"
