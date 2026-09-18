from __future__ import annotations

import argparse

from hand_gesture_recognition.reporting.comparison import compare_runs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare a metric across MLflow runs (by run name)."
    )
    p.add_argument("--runs", nargs="+", required=True, help="run names to compare")
    p.add_argument("--metric", default="macro_f1", help="metric to compare")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="hand_gesture_recognition")
    p.add_argument(
        "--output", default=None, help="optional CSV path (also writes a PNG)"
    )
    args = p.parse_args()

    df = compare_runs(
        args.runs,
        tracking_uri=args.tracking_uri,
        metric=args.metric,
        experiment_name=args.experiment,
        output=args.output,
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
