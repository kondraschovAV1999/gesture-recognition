from __future__ import annotations

import argparse

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="compose", choices=["compose", "replay"])
    p.add_argument("--resolved", default=None, help="path to resolved.yaml for replay")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(
        mode=args.mode, overrides=args.overrides, resolved_path=args.resolved
    )
    ExperimentRunner(cfg).run()


if __name__ == "__main__":
    main()
