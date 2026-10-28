from __future__ import annotations

from typing import Any, Dict, List

from hand_gesture_recognition.configuration.schema.feature_extraction import (
    FREQUENCY_DOMAIN_FEATURE_NAMES,
    TIME_DOMAIN_FEATURE_NAMES,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import (
    get_windowing_strategy,
)


def _features_per_channel(feature_specs: List[Dict[str, Any]]) -> int:
    """Number of handcrafted features per channel from the resolved specs."""
    total = 0
    for spec in feature_specs:
        target = spec.get("_target_", "")
        if "TimeDomainFeatures" in target:
            names = spec.get("features") or TIME_DOMAIN_FEATURE_NAMES
        elif "FrequencyDomainFeatures" in target:
            names = spec.get("features") or FREQUENCY_DOMAIN_FEATURE_NAMES
        else:
            continue
        total += len(names)
    return total


def compute_shape_summary(
    dataset,
    windowing,
    feature_specs: List[Dict[str, Any]],
    task_name: str = "recognition",
) -> Dict[str, Any]:
    """Report the shape at each pipeline stage for a representative trial.

    Picks the first non-rest trial and reports symbolic shapes (channels-first):

    - two_level: input -> big_windows -> small_windows -> features
    - single:    input -> windows -> features

    Returns ``{}`` when there is no non-rest trial.
    """
    trial = next((t for t in dataset.trials if not t.is_rest), None)
    if trial is None:
        return {}

    channels = trial.emg.shape[1]
    datapoints = trial.emg.shape[0]
    summary: Dict[str, Any] = {
        "mode": windowing.mode,
        "input": {"channels": channels, "datapoints_per_channel": datapoints},
    }

    single = dataset.replace_trials([trial])
    features_per_channel = _features_per_channel(feature_specs)

    if windowing.mode == "two_level":
        grouped = get_windowing_strategy("two_level").window(single, windowing, [])
        big_windows = grouped.big.X.shape[0]
        big_window_size = grouped.big.X.shape[1]
        small_windows = (
            grouped.small_windows[0].shape[0] if grouped.small_windows else 0
        )
        small_window_size = (
            grouped.small_windows[0].shape[1] if grouped.small_windows else 0
        )
        summary["big_windows"] = {
            "channels": channels,
            "big_windows": big_windows,
            "big_window_size": big_window_size,
        }
        summary["small_windows"] = {
            "channels": channels,
            "big_windows": big_windows,
            "small_windows": small_windows,
            "small_window_size": small_window_size,
        }
        summary["features"] = {
            "channels": channels,
            "big_windows": big_windows,
            "small_windows": small_windows,
            "features_per_small_window": features_per_channel,
        }
    else:
        strategy = get_windowing_strategy(task_name)
        wd = strategy.window(single, windowing, [])
        windows = wd.X.shape[0]
        window_size = wd.X.shape[1]
        summary["windows"] = {
            "channels": channels,
            "windows": windows,
            "window_size": window_size,
        }
        summary["features"] = {
            "channels": channels,
            "windows": windows,
            "features_per_window": features_per_channel,
        }
    return summary
