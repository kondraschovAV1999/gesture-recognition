from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from hand_gesture_recognition.configuration.errors import ConfigValidationError

_FEATURE_TYPES = ("time_domain", "frequency_domain", "combined")

# Canonical, ordered feature-name catalogs. The extractors import these (via
# their FEATURE_NAMES class attribute) so the schema, training, and
# visualization all share one source of truth.
TIME_DOMAIN_FEATURE_NAMES = [
    "rms",
    "std",
    "min",
    "max",
    "mean",
    "max_over_mean",
    "min_over_max",
    "mav",
    "iemg",
    "variance",
    "wl",
]

FREQUENCY_DOMAIN_FEATURE_NAMES = [
    "total_power",
    "peak_frequency",
    "mean_frequency",
    "median_frequency",
]


def _time_domain_feature(features: List[str]) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures",
    }
    if features:
        spec["features"] = features
    return spec


def _frequency_domain_feature(
    sampling_rate: float, features: List[str]
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.frequency_domain.FrequencyDomainFeatures",
        "sampling_rate": sampling_rate,
        "use_hann": True,
        "remove_dc": True,
    }
    if features:
        spec["features"] = features
    return spec


@dataclass
class FeatureExtractionConfig:
    feature_type: str = "time_domain"  # time_domain | frequency_domain | combined
    # Feature names to compute (empty = all features of the selected type(s)).
    features: List[str] = field(default_factory=list)

    def resolve_features(self, sampling_rate: float) -> List[Dict[str, Any]]:
        """Return the concrete feature-extractor specs for this config.

        A single ``feature_type`` knob selects the representation; ``features``
        (when non-empty) restricts which named features each extractor computes.
        The sampling rate is injected from the dataset so the frequency
        extractor never needs it hardcoded in YAML.
        """
        if self.feature_type == "time_domain":
            return [_time_domain_feature(self.features)]
        if self.feature_type == "frequency_domain":
            return [_frequency_domain_feature(sampling_rate, self.features)]
        if self.feature_type == "combined":
            return [
                _time_domain_feature(self.features),
                _frequency_domain_feature(sampling_rate, self.features),
            ]
        raise ConfigValidationError(
            f"feature_extraction.feature_type={self.feature_type!r} not in {_FEATURE_TYPES}"
        )

    def validate(self) -> None:
        if self.feature_type not in _FEATURE_TYPES:
            raise ConfigValidationError(
                f"feature_extraction.feature_type={self.feature_type!r} not in {_FEATURE_TYPES}"
            )
        if self.feature_type == "time_domain":
            allowed = set(TIME_DOMAIN_FEATURE_NAMES)
        elif self.feature_type == "frequency_domain":
            allowed = set(FREQUENCY_DOMAIN_FEATURE_NAMES)
        else:  # combined
            allowed = set(TIME_DOMAIN_FEATURE_NAMES) | set(
                FREQUENCY_DOMAIN_FEATURE_NAMES
            )
        unknown = [f for f in self.features if f not in allowed]
        if unknown:
            raise ConfigValidationError(
                f"feature_extraction.features={unknown!r} contains feature name(s) "
                f"not valid for feature_type={self.feature_type!r}; "
                f"valid names: {sorted(allowed)}"
            )
