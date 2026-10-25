from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from hand_gesture_recognition.configuration.errors import ConfigValidationError

_FEATURE_TYPES = ("time_domain", "frequency_domain", "combined", "custom")

_TIME_DOMAIN_FEATURE = {
    "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures",
}


def _frequency_domain_feature(sampling_rate: float) -> Dict[str, Any]:
    return {
        "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.frequency_domain.FrequencyDomainFeatures",
        "sampling_rate": sampling_rate,
        "use_hann": True,
        "remove_dc": True,
    }


@dataclass
class FeatureExtractionConfig:
    feature_type: str = (
        "time_domain"  # time_domain | frequency_domain | combined | custom
    )
    # Only used when feature_type == "custom" (an explicit list of _target_ entries).
    features: List[Dict[str, Any]] = field(default_factory=list)

    def resolve_features(self, sampling_rate: float) -> List[Dict[str, Any]]:
        """Return the concrete feature-extractor specs for this config.

        ``custom`` defers to the explicit ``features`` list; the other presets
        are built here so a single ``feature_type`` knob selects the
        representation. The sampling rate is injected from the dataset so the
        frequency extractor never needs it hardcoded in YAML.
        """
        if self.feature_type == "custom":
            return list(self.features)
        if self.feature_type == "time_domain":
            return [_TIME_DOMAIN_FEATURE]
        if self.feature_type == "frequency_domain":
            return [_frequency_domain_feature(sampling_rate)]
        if self.feature_type == "combined":
            return [_TIME_DOMAIN_FEATURE, _frequency_domain_feature(sampling_rate)]
        raise ConfigValidationError(
            f"feature_extraction.feature_type={self.feature_type!r} not in {_FEATURE_TYPES}"
        )

    def validate(self) -> None:
        if self.feature_type not in _FEATURE_TYPES:
            raise ConfigValidationError(
                f"feature_extraction.feature_type={self.feature_type!r} not in {_FEATURE_TYPES}"
            )
        if self.feature_type == "custom" and not self.features:
            raise ConfigValidationError(
                "feature_extraction.feature_type='custom' requires a non-empty "
                "feature_extraction.features list"
            )
