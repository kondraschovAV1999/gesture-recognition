# Configurable Feature Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four time-domain features (`mav`, `iemg`, `variance`, `wl`) and allow per-feature name selection for both time-domain and frequency-domain features.

**Architecture:** The canonical, ordered feature-name catalogs live in the schema leaf (`configuration/schema/feature_extraction.py`), and the extractors import them as their `FEATURE_NAMES`. Each extractor gains an optional `features` parameter and computes only the selected features via a `name -> compute function` dispatch map. `FeatureExtractionConfig.features` becomes a `List[str]` (empty = all), `custom` mode is removed, and `resolve_features()` injects the subset into the extractor specs.

**Tech Stack:** Python 3.10, numpy, scikit-learn (`BaseEstimator`, `TransformerMixin`), hydra (`instantiate`), pytest.

## Global Constraints

- `configuration/schema/` is a **leaf** of the dependency graph: its submodules may import only stdlib, `numpy`, `hand_gesture_recognition.configuration.errors`, and sibling schema modules — **never** from `pipeline/`, `utils/`, or `runner/`. This is why the name catalogs live in the schema and the extractors import *from* the schema.
- `feature_type` values after this change: `time_domain | frequency_domain | combined` (no `custom`).
- `features: []` (or omitted) means "all features of the selected type(s)".
- Extractors must compute **only** the selected features (no compute-all-then-filter).
- New time-domain column order (feature-major, matching `FEATURE_NAMES`): `rms, std, min, max, mean, max_over_mean, min_over_max, mav, iemg, variance, wl`.
- Configs that omit `features` behave exactly as today (all features).

---

### Task 1: Schema — name catalogs, `features: List[str]`, remove `custom`

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema/feature_extraction.py`
- Modify: `src/hand_gesture_recognition/configuration/schema/__init__.py:32-34` and `:82-84` (`__all__`)
- Test: `tests/unit/test_config_schema.py:184-198`
- Test: `tests/unit/test_config_loader.py:61-96`

**Interfaces:**
- Produces: `TIME_DOMAIN_FEATURE_NAMES` and `FREQUENCY_DOMAIN_FEATURE_NAMES` (module-level `List[str]`), re-exported from `hand_gesture_recognition.configuration.schema`.
- Produces: `FeatureExtractionConfig(feature_type: str = "time_domain", features: List[str] = field(default_factory=list))` with `resolve_features(sampling_rate)` and `validate()`.
- Consumes: `ConfigValidationError` from `hand_gesture_recognition.configuration.errors`.

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_config_schema.py` (replace the existing `test_feature_extraction_validate_rejects_custom_without_features` at lines 192-197):

```python
def test_feature_extraction_validate_rejects_custom():
    from hand_gesture_recognition.configuration.schema import FeatureExtractionConfig

    f = FeatureExtractionConfig(feature_type="custom")
    with pytest.raises(ConfigValidationError):
        f.validate()


def test_feature_extraction_validate_rejects_unknown_feature():
    from hand_gesture_recognition.configuration.schema import FeatureExtractionConfig

    f = FeatureExtractionConfig(feature_type="time_domain", features=["rms", "bogus"])
    with pytest.raises(ConfigValidationError):
        f.validate()


def test_feature_extraction_validate_rejects_cross_domain_feature():
    from hand_gesture_recognition.configuration.schema import FeatureExtractionConfig

    # total_power is a frequency-domain feature, not time-domain.
    f = FeatureExtractionConfig(feature_type="time_domain", features=["total_power"])
    with pytest.raises(ConfigValidationError):
        f.validate()


def test_feature_extraction_validate_accepts_combined_names():
    from hand_gesture_recognition.configuration.schema import FeatureExtractionConfig

    f = FeatureExtractionConfig(
        feature_type="combined", features=["rms", "total_power"]
    )
    f.validate()  # must not raise
```

Also update the existing `test_feature_extraction_config_defaults` (lines 184-189) to additionally assert the name catalogs:

```python
def test_feature_extraction_config_defaults():
    from hand_gesture_recognition.configuration.schema import (
        FeatureExtractionConfig,
        FREQUENCY_DOMAIN_FEATURE_NAMES,
        TIME_DOMAIN_FEATURE_NAMES,
    )

    f = FeatureExtractionConfig()
    assert f.feature_type == "time_domain"
    assert f.features == []
    assert TIME_DOMAIN_FEATURE_NAMES == [
        "rms", "std", "min", "max", "mean",
        "max_over_mean", "min_over_max",
        "mav", "iemg", "variance", "wl",
    ]
    assert FREQUENCY_DOMAIN_FEATURE_NAMES == [
        "total_power", "peak_frequency", "mean_frequency", "median_frequency",
    ]
```

Add to `tests/unit/test_config_loader.py` (after `test_feature_extraction_defaults_to_time_domain` at line 96):

```python
def test_feature_extraction_injects_selected_features():
    cfg = load_config(
        mode="compose",
        overrides=[
            "feature_extraction.feature_type=time_domain",
            "feature_extraction.features=[rms,mav]",
            "backend=classical",
        ],
    )
    specs = cfg.feature_extraction.resolve_features(cfg.dataset.sampling_rate)
    assert specs[0]["features"] == ["rms", "mav"]


def test_feature_extraction_empty_features_omit_key():
    cfg = load_config(mode="compose", overrides=["backend=classical"])
    specs = cfg.feature_extraction.resolve_features(cfg.dataset.sampling_rate)
    assert "features" not in specs[0]  # empty list -> all features


def test_feature_extraction_rejects_unknown_feature_name():
    with pytest.raises(ConfigValidationError):
        load_config(
            mode="compose",
            overrides=[
                "feature_extraction.features=[nope]",
                "backend=classical",
            ],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -q`
Expected: the new tests FAIL (name constants / validation not implemented; `custom` still accepted), and `test_feature_extraction_config_defaults` FAILS on the new catalog assertions.

- [ ] **Step 3: Rewrite the schema module**

Replace the whole contents of `src/hand_gesture_recognition/configuration/schema/feature_extraction.py`:

```python
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
            allowed = set(TIME_DOMAIN_FEATURE_NAMES) | set(FREQUENCY_DOMAIN_FEATURE_NAMES)
        unknown = [f for f in self.features if f not in allowed]
        if unknown:
            raise ConfigValidationError(
                f"feature_extraction.features={unknown!r} contains feature name(s) "
                f"not valid for feature_type={self.feature_type!r}; "
                f"valid names: {sorted(allowed)}"
            )
```

- [ ] **Step 4: Update the schema package exports**

In `src/hand_gesture_recognition/configuration/schema/__init__.py`, change the import block (lines 32-34):

```python
from hand_gesture_recognition.configuration.schema.feature_extraction import (
    FREQUENCY_DOMAIN_FEATURE_NAMES,
    FeatureExtractionConfig,
    TIME_DOMAIN_FEATURE_NAMES,
)
```

and add `"FREQUENCY_DOMAIN_FEATURE_NAMES"` and `"TIME_DOMAIN_FEATURE_NAMES"` to `__all__` (alphabetically near `"FeatureExtractionConfig"`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -q`
Expected: PASS (note: the other pre-existing failures in `test_config_loader.py` unrelated to features — `test_compose_mode_rejects_inconsistent_windowing`, `test_gesture_crop_config_default_enabled`, `test_compose_defaults_to_recognition_task` — may still fail due to the working-tree `config.yaml` defaults; that's expected and out of scope).

- [ ] **Step 6: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema/feature_extraction.py src/hand_gesture_recognition/configuration/schema/__init__.py tests/unit/test_config_schema.py tests/unit/test_config_loader.py
git commit -m "feat-feature-extraction-name-catalogs-and-selection-config"
```

---

### Task 2: `TimeDomainFeatures` — 4 new features + subset support

**Files:**
- Modify: `src/hand_gesture_recognition/pipeline/feature_extraction/extraction_strategies/time_domain.py`
- Test: `tests/unit/test_classical_pipeline.py:43-72`
- Test: `tests/unit/test_feature_extractor.py:38-39,58-62,87`
- Test: `tests/unit/test_visualization.py:81-96`
- Test: `tests/unit/test_grouped_datamodule.py:54`

**Interfaces:**
- Consumes: `TIME_DOMAIN_FEATURE_NAMES` from `hand_gesture_recognition.configuration.schema.feature_extraction`.
- Produces: `TimeDomainFeatures(features: Optional[List[str]] = None)` with `FEATURE_NAMES` class attribute equal to `TIME_DOMAIN_FEATURE_NAMES`; `transform(X) -> (n_windows, n_selected * n_channels)`.

- [ ] **Step 1: Write failing tests**

In `tests/unit/test_classical_pipeline.py`, replace `test_time_domain_features_per_channel` (lines 43-55) with an 11-feature version and add a subset test:

```python
def test_time_domain_features_per_channel():
    # 1 window, 3 time steps, 1 channel: values [0, 2, 4]
    X = np.array([[[0.0], [2.0], [4.0]]], dtype=np.float32)  # (1, 3, 1)
    out = TimeDomainFeatures().transform(X)
    assert out.shape == (1, 11)  # 11 features x 1 channel
    (
        rms, std, mn, mx, mean, mx_mean, mn_mx,
        mav, iemg, variance, wl,
    ) = out[0]
    assert rms == pytest.approx(np.sqrt(20 / 3))
    assert std == pytest.approx(np.std([0.0, 2.0, 4.0]))
    assert mn == 0.0
    assert mx == 4.0
    assert mean == 2.0
    assert mx_mean == pytest.approx(2.0)  # max / mean = 4 / 2
    assert mn_mx == 0.0  # min / max = 0 / 4
    assert mav == pytest.approx(2.0)  # (0 + 2 + 4) / 3
    assert iemg == pytest.approx(6.0)  # 0 + 2 + 4
    assert variance == pytest.approx(np.var([0.0, 2.0, 4.0]))  # 8 / 3
    assert wl == pytest.approx(4.0)  # |2-0| + |4-2|


def test_time_domain_features_subset():
    X = np.array([[[0.0], [2.0], [4.0]]], dtype=np.float32)  # (1, 3, 1)
    out = TimeDomainFeatures(features=["wl", "mav"]).transform(X)
    assert out.shape == (1, 2)  # 2 selected features x 1 channel
    wl, mav = out[0]
    assert wl == pytest.approx(4.0)
    assert mav == pytest.approx(2.0)
```

In `tests/unit/test_classical_pipeline.py`, update `test_time_domain_features_8_channels` (line 61) and `test_time_and_frequency_features_concatenate` (line 71):

```python
def test_time_domain_features_8_channels():
    X = np.zeros((5, 40, 8), dtype=np.float32)  # 5 windows, 40 time, 8 channels
    out = TimeDomainFeatures().transform(X)
    assert out.shape == (5, 88)  # 11 features x 8 channels
    assert np.isfinite(out).all()


def test_time_and_frequency_features_concatenate():
    X = np.random.default_rng(0).normal(0, 1, size=(5, 40, 8)).astype(np.float32)
    time_feat = TimeDomainFeatures().fit_transform(X)
    freq_feat = FrequencyDomainFeatures(sampling_rate=200.0).fit_transform(X)
    combined = np.concatenate([time_feat, freq_feat], axis=1)
    assert combined.shape == (5, 11 * 8 + 4 * 8)  # union of both extractors
    assert np.isfinite(combined).all()
```

In `tests/unit/test_feature_extractor.py`, change every `7 *` to `11 *` (lines 38, 39, 58, 61, 62, 87):

```python
    assert fe.fit_transform(X).shape == (n_windows, 11 * n_channels)
    assert fe.transform(X).shape == (n_windows, 11 * n_channels)
    # ...
    expected_cols = 11 * n_channels + n_time * n_channels
    assert np.allclose(out[:, : 11 * n_channels], TimeDomainFeatures().transform(X))
    assert np.allclose(out[:, 11 * n_channels :], (X * 2.0).reshape(n_windows, -1))
    # ...
        assert o.shape == (w.shape[0], 11 * 4)
```

In `tests/unit/test_visualization.py`, update `test_feature_name_constants` (lines 82-90):

```python
    assert TimeDomainFeatures.FEATURE_NAMES == [
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
```

In `tests/unit/test_grouped_datamodule.py`, change line 54:

```python
    n_features = 11 * 2  # 11 time-domain features x 2 channels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_classical_pipeline.py tests/unit/test_feature_extractor.py tests/unit/test_visualization.py tests/unit/test_grouped_datamodule.py -q`
Expected: FAIL — `TimeDomainFeatures` still returns 7 columns and lacks `mav`/`iemg`/`variance`/`wl`.

- [ ] **Step 3: Rewrite `TimeDomainFeatures`**

Replace the whole contents of `src/hand_gesture_recognition/pipeline/feature_extraction/extraction_strategies/time_domain.py`:

```python
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from hand_gesture_recognition.configuration.schema.feature_extraction import (
    TIME_DOMAIN_FEATURE_NAMES,
)


def _rms(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(X**2, axis=1))


def _std(X: np.ndarray) -> np.ndarray:
    return np.std(X, axis=1)


def _min(X: np.ndarray) -> np.ndarray:
    return np.min(X, axis=1)


def _max(X: np.ndarray) -> np.ndarray:
    return np.max(X, axis=1)


def _mean(X: np.ndarray) -> np.ndarray:
    return np.mean(X, axis=1)


def _mav(X: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(X), axis=1)


def _iemg(X: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(X), axis=1)


def _variance(X: np.ndarray) -> np.ndarray:
    return np.var(X, axis=1)


def _wl(X: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(X, axis=1)), axis=1)


def _max_over_mean(X: np.ndarray) -> np.ndarray:
    mx = np.max(X, axis=1)
    mean = np.mean(X, axis=1)
    return np.divide(
        mx, mean, out=np.zeros_like(mx, dtype=np.float32), where=mean != 0
    )


def _min_over_max(X: np.ndarray) -> np.ndarray:
    mn = np.min(X, axis=1)
    mx = np.max(X, axis=1)
    return np.divide(
        mn, mx, out=np.zeros_like(mn, dtype=np.float32), where=mx != 0
    )


_COMPUTE = {
    "rms": _rms,
    "std": _std,
    "min": _min,
    "max": _max,
    "mean": _mean,
    "max_over_mean": _max_over_mean,
    "min_over_max": _min_over_max,
    "mav": _mav,
    "iemg": _iemg,
    "variance": _variance,
    "wl": _wl,
}


class TimeDomainFeatures(BaseEstimator, TransformerMixin):
    """Per-channel handcrafted time-domain features.

    Computes the features named in ``FEATURE_NAMES`` (or the subset passed via
    ``features``). Each feature is computed per window and per channel, then the
    blocks are concatenated column-wise in feature-major order.
    Output shape: (n_windows, n_selected * n_channels).
    """

    FEATURE_NAMES = TIME_DOMAIN_FEATURE_NAMES

    def __init__(self, features: Optional[List[str]] = None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (n_windows, time, channels)
        X = np.asarray(X, dtype=np.float32)
        names = self.features if self.features is not None else self.FEATURE_NAMES
        return np.concatenate([_COMPUTE[n](X) for n in names], axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_classical_pipeline.py tests/unit/test_feature_extractor.py tests/unit/test_visualization.py tests/unit/test_grouped_datamodule.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/pipeline/feature_extraction/extraction_strategies/time_domain.py tests/unit/test_classical_pipeline.py tests/unit/test_feature_extractor.py tests/unit/test_visualization.py tests/unit/test_grouped_datamodule.py
git commit -m "feat-add-mav-iemg-variance-wl-and-time-domain-subset"
```

---

### Task 3: `FrequencyDomainFeatures` — subset support

**Files:**
- Modify: `src/hand_gesture_recognition/pipeline/feature_extraction/extraction_strategies/frequency_domain.py`
- Test: `tests/unit/test_frequency_domain.py`

**Interfaces:**
- Consumes: `FREQUENCY_DOMAIN_FEATURE_NAMES` from `hand_gesture_recognition.configuration.schema.feature_extraction`.
- Produces: `FrequencyDomainFeatures(sampling_rate, use_hann=True, remove_dc=True, features=None)` with `FEATURE_NAMES` class attribute equal to `FREQUENCY_DOMAIN_FEATURE_NAMES`.

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_frequency_domain.py`:

```python
def test_subset_selection():
    fs = 200.0
    n = 100
    t = np.arange(n) / fs
    tone = np.sin(2 * np.pi * 20.0 * t).astype(np.float32)
    X = tone[None, :, None]  # (1, n, 1)

    full = FrequencyDomainFeatures(sampling_rate=fs).transform(X)
    sub = FrequencyDomainFeatures(sampling_rate=fs, features=["total_power"]).transform(X)
    assert sub.shape == (1, 1)  # 1 selected feature x 1 channel
    assert np.allclose(sub[:, 0], full[:, 0])  # total_power block matches full


def test_subset_selection_order():
    X = np.random.default_rng(0).normal(0, 1, size=(3, 40, 2)).astype(np.float32)
    out = FrequencyDomainFeatures(
        sampling_rate=200.0, features=["median_frequency", "peak_frequency"]
    ).transform(X)
    assert out.shape == (3, 4)  # 2 selected features x 2 channels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_frequency_domain.py -q`
Expected: FAIL — `FrequencyDomainFeatures.__init__` has no `features` parameter.

- [ ] **Step 3: Rewrite `FrequencyDomainFeatures`**

Replace the whole contents of `src/hand_gesture_recognition/pipeline/feature_extraction/extraction_strategies/frequency_domain.py`:

```python
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from hand_gesture_recognition.configuration.schema.feature_extraction import (
    FREQUENCY_DOMAIN_FEATURE_NAMES,
)


def _total_power(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    return np.sum(power, axis=1)


def _peak_frequency(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    peak_idx = np.argmax(power, axis=1)
    return frequencies[peak_idx]


def _mean_frequency(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    total = np.sum(power, axis=1)
    weighted = np.sum(power * frequencies[None, :, None], axis=1)
    return np.divide(
        weighted, total, out=np.zeros_like(total), where=total > 0
    )


def _median_frequency(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    total = np.sum(power, axis=1)
    cumulative = np.cumsum(power, axis=1)
    half_power = total[:, None, :] / 2.0
    median_idx = np.argmax(cumulative >= half_power, axis=1)
    return frequencies[median_idx]


_COMPUTE = {
    "total_power": _total_power,
    "peak_frequency": _peak_frequency,
    "mean_frequency": _mean_frequency,
    "median_frequency": _median_frequency,
}


class FrequencyDomainFeatures(BaseEstimator, TransformerMixin):
    """Per-channel frequency-domain features using the real FFT.

    Input
    -----
    X : np.ndarray
        Shape: (n_windows, n_samples, n_channels)

    Output
    ------
    np.ndarray
        Shape: (n_windows, n_selected * n_channels)

        For each selected channel-feature pair: total power, peak frequency,
        mean frequency, and/or median frequency.

    Parameters
    ----------
    sampling_rate : float
        Sampling frequency in Hz.

    use_hann : bool, default=True
        Apply a Hann window before the FFT to reduce spectral leakage.

    remove_dc : bool, default=True
        Remove the mean of each window before computing the FFT.

    features : list of str, optional
        Subset of ``FEATURE_NAMES`` to compute. ``None`` (default) computes all.
        ``mean_frequency`` and ``median_frequency`` compute ``total_power``
        internally because they depend on it, but it is only emitted when
        requested.
    """

    FEATURE_NAMES = FREQUENCY_DOMAIN_FEATURE_NAMES

    def __init__(
        self,
        sampling_rate: float,
        use_hann: bool = True,
        remove_dc: bool = True,
        features: Optional[List[str]] = None,
    ):
        self.sampling_rate = sampling_rate
        self.use_hann = use_hann
        self.remove_dc = remove_dc
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract frequency-domain features from windowed sEMG."""
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                "Expected X with shape "
                "(n_windows, n_samples, n_channels), "
                f"got shape {X.shape}"
            )

        n_windows, n_samples, n_channels = X.shape

        if n_samples < 2:
            raise ValueError("Each window must contain at least 2 samples.")

        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be greater than 0.")

        # 1. Remove DC component
        if self.remove_dc:
            X = X - np.mean(X, axis=1, keepdims=True)

        # 2. Apply Hann window for spectral analysis
        if self.use_hann:
            window = np.hanning(n_samples).astype(np.float32)
            X_fft = X * window[None, :, None]
        else:
            X_fft = X

        # 3. Real FFT along the time axis
        fft = np.fft.rfft(X_fft, axis=1)
        frequencies = np.fft.rfftfreq(n_samples, d=1.0 / self.sampling_rate)

        # 4. Power spectrum (drop the DC bin)
        power = np.abs(fft) ** 2
        power = power[:, 1:, :]
        frequencies = frequencies[1:]

        names = self.features if self.features is not None else self.FEATURE_NAMES
        return np.concatenate(
            [_COMPUTE[n](power, frequencies) for n in names], axis=1
        ).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_frequency_domain.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/pipeline/feature_extraction/extraction_strategies/frequency_domain.py tests/unit/test_frequency_domain.py
git commit -m "feat-frequency-domain-subset-selection"
```

---

### Task 4: Documentation

**Files:**
- Modify: `docs/experiments/configuration.md:204-218,347-348`
- Modify: `docs/development/extending.md:131-157`
- Modify: `docs/experiments/running.md:40`

- [ ] **Step 1: Update `docs/experiments/configuration.md`**

Replace the `### feature_extraction` section (lines 204-218) with:

```markdown
### `feature_extraction`

| Field | Type | Meaning |
|---|---|---|
| `feature_type` | `time_domain` \| `frequency_domain` \| `combined` | Which extractor(s) to run |
| `features` | list of **names** (strings) | Which features to compute; `[]` = all |

`feature_type` picks the handcrafted feature representation:

| `feature_type` | extractors |
|---|---|
| `time_domain` | `TimeDomainFeatures` (11 per-channel features) |
| `frequency_domain` | `FrequencyDomainFeatures` (4 per-channel features) |
| `combined` | both, concatenated column-wise |

`features` is a list of feature names (not channel numbers):

```yaml
feature_extraction:
  feature_type: time_domain
  features: [rms, mav, wl]     # compute only these three
```

Feature names:

- Time-domain: `rms`, `std`, `min`, `max`, `mean`, `max_over_mean`,
  `min_over_max`, `mav`, `iemg`, `variance`, `wl`
- Frequency-domain: `total_power`, `peak_frequency`, `mean_frequency`,
  `median_frequency`
```

Also update the visualization feature list at lines 347-348 to add the four new time-domain names:

```markdown
- Time-domain: `rms`, `std`, `min`, `max`, `mean`, `max_over_mean`,
  `min_over_max`, `mav`, `iemg`, `variance`, `wl`
- Frequency-domain: `total_power`, `peak_frequency`, `mean_frequency`,
  `median_frequency`
```

- [ ] **Step 2: Update `docs/development/extending.md`**

Replace the `feature_type` table (lines 131-139) and the `custom`-referencing prose (line 151) so it no longer mentions `custom`, and note the `features` name list. Replace lines 131-157 with:

```markdown
The handcrafted feature representation is chosen with a single `feature_type`
knob (select from the CLI with `feature_extraction.feature_type=<name>`):

| `feature_type` | feature extractors applied |
|---|---|
| `time_domain` | `TimeDomainFeatures` |
| `frequency_domain` | `FrequencyDomainFeatures` |
| `combined` | both, concatenated column-wise |

A subset of features can be requested by name with
`feature_extraction.features=[rms, wl, ...]` (empty = all). The canonical
feature-name catalogs live in
`configuration/schema/feature_extraction.py`
(`TIME_DOMAIN_FEATURE_NAMES` / `FREQUENCY_DOMAIN_FEATURE_NAMES`) and are
imported by the extractors as their `FEATURE_NAMES`.

To add a new handcrafted feature, add its name to the appropriate catalog and
add a compute function to the extractor's `_COMPUTE` dispatch map (see
`pipeline/feature_extraction/extraction_strategies/time_domain.py`).
```

- [ ] **Step 3: Update `docs/experiments/running.md`**

Change line 40's `feature_type` row to drop `custom` and mention `features`:

```markdown
| `feature_extraction.feature_type=time_domain` / `frequency_domain` / `combined` | choose the handcrafted feature representation |
| `feature_extraction.features=[rms,wl,...]` | restrict which named features are computed (empty = all) |
```

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/configuration.md docs/development/extending.md docs/experiments/running.md
git commit -m "docs-configurable-feature-extraction"
```

---

## Self-Review

- **Spec coverage:** §1 new features → Task 2 (compute functions + updated tests); §2 schema → Task 1; §3 extractor subset → Tasks 2 & 3; §4 validation → Task 1; §5 data flow (no change needed) → implicitly covered by existing classical/neural strategy tests; testing → Tasks 1-3; documentation → Task 4. All covered.
- **Placeholder scan:** no TBD/TODO; every code step includes full code; every test step includes full test code.
- **Type consistency:** `TIME_DOMAIN_FEATURE_NAMES` / `FREQUENCY_DOMAIN_FEATURE_NAMES` defined in Task 1 and consumed as `FEATURE_NAMES` in Tasks 2 & 3 with identical spelling. `features: Optional[List[str]]` on both extractors matches `resolve_features` injection. `FeatureExtractionConfig.features: List[str]` used consistently.
