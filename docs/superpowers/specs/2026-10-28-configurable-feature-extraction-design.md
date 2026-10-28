# Configurable feature extraction with new time-domain features

Date: 2026-10-28
Status: approved (pending implementation)

## Context

The feature-extraction pipeline currently selects features only at the
*extractor* level via `feature_extraction.feature_type`
(`time_domain | frequency_domain | combined | custom`). Each extractor always
computes **all** of its features; there is no way to ask for a subset such as
"only RMS and waveform length".

`TimeDomainFeatures` today computes 7 features: `rms`, `std`, `min`, `max`,
`mean`, `max_over_mean`, `min_over_max`. `FrequencyDomainFeatures` computes 4:
`total_power`, `peak_frequency`, `mean_frequency`, `median_frequency`.

## Goals

1. Add four time-domain features: `mav`, `iemg`, `variance`, `wl`.
2. Allow per-feature name selection for both time-domain and frequency-domain
   features, e.g. `features: [rms, mav, wl, total_power]`.
3. Extractors must compute **only** the selected features (no compute-all-then-filter).

## Non-goals

- The `custom` mode is removed. It was unused (no config preset referenced it;
  only a validation test did) and its `_target_` list conflicted with
  reusing `features` as a name list.
- No change to the visualization path beyond the fact that new feature names
  automatically appear there via `FEATURE_NAMES`.

## Design

### 1. New time-domain features

`TimeDomainFeatures.FEATURE_NAMES` becomes an 11-element ordered list:

```
rms, std, min, max, mean, max_over_mean, min_over_max, mav, iemg, variance, wl
```

New feature definitions (each computed per window per channel):

| name | formula |
|---|---|
| `mav` | `1/N Σ |xᵢ|` |
| `iemg` | `Σ |xᵢ|` |
| `variance` | `1/N Σ (xᵢ − x̄)²` |
| `wl` | `Σᵢ₌₁ᴺ⁻¹ |xᵢ₊₁ − xᵢ|` |

`variance` coexists with the existing `std` (its square root).

### 2. Configuration schema

`FeatureExtractionConfig` (in
`configuration/schema/feature_extraction.py`) changes to:

```python
_FEATURE_TYPES = ("time_domain", "frequency_domain", "combined")

@dataclass
class FeatureExtractionConfig:
    feature_type: str = "time_domain"
    features: List[str] = field(default_factory=list)  # names; [] = all
```

- `feature_type` loses the `custom` value.
- `features` is a list of feature **names**; empty means "all features of the
  selected type(s)".

`resolve_features(sampling_rate)` injects `self.features` into each extractor
spec it builds. When `self.features` is empty, the extractor spec passes
`features=None` (i.e. "all"), never an empty list, so `[]` and "all" are the
same thing:

```python
if self.feature_type == "time_domain":
    return [_time_domain_feature(self.features)]
if self.feature_type == "frequency_domain":
    return [_frequency_domain_feature(sampling_rate, self.features)]
if self.feature_type == "combined":
    return [_time_domain_feature(self.features),
            _frequency_domain_feature(sampling_rate, self.features)]
```

### 3. Extractor subset support

`TimeDomainFeatures(features=None)` and
`FrequencyDomainFeatures(sampling_rate, use_hann=True, remove_dc=True, features=None)`
gain an optional `features` parameter:

- `None` (default) → compute all features (backward compatible).
- `[names]` → compute only the named features, in the given order.

Implementation: a module-level `name -> compute function` dispatch map. For
time-domain each function maps `X` (n_windows, time, channels) to a
`(n_windows, channels)` column block; `transform` concatenates only the selected
blocks. The ratio features (`max_over_mean`, `min_over_max`) compute their own
`min`/`max`/`mean` intermediates internally.

For frequency-domain the FFT + power spectrum is always computed (it is the
transform); only the selected summary features are computed from it.
`mean_frequency` and `median_frequency` compute `total_power` internally as a
cheap intermediate because they depend on it.

### 4. Validation

`FeatureExtractionConfig.validate()`:

- `feature_type` must be in `("time_domain", "frequency_domain", "combined")`.
- Every entry in `features` must be a known feature name for the selected
  domain(s): time-domain names for `time_domain`, frequency-domain names for
  `frequency_domain`, either for `combined`. Unknown names raise
  `ConfigValidationError`.

The feature name catalogs are exposed as constants (the extractors'
`FEATURE_NAMES`) so validation stays in sync with the extractors.

### 5. Data flow (unchanged)

Both backends already call `cfg.feature_extraction.resolve_features(...)` and
pass the resulting specs to `FeatureExtractor`, which `instantiate`s each
`_target_` spec. Adding `features` to the specs therefore flows automatically
through the classical and neural (flat + two-level) paths with no further
changes. The visualization's `_feature_matrix` reads `FEATURE_NAMES`, so the new
features appear there automatically.

## Backward compatibility

- Configs that omit `features` (or set `features: []`) behave exactly as today.
- `feature_type` presets are unchanged except that `custom` is removed.

## Testing

- Update `test_feature_name_constants` (7 → 11 time-domain names).
- Update `test_feature_extractor` shape assertions (7 → 11 time-domain columns).
- Replace `test_feature_extraction_validate_rejects_custom_without_features`
  with validation tests for: unknown feature name, domain mismatch, and the
  removed `custom` value.
- Add tests for: each new feature's formula (hand-computed on a small input),
  and subset selection for both time-domain and frequency-domain
  (column count + column ordering).
- Update `test_config_loader` feature-resolution tests to cover `features`
  injection and `combined` with a mixed-domain name list.

## Documentation

- Update `docs/experiments/configuration.md` feature list and the
  `feature_extraction` group description.
