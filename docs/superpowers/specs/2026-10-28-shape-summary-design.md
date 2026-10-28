# Shape summary at each pipeline stage

Date: 2026-10-28
Status: approved (pending implementation)

## Context

The pipeline logs stage *timing* (`[stage] windowing started/finished in Xs`)
but never reports tensor shapes, which makes it hard to verify the data flow
(raw signal → windows → features). A shape summary should report the shape at
each stage using clear symbolic dimension names.

## Goals

1. Emit a shape summary (console log + a saved `shape_summary.json` artifact,
   logged to MLflow) at each pipeline stage.
2. Report shapes using symbolic dimension names (not raw integer tuples), so a
   reader can tell which dimension is which.
3. Cover both windowing modes (`single` and `two_level`).
4. Report the shape for a **representative trial** (the first non-rest trial).

## Non-goals

- No change to the training data flow or feature computation.
- The summary is informational; it does not reshape or transform any tensor.

## Design

### Stage shapes (channels-first, symbolic names)

Two-level mode (four stages):

| Stage | Dimensions |
|---|---|
| `input` | `channels × datapoints_per_channel` |
| `big_windows` | `channels × big_windows × big_window_size` |
| `small_windows` | `channels × big_windows × small_windows × small_window_size` |
| `features` | `channels × big_windows × small_windows × features_per_small_window` |

Single-level mode (three stages; `big_windows`/`small_windows` collapse to one
`windows` stage):

| Stage | Dimensions |
|---|---|
| `input` | `channels × datapoints_per_channel` |
| `windows` | `channels × windows × window_size` |
| `features` | `channels × windows × features_per_window` |

`features_per_small_window` / `features_per_window` is the number of
handcrafted features per channel per window (11 time-domain, 4 frequency, 15
combined), derived from the resolved feature specs. A subset selection
(e.g. `features: [rms, mav]`) yields 2.

### Output example (two-level)

```json
{
  "mode": "two_level",
  "input":         {"channels": 8, "datapoints_per_channel": 994},
  "big_windows":   {"channels": 8, "big_windows": 15, "big_window_size": 250},
  "small_windows": {"channels": 8, "big_windows": 15, "small_windows": 23, "small_window_size": 25},
  "features":      {"channels": 8, "big_windows": 15, "small_windows": 23, "features_per_small_window": 11}
}
```

### Implementation

A new function `compute_shape_summary(dataset, windowing, feature_specs) ->
dict` in `utils/observability/shapes.py`:

1. Pick the first non-rest trial (`next(t for t in dataset.trials if not t.is_rest)`).
2. `channels = trial.emg.shape[1]`, `datapoints_per_channel = trial.emg.shape[0]`.
3. Build a single-trial dataset (`dataset.replace_trials([trial])`) and window
   it with the configured strategy (`recognition` for single, `two_level` for
   two-level), reusing the existing windowing strategies (no new windowing
   logic).
4. Derive the window counts/sizes from the windowed result and the feature
   count per channel from `feature_specs` (count the resolved names per
   extractor: `TimeDomainFeatures` → `len(TIME_DOMAIN_FEATURE_NAMES)` or the
   spec's `features` subset; likewise `FrequencyDomainFeatures`).

The pipeline calls it once per run (not gated by `visualization.enabled`), logs
`shape_summary: {...}` via the run logger, writes `shape_summary.json` into
`output/evaluation/<run_id>/`, and logs it to MLflow via the tracker.

## Backward compatibility

- Additive only; no existing behavior changes.

## Testing

- Unit tests for `compute_shape_summary`: single-level and two-level cases,
  correct symbolic keys and counts, feature-subset reflecting a reduced
  `features_per_small_window`.
- Integration: a run produces `output/evaluation/<run_id>/shape_summary.json`
  with the expected keys.

## Documentation

- Note the new artifact in `docs/experiments/configuration.md` (the
  observability/reporting section), one line.
