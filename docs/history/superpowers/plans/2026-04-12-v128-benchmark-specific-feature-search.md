# v128 Benchmark-Specific Full Feature Search

Date: 2026-04-12
Status: implemented on the active branch

## Why v128 exists

The classification stack was still using one shared 12-feature `lean_baseline`
subset for every benchmark-specific model, even though the project had reached
the point where:

1. the point-in-time feature matrix had expanded to a 72-feature non-target
   universe,
2. the benchmark suite had grown to 10 current classifier targets, and
3. the strongest remaining research question was whether benchmark-specific
   feature selection could improve covered balanced accuracy without increasing
   feature count.

v128 exists to answer that question in a controlled way while preserving the
current classifier architecture:

- target: `actionable_sell_3pct`
- model family: separate benchmark-specific balanced logistic
- validation: rolling `TimeSeriesSplit` with the repo-standard purge / embargo
- calibration: prequential logistic calibration

## What changed

### 1. Added a full benchmark-specific feature-search harness

File:
- `results/research/v128_benchmark_feature_search.py`

The new harness performs five phases:

1. reproduces the incumbent `lean_baseline` benchmark metrics,
2. screens every eligible single feature on every benchmark,
3. runs forward stepwise search from the best single feature,
4. builds `L1` and elastic-net consensus subsets from fold-level regularized
   selections,
5. evaluates a ridge full-pool control as a diagnostic only.

Important implementation details:

- candidate universe is the full 72-column non-target feature matrix returned by
  `get_feature_columns(feature_df)`
- a feature is benchmark-eligible only if it has at least 60 non-null
  observations on that benchmark's aligned history
- all production-eligible candidates remain capped at 12 features
- forward-stepwise additions are accepted only if:
  - covered balanced accuracy improves by at least `+0.005`
  - `ece_10` worsens by no more than `+0.01`
  - `brier_score` worsens by no more than `+0.005`
- benchmark winners are chosen against the incumbent baseline using the same
  calibration guardrails; ridge is explicitly excluded from recommendation

### 2. Added v128 research artifacts

Files written to `results/research/`:

- `v128_feature_inventory.csv`
- `v128_baseline_metrics.csv`
- `v128_single_feature_results.csv`
- `v128_forward_stepwise_trace.csv`
- `v128_regularized_selection_detail.csv`
- `v128_regularized_comparison.csv`
- `v128_benchmark_feature_search_comparison.csv`
- `v128_benchmark_feature_map.csv`
- `v128_benchmark_feature_search_summary.md`

These artifacts provide:

- benchmark-by-feature eligibility and sparsity inventory
- full single-feature leaderboard
- every considered forward-stepwise addition
- fold-level `L1` / elastic-net selection frequencies
- final per-benchmark comparison against the incumbent
- the final benchmark-specific feature map for any future shadow integration

### 3. Added dedicated v128 tests

File:
- `tests/test_research_v128_benchmark_feature_search.py`

Coverage includes:

- exclusion of target / leakage columns from the candidate universe
- deterministic and capped consensus subset construction
- forward-stepwise gate behavior
- fallback to incumbent baseline when guardrails fail
- a reduced end-to-end smoke run that produces a valid feature map across all 10
  benchmarks

## Implemented result

The final v128 benchmark feature map switched 4 of the 10 benchmarks away from
the incumbent shared baseline:

- `BND` -> `elastic_net_consensus` (12 features)
- `DBC` -> `elastic_net_consensus` (12 features)
- `VGT` -> `forward_stepwise` (`rate_adequacy_gap_yoy`, `severity_index_yoy`)
- `VIG` -> `elastic_net_consensus` (12 features)

The other 6 benchmarks stayed on `lean_baseline`.

Most important per-benchmark result:

- `VGT` improved covered balanced accuracy from `0.5789` to `0.9474` while also
  improving `ece_10` and `brier_score`

Pooled final-map result versus incumbent:

- covered balanced accuracy: `0.5000` -> `0.5016`
- `ece_10`: `0.0488` -> `0.0387`
- `brier_score`: `0.1813` -> `0.1819`
- coverage: `0.8700` -> `0.8891`

Interpretation:

- benchmark-specific subsetting clearly helps a minority of benchmarks,
  especially `VGT`
- the pooled aggregate improvement is modest and primarily reliability-driven
- ridge did not beat the feature-capped candidates, which supports keeping the
  "no feature-count increase" constraint in place for now

## Commands used for verification

Research:
- `python results/research/v128_benchmark_feature_search.py`

Tests:
- `python -m pytest tests/test_research_v128_benchmark_feature_search.py`
- `python -m pytest tests/test_research_v128_benchmark_feature_search.py tests/test_classification_shadow.py tests/test_classification_artifacts.py tests/test_research_v122_classifier_audit.py tests/test_classification_config.py`

## Handoff notes for Claude Code

If this branch is handed back to Claude Code later, the intended next work is:

1. Treat `v128_benchmark_feature_map.csv` as the source of truth for any future
   benchmark-specific shadow integration.
2. Keep the benchmark-specific map research-only until we explicitly decide to
   wire `classification_shadow` to resolve `feature_map[benchmark]`.
3. Give extra scrutiny to the `VGT` two-feature result before any production
   promotion; it is the strongest v128 win, but it should be monitored
   prospectively rather than promoted on retrospective evidence alone.
4. If a `v129` follow-on is opened, focus on:
   - optional shadow integration of the benchmark-specific feature map,
   - prospective monitoring of the 4 switched benchmarks,
   - selective new-feature research only after the benchmark-specific map is
     either adopted or rejected.
