# v128 — benchmark feature search

<!-- Registry entry: research/registry.yaml (id: v128). -->

**Question.** Benchmark-specific full feature search for classification models.

**Date.** 2026-04-12 (first commit).

**Status.** `shadow`, feeds `config.features.V128_BENCHMARK_FEATURE_MAP_PATH`.

**Notes.** The monthly run reads v128_benchmark_feature_map.csv for the benchmark-specific shadow classifier. Re-run pending (WP11).

**Closeout / record.** [2026-04-12-v123-v128-classification-enhancement-plan.md](../../../docs/superpowers/plans/2026-04-12-v123-v128-classification-enhancement-plan.md)

## Run

From the repository root:

```bash
python research/studies/v128_benchmark_feature_search/v128_benchmark_feature_search.py
```

## Outputs

`outputs/` (8 files, committed): `v128_baseline_metrics.csv`, `v128_benchmark_feature_map.csv`, `v128_benchmark_feature_search_comparison.csv`, `v128_benchmark_feature_search_summary.md`, `v128_feature_inventory.csv`, `v128_forward_stepwise_trace.csv`, `v128_regularized_comparison.csv`, `v128_single_feature_results.csv`

### Detail file (not committed)

`v128_regularized_selection_detail.csv` is over 1 MB, so it is written to `outputs/detail/`, which is
gitignored. To regenerate it, run:

```bash
python research/studies/v128_benchmark_feature_search/v128_benchmark_feature_search.py
```

The last committed copy is in history as `results/research/v128_regularized_selection_detail.csv`:

```bash
git show ae61167:results/research/v128_regularized_selection_detail.csv > v128_regularized_selection_detail.csv
```

## Tests

- [`tests/research/test_research_v128_benchmark_feature_search.py`](../../../tests/research/test_research_v128_benchmark_feature_search.py)
