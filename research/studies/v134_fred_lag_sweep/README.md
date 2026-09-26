# v134 — fred lag sweep

<!-- Registry entry: research/registry.yaml (id: v134). -->

**Question.** FRED publication lag sweep against the production ensemble path.

**Date.** 2026-04-14 (first commit).

**Status.** `retained`, feeds `config.features.FRED_SERIES_LAGS`.

**Notes.** Run on the pre-lagged DB, so its lag 0 was really lag 1 (review F06). Re-run pending (WP11).

## Run

From the repository root:

```bash
python research/studies/v134_fred_lag_sweep/v134_fred_lag_sweep.py
```

## Outputs

`outputs/` (3 files, committed): `v134_fred_lag_autoresearch_log.jsonl`, `v134_fred_lag_search_summary.md`, `v134_lag_candidate.json`

## Tests

- [`tests/research/test_research_v134_fred_lag_sweep.py`](../../../tests/research/test_research_v134_fred_lag_sweep.py)
