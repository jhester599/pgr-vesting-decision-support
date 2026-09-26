# v137 — gbt param sweep

<!-- Registry entry: research/registry.yaml (id: v137). -->

**Question.** Standalone GBT hyperparameter sweep on the production research frame.

**Date.** 2026-04-14 (first commit).

**Status.** `closed`.

## Run

From the repository root:

```bash
python research/studies/v137_gbt_param_sweep/v137_gbt_param_sweep.py
```

## Outputs

`outputs/` (3 files, committed): `v137_gbt_param_autoresearch_log.jsonl`, `v137_gbt_param_search_summary.md`, `v137_gbt_params_candidate.json`

## Tests

- [`tests/research/test_research_v137_gbt_param_sweep.py`](../../../tests/research/test_research_v137_gbt_param_sweep.py)
