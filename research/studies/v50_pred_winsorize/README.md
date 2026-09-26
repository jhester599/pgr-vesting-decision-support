# v50 — pred winsorize

<!-- Registry entry: research/registry.yaml (id: v50). -->

**Question.** Prediction winsorization: clip OOS predictions at training percentiles.

**Date.** 2026-04-10 (first commit).

**Status.** `closed`.

**Closeout / record.** [2026-04-10-v37-v60-results-summary.md](../../../docs/superpowers/plans/2026-04-10-v37-v60-results-summary.md)

## Run

From the repository root:

```bash
python research/studies/v50_pred_winsorize/v50_pred_winsorize.py
```

## Outputs

`outputs/` (1 files, committed): `v50_pred_winsorize_results.csv`

## Tests

- [`tests/research/test_research_v50_pred_winsorize.py`](../../../tests/research/test_research_v50_pred_winsorize.py)
