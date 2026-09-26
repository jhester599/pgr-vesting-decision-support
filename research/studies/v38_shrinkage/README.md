# v38 — shrinkage

<!-- Registry entry: research/registry.yaml (id: v38). -->

**Question.** Post-hoc prediction shrinkage: y_hat_shrunk = alpha * y_hat.

**Date.** 2026-04-10 (first commit).

**Status.** `promoted`, feeds `config.model.ENSEMBLE_SHRINKAGE_ALPHA_GRID`.

**Notes.** v38's shrinkage rule. Since review 2026-09-25 (F13) production applies it prequentially; the fixed alpha 0.50 (ENSEMBLE_PREDICTION_SHRINKAGE_ALPHA) is research-only. Re-run pending (WP11).

**Closeout / record.** [2026-04-10-v37-v60-results-summary.md](../../../docs/superpowers/plans/2026-04-10-v37-v60-results-summary.md)

## Run

From the repository root:

```bash
python research/studies/v38_shrinkage/v38_shrinkage.py
```

## Outputs

`outputs/` (2 files, committed): `v38_shrinkage_best_results.csv`, `v38_shrinkage_results.csv`

## Tests

- [`tests/research/test_research_v38_shrinkage.py`](../../../tests/research/test_research_v38_shrinkage.py)
