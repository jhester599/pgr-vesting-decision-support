# v132 — threshold validation

<!-- Registry entry: research/registry.yaml (id: v132). -->

**Question.** Temporal hold-out validation for asymmetric abstention thresholds.

**Date.** 2026-04-13 (first commit).

**Status.** `retained`, feeds `config.model.SHADOW_CLASSIFIER_HIGH_THRESH / SHADOW_CLASSIFIER_LOW_THRESH`.

**Notes.** Temporal hold-out: DO NOT ADOPT; the thresholds stay at (0.30, 0.70).

**Closeout / record.** [2026-04-13-threshold-constants-and-v132-validation.md](../../../docs/superpowers/plans/2026-04-13-threshold-constants-and-v132-validation.md)

## Run

From the repository root:

```bash
python research/studies/v132_threshold_validation/v132_threshold_validation.py
```

## Outputs

`outputs/` (2 files, committed): `v132_threshold_validation_results.csv`, `v132_threshold_validation_summary.md`

## Tests

- [`tests/research/test_research_v132_threshold_validation.py`](../../../tests/research/test_research_v132_threshold_validation.py)
