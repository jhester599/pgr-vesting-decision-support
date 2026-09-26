# v9 — experiments

<!-- Registry entry: research/registry.yaml (id: v9). -->

**Question.** Which features, targets, pooled benchmark families and classifier set-ups work on the reduced v9 benchmark universe, including point-in-time weekly snapshots? (five scripts: feature, target, pooled-benchmark, confirmatory-classifier and weekly-snapshot experiments)

**Date.** 2026-04-03 (first commit).

**Status.** `closed`.

**Closeout / record.** [V9_CLOSEOUT_AND_V91_NEXT.md](../../../docs/closeouts/V9_CLOSEOUT_AND_V91_NEXT.md)

## Run

From the repository root:

```bash
python research/studies/v9_experiments/confirmatory_classifier_experiments.py
python research/studies/v9_experiments/feature_experiments.py
python research/studies/v9_experiments/pooled_benchmark_experiments.py
python research/studies/v9_experiments/target_experiments.py
python research/studies/v9_experiments/weekly_snapshot_experiments.py
```

## Outputs

The committed outputs of these scripts are in [`research/legacy/v9/`](../../../research/legacy/v9/) (unchanged). New runs write to `outputs/` in this folder.

## Tests

- [`tests/research/test_confirmatory_classifier_experiments.py`](../../../tests/research/test_confirmatory_classifier_experiments.py)
- [`tests/research/test_feature_experiments.py`](../../../tests/research/test_feature_experiments.py)
- [`tests/research/test_pooled_benchmark_experiments.py`](../../../tests/research/test_pooled_benchmark_experiments.py)
- [`tests/research/test_target_experiments.py`](../../../tests/research/test_target_experiments.py)
- [`tests/research/test_weekly_snapshot_experiments.py`](../../../tests/research/test_weekly_snapshot_experiments.py)
