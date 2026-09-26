# v131 — threshold sweep eval

<!-- Registry entry: research/registry.yaml (id: v131). -->

**Question.** Asymmetric abstention threshold evaluator for Path B temperature-scaled classifier.

**Date.** 2026-04-13 (first commit).

**Status.** `retained`, feeds `config.model.SHADOW_CLASSIFIER_HIGH_THRESH / SHADOW_CLASSIFIER_LOW_THRESH`.

**Notes.** Found (0.15, 0.70); v132's hold-out rejected it, so the thresholds stay at (0.30, 0.70).

## Run

From the repository root:

```bash
python research/studies/v131_threshold_sweep_eval/v131_threshold_sweep_eval.py
```

## Outputs

`outputs/` (1 files, committed): `v131_threshold_sweep_log.jsonl`

## Tests

- [`tests/research/test_research_v131_threshold_sweep_eval.py`](../../../tests/research/test_research_v131_threshold_sweep_eval.py)
