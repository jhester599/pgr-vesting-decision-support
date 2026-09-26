# v46 — classification

<!-- Registry entry: research/registry.yaml (id: v46). -->

**Question.** Can a per-benchmark logistic classifier predict whether PGR outperforms each benchmark over 6 months? (script: src/research/binary_classification.py)

**Date.** 2026-04-10 (first commit).

**Status.** `closed`.

**Closeout / record.** [2026-04-10-v37-v60-results-summary.md](../../../docs/superpowers/plans/2026-04-10-v37-v60-results-summary.md)

## Run

From the repository root:

```bash
python -m src.research.binary_classification
```

## Outputs

`outputs/` (1 files, committed): `v46_classification_results.csv`

## Tests

- [`tests/research/test_research_v46_classification.py`](../../../tests/research/test_research_v46_classification.py)
