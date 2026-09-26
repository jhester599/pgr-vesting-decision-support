# v127 — path b calibration

<!-- Registry entry: research/registry.yaml (id: v127). -->

**Question.** Path B calibration sweep on the matched v126 fold frame.

**Date.** 2026-04-12 (first commit).

**Status.** `shadow`, feeds `src/models/path_b_classifier.py`.

**Notes.** Path B calibration (shadow).

**Closeout / record.** [2026-04-12-v123-v128-classification-enhancement-plan.md](../../../docs/superpowers/plans/2026-04-12-v123-v128-classification-enhancement-plan.md)

## Run

From the repository root:

```bash
python research/studies/v127_path_b_calibration/v127_path_b_calibration.py
```

## Outputs

`outputs/` (3 files, committed): `v127_path_b_calibration_detail.csv`, `v127_path_b_calibration_results.csv`, `v127_path_b_calibration_summary.md`

## Tests

- [`tests/research/test_research_v127_path_b_calibration.py`](../../../tests/research/test_research_v127_path_b_calibration.py)
