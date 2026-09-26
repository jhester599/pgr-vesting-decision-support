# v133 — ridge alpha sweep

<!-- Registry entry: research/registry.yaml (id: v133). -->

**Question.** Ridge alpha-grid sweep on the production research frame.

**Date.** 2026-04-14 (first commit).

**Status.** `closed`.

## Run

From the repository root:

```bash
python research/studies/v133_ridge_alpha_sweep/v133_ridge_alpha_sweep.py
```

## Outputs

`outputs/` (3 files, committed): `v133_alpha_max_candidate.txt`, `v133_ridge_alpha_autoresearch_log.jsonl`, `v133_ridge_alpha_search_summary.md`

## Tests

- [`tests/research/test_research_v133_ridge_alpha_sweep.py`](../../../tests/research/test_research_v133_ridge_alpha_sweep.py)
