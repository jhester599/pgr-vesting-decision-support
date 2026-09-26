# v41 — winsorize

<!-- Registry entry: research/registry.yaml (id: v41). -->

**Question.** Target winsorization within each WFO fold.

**Date.** 2026-04-10 (first commit).

**Status.** `closed`.

**Closeout / record.** [2026-04-10-v37-v60-results-summary.md](../../../docs/superpowers/plans/2026-04-10-v37-v60-results-summary.md)

## Run

From the repository root:

```bash
python research/studies/v41_winsorize/v41_winsorize.py
```

## Outputs

`outputs/` (1 files, committed): `v41_winsorize_results.csv`

## Tests

- [`tests/research/test_research_v41_winsorize.py`](../../../tests/research/test_research_v41_winsorize.py)
