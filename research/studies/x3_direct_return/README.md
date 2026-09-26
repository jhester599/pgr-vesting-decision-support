# x3 — direct return

<!-- Registry entry: research/registry.yaml (id: x3). -->

**Question.** Direct PGR forward-return regression benchmarks.

**Date.** 2026-04-22 (first commit).

**Status.** `closed`.

**Notes.** x-series: research-only PGR absolute-return lane (see docs/research/x_series_resume_2026-04-24.md).

**Closeout / record.** [2026-04-22-x3-direct-return-benchmark.md](../../../docs/superpowers/plans/2026-04-22-x3-direct-return-benchmark.md)

## Run

From the repository root:

```bash
python research/studies/x3_direct_return/x3_direct_return.py
```

## Outputs

`outputs/` (3 files, committed): `x3_direct_return_detail.csv`, `x3_direct_return_summary.json`, `x3_research_memo.md`

## Tests

- [`tests/research/test_research_x3_direct_return.py`](../../../tests/research/test_research_x3_direct_return.py)
