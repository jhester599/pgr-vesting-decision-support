# x5 — pb decomposition

<!-- Registry entry: research/registry.yaml (id: x5). -->

**Question.** P/B leg and recombined decomposition benchmarks.

**Date.** 2026-04-22 (first commit).

**Status.** `closed`.

**Notes.** x-series: research-only PGR absolute-return lane (see docs/research/x_series_resume_2026-04-24.md).

**Closeout / record.** [2026-04-22-x5-pb-decomposition-benchmark.md](../../../docs/superpowers/plans/2026-04-22-x5-pb-decomposition-benchmark.md)

## Run

From the repository root:

```bash
python research/studies/x5_pb_decomposition/x5_pb_decomposition.py
```

## Outputs

`outputs/` (4 files, committed): `x5_decomposition_detail.csv`, `x5_decomposition_summary.json`, `x5_pb_leg_detail.csv`, `x5_research_memo.md`

## Tests

- [`tests/research/test_research_x5_pb_decomposition.py`](../../../tests/research/test_research_x5_pb_decomposition.py)
