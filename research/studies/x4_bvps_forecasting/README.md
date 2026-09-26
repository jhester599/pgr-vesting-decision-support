# x4 — bvps forecasting

<!-- Registry entry: research/registry.yaml (id: x4). -->

**Question.** BVPS forecasting benchmarks.

**Date.** 2026-04-22 (first commit).

**Status.** `closed`.

**Notes.** x-series: research-only PGR absolute-return lane (see docs/research/x_series_resume_2026-04-24.md).

**Closeout / record.** [2026-04-22-x4-bvps-forecasting-leg.md](../../../docs/superpowers/plans/2026-04-22-x4-bvps-forecasting-leg.md)

## Run

From the repository root:

```bash
python research/studies/x4_bvps_forecasting/x4_bvps_forecasting.py
```

## Outputs

`outputs/` (3 files, committed): `x4_bvps_forecasting_detail.csv`, `x4_bvps_forecasting_summary.json`, `x4_research_memo.md`

## Tests

- [`tests/research/test_research_x4_bvps_forecasting.py`](../../../tests/research/test_research_x4_bvps_forecasting.py)
