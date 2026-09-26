# x12 — bvps target audit

<!-- Registry entry: research/registry.yaml (id: x12). -->

**Question.** BVPS raw-vs-adjusted target audit.

**Date.** 2026-04-23 (first commit).

**Status.** `closed`.

**Notes.** x-series: research-only PGR absolute-return lane (see docs/research/x_series_resume_2026-04-24.md).

**Closeout / record.** [2026-04-23-x12-bvps-target-audit.md](../../../docs/superpowers/plans/2026-04-23-x12-bvps-target-audit.md)

## Run

From the repository root:

```bash
python research/studies/x12_bvps_target_audit/x12_bvps_target_audit.py
```

## Outputs

`outputs/` (4 files, committed): `x12_bvps_discontinuities.csv`, `x12_bvps_target_audit_detail.csv`, `x12_bvps_target_audit_summary.json`, `x12_research_memo.md`

## Tests

- [`tests/research/test_research_x12_bvps_target_audit.py`](../../../tests/research/test_research_x12_bvps_target_audit.py)
