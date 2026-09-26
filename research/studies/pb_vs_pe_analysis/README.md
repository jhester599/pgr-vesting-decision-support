# pb_vs_pe — analysis

<!-- Registry entry: research/registry.yaml (id: pb_vs_pe). -->

**Question.** Is P/B or P/E the better predictor of PGR's forward returns?

**Date.** 2026-09-24 (first commit).

**Status.** `closed`.

## Run

From the repository root:

```bash
python research/studies/pb_vs_pe_analysis/pb_vs_pe_analysis.py
```

## Outputs

`outputs/` (14 files, committed): `annual_sample.csv`, `encompassing.csv`, `implied_roe.csv`, `noise.csv`, `one_way_full.csv`, `one_way_subperiods.csv`, `oos.csv`, `pgr_valuation_summary.html`, `relative_valuation.csv`, `return_decomposition.csv`, `rolling_ic.csv`, `signals_and_targets.csv`, …

## Tests

- [`tests/research/test_pb_vs_pe.py`](../../../tests/research/test_pb_vs_pe.py)
