# x15 Research Memo

## Scope

x15 tests a bounded P/B regime overlay at the 3m and 6m horizons.
It preserves the research-only boundary and compares against the
existing no-change P/B anchor.

## Results

- 3m best row overall: `no_change_pb_overlay` (P/B MAE 0.287, RMSE 0.386, action rate 0.000).
- 3m best overlay challenger: `hist_gbt_depth2` (P/B MAE 0.327, RMSE 0.444, action rate 0.383); did not beat the no-change anchor.
- 6m best row overall: `no_change_pb_overlay` (P/B MAE 0.393, RMSE 0.557, action rate 0.000).
- 6m best overlay challenger: `logistic_l2_balanced` (P/B MAE 0.435, RMSE 0.561, action rate 0.713); did not beat the no-change anchor.

## Interpretation

- x15 does not justify replacing `no_change_pb` with this bounded
  regime overlay.
- Treat any overlay improvement as provisional until it is
  recombined with the structural BVPS path in a later x-series
  step.
- The x15 overlay is intentionally bounded and does not let the
  classifier emit arbitrary P/B levels.
