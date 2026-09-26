# x4 Research Memo

## Scope

x4 runs the research-only BVPS forecasting leg for the future
BVPS x P/B decomposition benchmark. It does not forecast P/B,
recombine implied price, or alter production artifacts.

## Results By Horizon

- 1m best BVPS-MAE row: `ridge_bvps_growth` (growth, BVPS MAE 0.629, growth RMSE 0.0361, hit rate 0.737); no-change BVPS MAE 0.717, growth RMSE 0.0374; cleared no-change BVPS gate.
- 3m best BVPS-MAE row: `drift_bvps_growth` (growth, BVPS MAE 1.309, growth RMSE 0.0645, hit rate 0.694); no-change BVPS MAE 1.481, growth RMSE 0.0701; cleared no-change BVPS gate.
- 6m best BVPS-MAE row: `drift_bvps_growth` (growth, BVPS MAE 2.010, growth RMSE 0.0992, hit rate 0.747); no-change BVPS MAE 2.260, growth RMSE 0.1150; cleared no-change BVPS gate.
- 12m best BVPS-MAE row: `hist_gbt_bvps_growth` (growth, BVPS MAE 3.017, growth RMSE 0.1595, hit rate 0.907); no-change BVPS MAE 4.018, growth RMSE 0.2042; cleared no-change BVPS gate.

## Interpretation

- x4 isolates the BVPS leg; x5 must test whether P/B forecasting
  and recombination improve implied-price accuracy.
- BVPS targets use PGR monthly EDGAR BVPS normalized to the
  feature matrix's lagged business-month-end availability
  calendar.
- Treat BVPS model edges as structural inputs, not trading signals,
  until the P/B leg and recombined price benchmark exist.
