# x3 Research Memo

## Scope

x3 runs the research-only direct forward-return and log-return
regression benchmark for absolute PGR forecasting. It does not alter
production or monthly shadow artifacts.

## Results By Horizon

- 1m best price-MAE row: `no_change` (return, price MAE 4.116, return RMSE 0.079, hit rate 0.468); no-change price MAE 4.116, return RMSE 0.079; did not clear no-change gate.
- 3m best price-MAE row: `no_change` (return, price MAE 6.957, return RMSE 0.135, hit rate 0.390); no-change price MAE 6.957, return RMSE 0.135; did not clear no-change gate.
- 6m best price-MAE row: `ridge_log_return` (log_return, price MAE 11.004, return RMSE 0.206, hit rate 0.579); no-change price MAE 11.340, return RMSE 0.200; did not clear no-change gate.
- 12m best price-MAE row: `drift` (return, price MAE 12.446, return RMSE 0.239, hit rate 0.654); no-change price MAE 15.217, return RMSE 0.268; cleared no-change gate.

## Interpretation

- Treat any apparent direct-return edge as preliminary until x4/x5
  decomposition benchmarks exist.
- x3 uses horizon-specific WFO gaps and fold-local preprocessing.
- The direct-return feature subset excludes `pe_ratio` because the
  cached feature has extreme historical values; valuation exposure
  remains represented by `pb_ratio`.
- Raw future price-level regression remains deferred; x3 derives
  implied future price from return predictions.
