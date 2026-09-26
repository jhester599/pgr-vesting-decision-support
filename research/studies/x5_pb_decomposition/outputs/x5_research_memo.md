# x5 Research Memo

## Scope

x5 runs the research-only future P/B leg and recombined BVPS x
P/B implied-price benchmark. It does not alter production artifacts.

## Recombined Results By Horizon

- 1m best decomposition row: `no_change_bvps__no_change_pb` (price MAE 4.515, RMSE 7.931, hit rate 0.430).
- 3m best decomposition row: `drift_bvps_growth__no_change_pb` (price MAE 7.097, RMSE 11.825, hit rate 0.672).
- 6m best decomposition row: `drift_bvps_growth__no_change_pb` (price MAE 10.782, RMSE 18.965, hit rate 0.747).
- 12m best decomposition row: `ridge_bvps_growth__no_change_pb` (price MAE 16.053, RMSE 28.165, hit rate 0.778).

## P/B Leg Takeaway

- 1m best P/B row: `no_change_pb` (pb, P/B MAE 0.193, RMSE 0.270).
- 3m best P/B row: `no_change_pb` (pb, P/B MAE 0.291, RMSE 0.388).
- 6m best P/B row: `no_change_pb` (pb, P/B MAE 0.395, RMSE 0.561).
- 12m best P/B row: `no_change_pb` (pb, P/B MAE 0.537, RMSE 0.770).

## Interpretation

- x5 is still research-only and stacked on x4 while x4 is open.
- Compare x5 against x3 direct return after both PRs are merged
  and the x-series artifacts share one base branch.
