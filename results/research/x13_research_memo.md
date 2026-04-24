# x13 Research Memo

## Scope

x13 compares raw and dividend-adjusted BVPS decomposition paths at
the 3m and 6m horizons, keeping `no_change_pb` as the P/B anchor.

## Results

- 3m raw best `drift_bvps_growth__no_change_pb` (baseline, price MAE 8.196) vs adjusted best `elastic_net_bridge__no_change_pb` (logical_interactions, price MAE 9.017).
- 6m raw best `drift_bvps_growth__no_change_pb` (baseline, price MAE 12.143) vs adjusted best `ridge_bridge__no_change_pb` (bvps_lags, price MAE 11.675).

## Interpretation

- If adjusted decomposition improves the same horizons that x12
  improved, it becomes the best current candidate for a structural
  x-series indicator.
- x14 should only nominate an indicator if this adjusted path is
  directionally consistent with the broader x-series evidence.
