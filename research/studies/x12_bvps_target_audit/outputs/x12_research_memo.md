# x12 Research Memo

## Scope

x12 audits whether raw BVPS targets are being distorted by capital
return events. It compares raw vs dividend-adjusted BVPS targets
using bounded x9-style baselines and regularized models.

## Discontinuities

- Flagged discontinuity months: 11.
- Share of discontinuities in December/January: 27%.

## Raw vs Adjusted Leaders

- 1m raw best `elastic_net_bridge` (logical_interactions, MAE 0.598) vs adjusted best `ridge_bridge` (bvps_lags, MAE 0.745).
- 3m raw best `ridge_bridge` (bvps_lags, MAE 1.308) vs adjusted best `elastic_net_bridge` (logical_interactions, MAE 1.067).
- 6m raw best `drift_bvps_growth` (baseline, MAE 2.057) vs adjusted best `ridge_bridge` (bvps_lags, MAE 1.964).
- 12m raw best `drift_bvps_growth` (baseline, MAE 3.190) vs adjusted best `drift_bvps_growth` (baseline, MAE 3.949).

## Interpretation

- If adjusted targets help most at longer horizons, the next
  logical step is an adjusted BVPS x P/B recombination pass.
- If adjusted targets do not help much, the remaining weakness is
  more likely target-regime or model-specification driven than
  pure dividend discontinuity noise.
