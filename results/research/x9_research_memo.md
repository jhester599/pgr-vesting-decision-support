# x9 Research Memo

## Scope

x9 tests BVPS bridge baselines, lagged BVPS features, accounting
features, and pre-registered logical interactions. It is
research-only and does not touch production or shadow artifacts.

## Results By Horizon

- 1m best: `elastic_net_bridge` (logical_interactions, BVPS MAE 0.580, growth RMSE 0.0357, hit rate 0.731).
- 3m best: `ridge_bridge` (bvps_lags, BVPS MAE 1.300, growth RMSE 0.0643, hit rate 0.694).
- 6m best: `drift_bvps_growth` (baseline, BVPS MAE 2.010, growth RMSE 0.0992, hit rate 0.747).
- 12m best: `drift_bvps_growth` (baseline, BVPS MAE 3.202, growth RMSE 0.1630, hit rate 0.840).

## Feature Stability

- `bvps_growth_6m` selected in 100% of folds for 6m `bvps_lags` / `elastic_net_bridge`.
- `premium_to_surplus_x_cr_delta` selected in 100% of folds for 6m `logical_interactions` / `elastic_net_bridge`.
- `monthly_combined_ratio_delta` selected in 100% of folds for 3m `accounting_core` / `elastic_net_bridge`.
- `monthly_combined_ratio_delta` selected in 100% of folds for 6m `accounting_core` / `elastic_net_bridge`.
- `premium_to_surplus_x_cr_delta` selected in 100% of folds for 3m `logical_interactions` / `elastic_net_bridge`.
- `buyback_yield_x_bvps_growth_3m` selected in 100% of folds for 6m `logical_interactions` / `elastic_net_bridge`.
- `month_of_year` selected in 100% of folds for 1m `bvps_lags` / `elastic_net_bridge`.
- `premium_to_surplus_x_cr_delta` selected in 100% of folds for 1m `logical_interactions` / `elastic_net_bridge`.
- `current_bvps` selected in 100% of folds for 12m `bvps_lags` / `ridge_bridge`.
- `current_bvps` selected in 100% of folds for 12m `bridge_combined` / `ridge_bridge`.

## Decision Notes

- Interactions are bounded and economically pre-registered.
- Feature-count discipline is enforced by reporting stability,
  not by promoting the full candidate set.
- x10 should reuse the most interpretable capital-generation
  features for annual special-dividend testing.
