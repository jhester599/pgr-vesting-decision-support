# x11 Capital Synthesis Memo

## Scope

x11 synthesizes x9 BVPS bridge evidence and x10 special-dividend
capital-feature evidence against earlier x4/x6 baselines. It does
not train models or alter production/monthly/shadow artifacts.

## Recommendation

- Status: `continue_research`.
- Rationale: BVPS bridge evidence is horizon-specific and annual dividend confidence remains constrained.

## BVPS Comparison

- 1m: x9 `elastic_net_bridge` (logical_interactions) beat x4 (delta MAE -0.048).
- 3m: x9 `ridge_bridge` (bvps_lags) beat x4 (delta MAE -0.010).
- 6m: x9 `drift_bvps_growth` (baseline) did not beat x4 (delta MAE 0.000).
- 12m: x9 `drift_bvps_growth` (baseline) did not beat x4 (delta MAE 0.185).

## Dividend Comparison

- x10 `x9_capital_generation` / `logistic_l2_balanced__ridge_positive_excess` vs x6 `historical_rate__ridge_positive_excess`: EV MAE delta -0.058; confidence `low` with 18 OOS annual observations.

## Decision Questions

- Did x9 improve BVPS enough to supersede x4? Criterion: x9 must beat x4 on future BVPS MAE in most horizons. Answer: x9 beat x4 in 2 horizons.
- Did x9 capital features help special dividends? Criterion: x10 must beat x6 on expected-value MAE. Answer: yes
- Is this ready for shadow wiring? Criterion: Monthly and annual evidence must both be robust. Answer: continue_research

## Next Research Step

Run a narrower x12 BVPS target audit focused on capital-return
adjusted BVPS, December special-dividend discontinuities, and
whether the 12m x4 tree result is robust or target-regime driven.
