# x7 Research Memo

## Scope

x7 runs targeted, replacement-only TA experiments for x-series
absolute-direction classification. It does not add broad TA features
or alter production/monthly/shadow artifacts.

## Results

- `ta_minimal_plus_vwo_pct_b` cleared 2/4 horizons (mean delta BA 0.006, mean delta Brier -0.007).
- `ta_minimal_replacement` cleared 2/4 horizons (mean delta BA 0.004, mean delta Brier -0.017).
- `ta_bollinger_width_probe` cleared 2/4 horizons (mean delta BA 0.004, mean delta Brier -0.016).
- `x2_core_baseline` cleared 0/4 horizons (mean delta BA 0.000, mean delta Brier 0.000).

## Interpretation

- Treat TA as a bounded replacement experiment, not as additive
  indicator expansion.
- x8 should compare this x7 evidence against x2/x3/x4/x5/x6
  before any shadow-readiness recommendation.
