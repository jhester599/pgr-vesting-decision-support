# x10 Research Memo

## Scope

x10 re-tests the annual Q1 special-dividend sidecar with x9
capital-generation features. It remains research-only and uses
November business-month-end snapshots only.

## Sample

- OOS annual predictions per model-feature set: 18.
- Validation remains expanding annual train/test splits.

## Results

- Best row: `x9_capital_generation` / `logistic_l2_balanced__ridge_positive_excess` (EV MAE 1.439, stage-1 Brier 0.288, stage-1 balanced accuracy 0.583).

## Interpretation

- x10 should be read as feature-set diagnostics, not as a dividend
  deployment model.
- Annual sample size remains the main confidence limiter.
- x11 should compare x10 against x6 and document whether x9 capital
  features improved the dividend sidecar enough to continue.
