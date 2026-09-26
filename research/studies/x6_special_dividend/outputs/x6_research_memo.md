# x6 Research Memo

## Scope

x6 runs a research-only two-stage Q1 special-dividend sidecar using
November business-month-end snapshots only. It does not alter
production or monthly shadow artifacts.

## Sample

- OOS annual predictions per model: 18.
- The normal quarterly dividend baseline is inferred from repo
  dividend history by x1 target utilities, not hardcoded.

## Results

- Best expected-value row: `historical_rate__ridge_positive_excess` (EV MAE 1.497, stage-1 Brier 0.265, stage-2 positive MAE 2.022).

## Interpretation

- This annual sample is very small; treat all apparent edges as
  fragile and hypothesis-generating.
- Ridge conditional-size predictions are capped to the prior
  training-fold positive excess range to avoid false precision from
  the tiny positive-only sample.
- x6 should remain complementary to the BVPS/capital-generation
  research lane until a later synthesis step.
