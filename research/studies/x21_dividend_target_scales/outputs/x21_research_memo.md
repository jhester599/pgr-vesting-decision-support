# x21 Research Memo

## Scope

x21 compares post-policy dividend size targets after x20 concluded
that occurrence is not identifiable on the current overlap sample.

## Results

- Best row: `x10_capital_generation` / `to_current_bvps` / `ridge_scaled` (dollar MAE 3.666, scaled MAE 0.074, OOS folds 3).

## Interpretation

- x21 is a size-only diagnostic, not a deployable dividend model.
- Rankings are based on dollar error after back-transforming any
  normalized target, so normalized elegance does not get a free pass.
- A surviving normalized target must beat raw dollars on dollar MAE.
