# x19 Research Memo

## Scope

x19 rebuilds the annual dividend sidecar on post-policy snapshots only,
using the x18 regime-aware label and persistent-BVPS-capital features.

## Sample

- OOS annual predictions per model-feature set: 3.
- Validation remains expanding annual train/test splits with post-policy-only data.

## Results

- Best row: `persistent_capital_generation` / `logistic_l2_balanced__ridge_positive_excess` (EV MAE 4.406, stage-1 Brier 0.128, stage-1 balanced accuracy 0.667).

## Interpretation

- x19 is intentionally low-sample and low-confidence.
- The point is to test whether cleaner labels plus persistent-BVPS
  state improve the dividend lane enough to keep pursuing.
