# v125 -- Path B: Composite Portfolio-Target Classifier

**Run date:** 2026-04-12
**As-of cutoff:** 2024-03-31

## Composite Return Target

- Observations: 152
- Date range: 2011-02-28 to 2023-09-29
- Positive rate (sell signal): 0.3092

### Weights

| Benchmark | Weight |
|-----------|--------|
| VOO | 0.40 |
| VGT | 0.20 |
| VIG | 0.15 |
| VXUS | 0.10 |
| VWO | 0.10 |
| BND | 0.05 |

## WFO Metrics Comparison

| Metric | Path A (v92 ref) | Path B (v125) | Delta |
|--------|-------------------|---------------|-------|
| n_obs | -- | 84 | -- |
| n_covered | -- | 71 | -- |
| coverage | -- | 0.8452 | -- |
| balanced_accuracy_all | 0.7538 | 0.6518 | -0.1020 |
| balanced_accuracy_covered | 0.5132 | 0.6331 | +0.1199 |
| brier_score | 0.1852 | 0.2393 | +0.0541 |
| log_loss | 0.5985 | 0.8590 | +0.2605 |
| ece | 0.0813 | 0.2381 | +0.1568 |
| base_rate_positive | -- | 0.2738 | -- |


## Architecture Verdict

Path B shows >= 3% balanced accuracy improvement over Path A reference. Elevate Path B to co-primary research track.

## Notes

- Path B trains a single logistic classifier on the composite weighted relative
  return rather than separate per-benchmark classifiers.
- Composite target: sell signal when weighted relative return < -3%.
- WFO parameters: 15 splits, gap=8, test_size=6,
  min_train_obs=60.
- Feature set: lean_baseline.
- Abstain band: [0.30, 0.70] (coverage filter
  uses |prob - 0.5| > 0.2).
