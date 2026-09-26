# v126 -- Path B Methodology Hardening (v125 Remediation)

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

| Metric | Path A (matched) | Path B (matched) | Delta |
|--------|-------------------|---------------|-------|
| n_obs | 84 | 84 | +0.0000 |
| n_covered | 64 | 70 | +6.0000 |
| coverage | 0.7619 | 0.8333 | +0.0714 |
| balanced_accuracy_all | 0.5353 | 0.6112 | +0.0759 |
| balanced_accuracy_covered | 0.5000 | 0.6450 | +0.1450 |
| brier_score | 0.2058 | 0.2188 | +0.0130 |
| log_loss | 0.6132 | 0.8207 | +0.2075 |
| ece | 0.1269 | 0.2234 | +0.0965 |
| base_rate_positive | 0.2738 | 0.2738 | +0.0000 |


## Architecture Verdict

Path B improves covered balanced accuracy on the matched v126 comparison, but calibration worsens versus Path A. Keep Path B as a secondary research track until calibration work closes that gap.

## Notes

- Path A is recomputed here as the matched benchmark-specific baseline:
  separate per-benchmark logistic classifiers, prequential calibration,
  and row-wise renormalized fixed investable weights.
- Path B trains a single logistic classifier on the composite weighted relative
  return rather than separate per-benchmark classifiers.
- Composite target: sell signal when weighted relative return < -3%.
- Rolling WFO parameters: requested_splits=15, realized_splits=14,
  train_window=60, gap=8, test_size=6,
  min_train_obs=60.
- Feature set: lean_baseline.
- Abstain band: [0.30, 0.70] (coverage filter
  uses |prob - 0.5| > 0.2).
- Path A available benchmark count per matched month: 6-6.
