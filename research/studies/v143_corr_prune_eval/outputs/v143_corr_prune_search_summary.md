# v143 Correlation Pruning Search Summary

Date: 2026-04-16

## Baseline

- incumbent candidate: `rho=0.95`
- pooled headline metrics:
  - `pooled_oos_r2=-0.1578`
  - `pooled_ic=0.1261`
  - `pooled_hit_rate=0.6906`

## Sweep

Tested values: `0.80`, `0.85`, `0.90`, `0.95`, `0.99`

Best bounded candidate:

- `rho=0.80`
- `pooled_oos_r2=-0.1569`
- `pooled_ic=0.1411`
- `pooled_hit_rate=0.6944`

## Decision

Update `results/research/v143_corr_prune_candidate.txt` to `0.80` and carry it
forward as the new research-only threshold.
