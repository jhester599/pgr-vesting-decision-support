# v142 EDGAR Lag Search Summary

Date: 2026-04-16

## Baseline

- incumbent candidate: `lag=2`
- pooled headline metrics:
  - `pooled_oos_r2=-0.1578`
  - `pooled_ic=0.1261`
  - `pooled_hit_rate=0.6906`

## Sweep

Tested values: `0`, `1`, `2`, `3`

Notable result:

- `lag=1` improved pooled IC to `0.1492`
- but it weakened pooled OOS R^2 to `-0.1638`
- and reduced pooled hit rate to `0.6810`

## Decision

Keep `results/research/v142_edgar_lag_candidate.txt` at `2`.

This block refreshes the leakage-guard evidence but does not justify a research
candidate change.
