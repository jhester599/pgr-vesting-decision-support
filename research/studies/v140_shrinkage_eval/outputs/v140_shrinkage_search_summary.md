# v140 Shrinkage Search Summary

Date: 2026-04-16

## Baseline

- incumbent candidate: `0.50`
- pooled headline metrics:
  - `pooled_oos_r2=-0.1578`
  - `pooled_ic=0.1261`
  - `pooled_hit_rate=0.6906`

## Sweep

Tested values: `0.35`, `0.40`, `0.45`, `0.50`, `0.55`, `0.60`, `0.65`

All tested values produced the same pooled headline metrics on the current
post-v138 regression frame.

## Decision

Keep `results/research/v140_shrinkage_candidate.txt` at `0.50`.

This block closes as a no-change confirmation rather than a new promotion
candidate.
