# v141 Blend Weight Search Summary

Date: 2026-04-16

## Baseline

- midpoint baseline candidate: `ridge_weight=0.50`
- pooled headline metrics:
  - `pooled_oos_r2=-0.1634`
  - `pooled_ic=0.1250`
  - `pooled_hit_rate=0.6906`

## Sweep

Tested values: `0.30`, `0.40`, `0.50`, `0.55`, `0.60`, `0.65`, `0.70`

Best bounded candidate:

- `ridge_weight=0.60`
- `pooled_oos_r2=-0.1624`
- `pooled_ic=0.1263`
- `pooled_hit_rate=0.6935`

Nearby values `0.65` and `0.70` nudged hit rate and IC slightly higher, but
they were not as balanced once pooled OOS R^2 was considered.

## Decision

Update `results/research/v141_blend_weight_candidate.txt` to `0.60` and carry
it forward as a research-only candidate for later promotion review.
