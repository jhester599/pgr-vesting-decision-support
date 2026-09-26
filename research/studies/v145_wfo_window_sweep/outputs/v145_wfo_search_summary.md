# v145 WFO Window Search Summary

Date: 2026-04-16

## Baseline

- incumbent candidate: `{"train": 60, "test": 6}`
- pooled headline metrics:
  - `pooled_oos_r2=-0.1578`
  - `pooled_ic=0.1261`
  - `pooled_hit_rate=0.6906`

## Sweep

Tested pairs:

- `(48, 6)`
- `(54, 6)`
- `(60, 6)`
- `(72, 6)`
- `(60, 3)`
- `(60, 9)`

The most interesting tradeoff was `(48, 6)`, which improved pooled OOS R^2 to
`-0.1570` and pooled IC to `0.1856` but reduced pooled hit rate to `0.6535`.

## Decision

Keep `results/research/v145_wfo_candidate.json` at `{"train": 60, "test": 6}`.

No tested bounded window pair improved the full headline set cleanly enough to
replace the incumbent.
