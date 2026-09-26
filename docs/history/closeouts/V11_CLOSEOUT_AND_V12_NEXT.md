# V11 Closeout and V12 Next

Created: 2026-04-04

## Status

The v11 diversification-first research loop is complete.

## Conclusion

Do not promote a new live model stack yet.

- Best research candidate: `ensemble_ridge_gbt`
- Best overall policy row: `baseline_historical_mean` with `neutral_band_3pct`
- Main blocker: predictive edge remains modest even after diversification-aware simplification, and the reduced-universe candidate still does not decisively beat the simpler diversification-aware baseline.
- Practical gain from v11 is clearer recommendation logic and a better definition of where sold PGR exposure should go.

## v12 Recommendation

- keep the live production stack unchanged until a reduced-universe candidate clearly beats the baseline
- if continuing research, focus on target quality and decision-policy calibration, not on adding model families
- preserve diversification-first redeploy guidance even before any model promotion
