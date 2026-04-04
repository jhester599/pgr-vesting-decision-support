# V11 Results Summary

Created: 2026-04-04

## Headline

v11 scored each alternative by both forecasting usefulness and diversification value relative to PGR. Funds that remain too PGR-like were demoted to contextual-only status.

## Selected Universes

- Forecast benchmark universe: `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE, VFH`
- Recommended diversification universe: `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE`
- Mean diversification score of recommendation universe: `0.459`

## Best Candidate

- Best research candidate: `ensemble_ridge_gbt`
- Diversification-aware utility: `0.0598`
- Mean sign-policy return: `0.0680`
- Mean OOS R^2: `-0.6392`
- Mean IC: `0.2596`

## Best Policy

- Best overall policy row: `baseline_historical_mean` with `neutral_band_3pct`
- Diversification-aware utility: `0.0603`
- Mean policy return: `0.0650`
- Interpretation: the diversification-aware v11 candidate improved predictive quality, but the simpler baseline still held a slight edge on the final policy scorecard.

## Sidecar Classifier

- Best sidecar mode: `abstain_only_overlay`
- Mean balanced accuracy: `0.6710`
- Diversification-aware utility: `0.0658`
- Promotion implication: keep the classifier as a confidence / abstention sidecar only; do not make it the primary decision engine.

## Redeploy Guidance

- `broad_us_equity`: `VOO`. Broad US equity diversifies away from single-stock risk without concentrating further in insurance.
- `international_equity`: `VXUS, VWO`. International equity lowers home-market and insurance concentration.
- `fixed_income`: `VMBS, BND`. Fixed income is the cleanest concentration-reduction bucket when model confidence is weak.
- `real_assets`: `GLD, DBC`. Real assets add inflation and non-equity diversification.
- `sector_context`: `VDE`. Sector funds are context-only unless no stronger diversifying destination is available.

## Dry-Run Review Dates

- `2026-02-28`: `OUTPERFORM` / `DEFER-TO-TAX-DEFAULT` / sell `50%` of next vest.
- `2026-03-31`: `OUTPERFORM` / `DEFER-TO-TAX-DEFAULT` / sell `50%` of next vest.
- `2026-04-02`: `OUTPERFORM` / `DEFER-TO-TAX-DEFAULT` / sell `50%` of next vest.

## Key Conclusion

v11 favors diversification-aware simplification over adding more model complexity. `VFH` and `KIE` stay as context benchmarks, not as preferred destinations for capital leaving PGR. The best reduced-universe candidate still did not clear the bar to replace the simpler diversification-aware baseline.

Detailed CSV outputs are stored in `results\v11`.
