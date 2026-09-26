# v26 Productionization And Diagnostic-Cleanup Plan

Created: 2026-04-05

## Goal

Take the corrected `v25` foundation and finish the narrow productionization
work that remains before packaging the branch.

`v26` is intentionally small:

- keep the `v13.1` recommendation layer unchanged
- keep `ensemble_ridge_gbt_v18` as the visible promoted cross-check
- clean the remaining comparison-layer warning noise
- run one fresh production-style monthly dry run on the corrected foundation

## Scope

### v26.0 - Package The Promotion State Cleanly

- keep `config.V22_PROMOTED_CROSS_CHECK_CANDIDATE` at
  `ensemble_ridge_gbt_v18`
- verify the monthly decision path still surfaces that candidate as the visible
  cross-check
- update the repo documentation so `v22` is no longer treated as provisional

### v26.1 - Comparison-Layer Warning Cleanup

- remove the `invalid value encountered in divide` warning path in the
  historical comparison scripts
- keep the fix local to the reporting / comparison helpers rather than
  scattering warning filters through every script

### v26.2 - Production-Facing Validation

- run:
  - `python scripts/v21_historical_comparison.py`
  - `python scripts/monthly_decision.py --as-of 2026-04-04 --dry-run --skip-fred`
- confirm:
  - no new alignment warnings
  - promoted visible cross-check still renders
  - recommendation layer behavior is unchanged

## Expected Outputs

- `docs/results/V26_RESULTS_SUMMARY.md`
- `docs/closeouts/V26_CLOSEOUT_AND_V27_NEXT.md`
- small README / governance updates as needed

## Non-Goals

- no new feature work
- no benchmark-universe redesign
- no new promotion study
- no change to the active recommendation layer policy
