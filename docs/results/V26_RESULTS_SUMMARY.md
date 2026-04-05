# V26 Results Summary

Created: 2026-04-05

## Scope

- `v26` is the narrow productionization / cleanup pass after `v25`.
- It does not change the active recommendation layer or the promoted visible
  cross-check candidate.
- It focuses on cleaning remaining diagnostic warning noise and validating the
  production monthly path on the corrected `v25` foundation.

## What Changed

- [backtest_report.py](/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/src/reporting/backtest_report.py)
  now handles constant ranked series safely when computing Newey-West-style IC
  summaries.
- That removes the repeated numpy `invalid value encountered in divide` warning
  path that was still appearing in `v21`, `v23`, and `v24` reruns.

## Validation

- `python scripts/v21_historical_comparison.py`
  - completed without the previous invalid-divide warning spam
- `python scripts/monthly_decision.py --as-of 2026-04-04 --dry-run --skip-fred`
  - completed successfully on the corrected foundation

## Current Production-Facing State

- Active recommendation layer:
  - `shadow_promoted`
- Visible promoted cross-check:
  - `ensemble_ridge_gbt_v18`

## April Dry-Run Snapshot

- Consensus signal: `OUTPERFORM (LOW CONFIDENCE)`
- Predicted 6M relative return: `-0.31%`
- Recommendation mode: `DEFER-TO-TAX-DEFAULT`
- Sell %: `50%`
- Visible cross-check:
  - `ensemble_ridge_gbt_v18`
  - `DEFER-TO-TAX-DEFAULT`
  - `sell 50%`
  - `-15.29%`

## Conclusion

`v26` does not change the decision framework. It confirms that:

- the `v25` correctness fixes are compatible with the production monthly path
- the promoted visible cross-check still renders correctly
- the remaining comparison-layer diagnostic noise is materially reduced

This means the branch is now in good shape to package as one combined
`v25-v26` correctness / productionization PR.
