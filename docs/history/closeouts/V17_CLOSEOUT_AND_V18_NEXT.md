# V17 Closeout And V18 Next

Created: 2026-04-04

## Closeout

- v17 evaluated whether the modified Ridge+GBT pair is strong enough to replace the current live stack as the visible production cross-check.

## Result

- Status: `keep_current_live_cross_check`
- Recommended path: `live_production`
- Rationale: The modified Ridge+GBT candidate improved the reduced-universe metrics, but it did not behave clearly enough versus the current live cross-check over recent monthly snapshots to justify replacing the current cross-check path.

More specifically:

- the modified pair was much steadier than the current live cross-check
- but it disagreed with the promoted simpler baseline on direction in every reviewed month
- so replacing the current cross-check now would likely make the production report more confusing, not less

## Recommended V18 Scope

- keep the current production cross-check unchanged
- do not replace the live cross-check with the modified Ridge+GBT pair yet
- focus v18 on explaining and reducing the candidate's directional bias versus the promoted simpler baseline
- highest-value next targets:
  - deferred benchmark-side / peer-relative feature families
  - sign-bias and calibration diagnostics on the reduced universe
  - narrow tests that preserve the lean Ridge+GBT stack rather than reopening model complexity
