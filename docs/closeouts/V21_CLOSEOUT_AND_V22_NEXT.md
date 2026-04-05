# V21 Closeout And V22 Next

Created: 2026-04-04

## Closeout

- v21 replaced the recent-window promotion gate with a point-in-time historical comparison over the full common evaluable period.

## Result

- Status: `promote_candidate_cross_check`
- Recommended path: `ensemble_ridge_gbt_v18`
- Rationale: The best v21 candidate improved on the reduced live stack and matched or exceeded the live cross-check's agreement with the promoted simpler baseline over the full historical window.

## Recommended V22 Scope

- If the current live cross-check still wins historically, avoid another generic feature sweep.
- Focus next on blocked-source expansion or narrow calibration / sign-bias diagnostics.
