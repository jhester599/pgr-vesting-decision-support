# V24 Results Summary

Created: 2026-04-05

## Scope

- v24 tests whether replacing `VOO` with `VTI` improves the reduced forecast universe when everything else is held fixed.
- It compares the current VOO-based universe, an actual VTI replacement universe, and a stitched-history VTI replacement universe.

## Decision

- Status: `keep_voo`
- Recommended universe: `current_voo_actual`
- Rationale: Replacing VOO with VTI did not improve the leading candidate cleanly enough on agreement, policy utility, and OOS fit to justify changing the reduced forecast universe.

## Scenario Summary

### current_voo_actual

- Common window: `2016-10-31` to `2025-09-30` (108 monthly dates)
- Mean sign-policy return: `0.0798`
- Mean OOS R^2: `-0.1424`
- Mean IC: `0.1911`
- Signal agreement with simpler baseline: `84.3%`
- Signal changes: `15`

### vti_replacement_actual

- Common window: `2016-10-31` to `2025-09-30` (108 monthly dates)
- Mean sign-policy return: `0.0782`
- Mean OOS R^2: `-0.2179`
- Mean IC: `0.1965`
- Signal agreement with simpler baseline: `83.3%`
- Signal changes: `11`

### vti_replacement_stitched

- Common window: `2013-04-30` to `2025-09-30` (150 monthly dates)
- Mean sign-policy return: `0.0748`
- Mean OOS R^2: `-0.2158`
- Mean IC: `0.2062`
- Signal agreement with simpler baseline: `80.0%`
- Signal changes: `20`

