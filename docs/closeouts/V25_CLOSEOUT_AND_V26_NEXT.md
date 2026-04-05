# V25 Closeout And V26 Next

Created: 2026-04-05

## v25 Closeout

`v25` is complete.

It did what the external peer reviews asked for:

- fixed the active monthly-index mismatch
- fixed the remaining WFO / inner-CV / CPCV integrity issues
- added explicit guards around a previously fragile ROE / feature-group path
- reran the promotion-sensitive `v20-v24` studies on the corrected foundation

## What Changed In The Decision Picture

The important governance picture is now cleaner:

- `v20` still should not be treated as the final promotion basis
- `v21` remains the authoritative promotion study for the visible cross-check
- `v23` confirms that result on the longer stitched-history window
- `v24` still says `keep_voo`

So the `v25` bottom line is:

- the repo does **not** need another broad feature or model sweep right now
- the repo **does** have a stronger basis for the `v22` cross-check promotion

## Recommended Next Step: v26

`v26` should be a narrow productionization / calibration cycle, not a new
search program.

### v26.0 - Package The Promotion Cleanly

- carry forward the `v22` visible cross-check promotion with the `v25`
  correctness fixes included
- ensure docs and governance language consistently reference
  `ensemble_ridge_gbt_v18` as the promoted visible cross-check candidate

### v26.1 - Clean The Remaining Comparison-Layer Warning Noise

- remove the remaining numpy invalid-divide warning paths in the historical
  comparison scripts
- use constant-safe correlation / agreement helpers so diagnostic logs stay
  readable

### v26.2 - Production-Facing Validation

- run a fresh monthly dry run on the corrected `v25` foundation
- confirm that:
  - the recommendation layer is unchanged
  - the visible cross-check is the promoted candidate
  - no new silent-failure warnings or date-misalignment issues appear

## Explicit Non-Goals For v26

- no new broad feature sweep
- no new model-family experimentation
- no `VOO -> VTI` swap
- no new benchmark-universe redesign

## Practical Conclusion

The peer reviews were useful.

Their best recommendation was not "try something more complex," it was "fix the
foundation first." `v25` did that, and the key historical promotion result
survived. The next step should therefore be a narrow productionization cycle,
not another exploratory research branch.
