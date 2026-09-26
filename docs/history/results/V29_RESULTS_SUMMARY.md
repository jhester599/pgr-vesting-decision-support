# V29 Results Summary

Created: 2026-04-05

## Objective

v29 improves the interpretation layer of the monthly report and email.

The main question was:

- how should the project present broader forecast context without implying that
  every benchmark is also a buy recommendation?

## What Changed

v29 adds:

- a `Confidence Snapshot` section with explicit pass/fail gates
- benchmark-role labels:
  - `Buy candidate`
  - `Optional substitute`
  - `Context only`
  - `Forecast only`
- clearer plain-English top-line model wording
- clearer tax-scenario wording for the next tranche
- matching interpretation upgrades in the HTML email and plaintext fallback

## Interpretation Improvements

The monthly output now separates three concepts more explicitly:

1. forecast view
2. recommendation mode
3. realistic redeploy destinations

That makes the report more honest about what each benchmark means:

- some funds are useful forecast anchors
- some are valid substitutes if redeploying
- some should not be treated as buy recommendations at all

## Confidence Snapshot

The new confidence block summarizes the core quality gate in one place:

- mean IC
- mean hit rate
- aggregate OOS R^2
- representative CPCV

This makes it easier to explain why a month is:

- `ACTIONABLE`
- `MONITORING-ONLY`
- `DEFER-TO-TAX-DEFAULT`

without forcing the user to parse the full diagnostic appendix.

## Practical Conclusion

v29 does not change the active model or recommendation policy.

It makes the current workflow easier to interpret by:

- showing which comparisons are forecast-only context
- showing which checks passed or failed
- reducing ambiguity in the next-vest and benchmark-detail sections
