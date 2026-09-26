# V27 Closeout And v28 Next

Created: 2026-04-05

## V27 Closeout

v27 is complete.

It adds a practical, repeatable redeploy-portfolio answer to the monthly
workflow and email output. The project can now answer not only:

- should the user sell some PGR shares?

but also:

- what should the user buy with the proceeds?

The v27 default is:

- a high-equity redeploy portfolio
- based on `balanced_pref_95_5`
- with a modest `0.25` signal tilt

The live investable redeploy set is now:

- `VOO`
- `VGT`
- `SCHD`
- `VXUS`
- `VWO`
- `BND`

This keeps the answer aligned with the user's stated preference:

- broad US equity
- tech
- value / dividend
- smaller international
- one bond fund at most

## What v27 Did Not Change

v27 did not:

- replace the sell / hold recommendation layer
- replace the visible prediction-layer cross-check
- automatically shrink the production forecast universe

That was intentional. The monthly buy answer and the prediction universe should
not be conflated without a dedicated forecast-study pass.

## Main Decision

The project should now treat the investable redeploy universe and the forecast
benchmark universe as separate concepts:

- forecast benchmarks may still include contextual funds
- redeploy recommendations should only use realistic buy candidates

## v28 Recommendation

The next logical step is a narrow `v28` forecast-universe review, not another
portfolio-UX pass.

v28 should ask:

- should the production forecast universe itself be pruned to align more
  closely with realistic alternative buys?

That study should compare at least:

1. the current forecast universe
2. the current forecast universe plus the v27 investable-pruning labels
3. a narrower forecast universe built around the v27 buyable set plus any
   clearly necessary contextual benchmarks

v28 should only recommend changing the forecast universe if the narrower set:

- preserves or improves policy utility
- preserves or improves historical agreement with the promoted simpler baseline
- does not materially weaken diversification-aware recommendation usefulness

## Until Then

Production should:

- keep the current v13.1 recommendation layer
- keep the current promoted visible cross-check
- include the new v27 `Suggested Redeploy Portfolio` section in each monthly run
- keep the investable redeploy list pruned even while the forecast universe
  remains broader
