# V15 Closeout And V16 Next

Created: 2026-04-04

## Closeout

v15 is complete for the research features that were available or could be engineered from the repo's existing data sources without broad new ingestion work.

Completed:

- archived and reviewed 4 external research reports
- built a canonical candidate inventory
- generated a one-feature-at-a-time swap queue
- ran `v15.0` exhaustive screening on the surviving v14 models
- ran `v15.1` cross-model confirmation on all deployed model types
- ran `v15.2` final cross-model bakeoff

## What v15 proved

The reports were useful.

They were right that the current feature set still had replaceable weak spots, especially:

- `vmt_yoy`
- parts of the generic rate / volatility block
- some generic linear-model company features

The clearest winners were:

- `rate_adequacy_gap_yoy` for GBT
- `book_value_per_share_growth_yoy` for the linear models

The best single v15 model result was:

- `gbt_lean_plus_two__v15_best`
  - `rate_adequacy_gap_yoy` replacing `vmt_yoy`

## What v15 did not prove

v15 did not prove that the project is ready for a production prediction-stack replacement.

Why:

- the strongest candidates still have negative mean OOS R²
- the final stage compared individual modified models, not yet a promoted modified ensemble
- several harder candidate features remain deferred because they need extra plumbing:
  - peer-relative underwriting features
  - cleaner benchmark-side valuation features
  - some harder credit / term-premium constructions

## Recommended v16 Scope

v16 should be a narrow promotion study, not a broad new brainstorm.

Recommended `v16.0`:

- build a modified `ridge + gbt` candidate stack using:
  - Ridge:
    - `book_value_per_share_growth_yoy` replacing `roe_net_income_ttm`
  - GBT:
    - `rate_adequacy_gap_yoy` replacing `vmt_yoy`
- compare that modified pair against:
  - current live prediction stack
  - current v14/v15 baselines
  - `historical_mean`

Recommended `v16.1`:

- only if the modified pair wins, evaluate whether the live prediction layer should adopt those replacements while leaving the v13.1 recommendation layer intact

Recommended `v16.2`:

- if needed, extend only the highest-value deferred v15 families:
  - `pgr_cr_vs_peer_cr`
  - `pgr_price_to_book_relative`
  - `equity_risk_premium`
  - `mortgage_spread_30y_10y`

## Production Recommendation

Current recommendation:

- keep the live production stack unchanged for now
- keep the promoted recommendation layer unchanged
- use v15 as a successful feature-research milestone
- use v16 as the promotion gate for the best confirmed v15 replacements
