# V19 Closeout And V20 Next

## v19 Conclusion

v19 completed the original v15 inventory.

Final coverage:

- `44 / 46` original feature ideas were tested through the swap framework
- `2 / 46` were explicitly closed out as blocked

The blocked features are:

- `pgr_cr_vs_peer_cr`
  - requires point-in-time peer combined-ratio history for `ALL`, `TRV`, `CB`,
    and `HIG`, which the repo does not currently ingest
- `pgr_fcf_yield`
  - requires quarterly operating-cash-flow and capex ingestion from EDGAR,
    which is not present in the current fundamentals schema

That means the repo no longer has unresolved “maybe later” rows from the
original v15 candidate inventory. Everything is now either tested or
explicitly blocked by a named data gap.

## What v19 Added

- public-macro backfill from FRED graph CSV, BLS, and Multpl
- the remaining EDGAR-derived insurance-quality features
- valuation and benchmark-side macro features that were previously only ideas
- a complete traceability matrix under `results/v19/`

Key files:

- `scripts/v19_feature_completion.py`
- `src/research/v19.py`
- `results/v19/v19_feature_traceability_20260404.csv`
- `docs/results/V19_RESULTS_SUMMARY.md`

## What v19 Proved

Among the previously unresolved features, the strongest positive additions were:

- `pgr_pe_vs_market_pe`
- `usd_broad_return_3m`
- `auto_pricing_power_spread`

These improved sign-policy utility in one-for-one tests, but they did not
change the broader conclusion enough to justify a production prediction-layer
promotion on their own.

Many of the other deferred features were neutral to negative once tested,
including:

- `equity_risk_premium`
- `pgr_premium_to_surplus`
- `loss_ratio_ttm`
- `expense_ratio_ttm`
- `mortgage_spread_30y_10y`
- `term_premium_10y`

So the remaining blocker is no longer “we have not tested enough features.”
It is now whether the best confirmed features can be assembled into a candidate
that improves user-facing behavior enough to replace the current live
cross-check.

## Current Recommendation

Keep:

- the promoted `v13.1` recommendation layer
- the current live production cross-check

Do not promote a new prediction layer yet based on v19 alone.

## v20 Recommendation

v20 should be a narrow synthesis and promotion-readiness study rather than
another broad feature hunt.

Recommended scope:

1. build one best-of-v19 candidate stack from the strongest confirmed swaps
2. compare it directly against:
   - the current live production cross-check
   - the best prior reduced-universe candidate
   - the simpler promoted baseline
3. focus on:
   - directional agreement
   - recommendation stability
   - user-facing usefulness
   - whether the candidate reduces confusion rather than only improving
     reduced-universe research metrics

If v20 cannot clear that narrower gate, the next meaningful work should be one
of the blocked-source items rather than another generic feature sweep.
