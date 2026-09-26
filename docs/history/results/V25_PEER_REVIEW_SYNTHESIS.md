# V25 Peer Review Synthesis

Created: 2026-04-05

## Summary

Two external repo reviews were added on 2026-04-05:

- [chatgpt_repo_peerreview_20260404-1.md](/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/docs/history/repo-peer-reviews/2026-04-05/chatgpt_repo_peerreview_20260404-1.md)
- [claude_repo_peerreview_20260404.md](/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/docs/history/repo-peer-reviews/2026-04-05/claude_repo_peerreview_20260404.md)

The reports point in the same high-level direction:

- stop broad, unconstrained model-search work for now
- fix correctness and evaluation-integrity issues first
- keep the diversification-first recommendation layer as the production anchor
- treat any continued prediction research as narrow, pre-registered, and
  strictly bounded

## Where The Reports Agree

- Time-index alignment is still the highest-value technical issue.
- Validation integrity matters more than another feature or model sweep.
- Silent failure modes are a bigger near-term risk than lack of model
  complexity.
- The recommendation layer is the most defensible production value today.
- Future research should be smaller, more disciplined, and easier to stop.

## Current Repo Status Versus The Reviews

### Still Open And High Priority

- `ME` / `BME` monthly-index inconsistency is still present.
  - `src/processing/feature_engineering.py` still mixes `BME` month-end logic
    for core market-derived series with several `ME` resamples and
    `MonthEnd(0)` normalizations for EDGAR- and benchmark-derived series.
- Inner CV leakage risk is still open.
  - `src/models/regularized_models.py` still uses `TimeSeriesSplit(n_splits=3)`
    without a `gap` in `_make_inner_cv()` and in the Ridge path.
- The WFO minimum-length guard still does not include the gap.
  - `src/models/wfo_engine.py` still checks only `TRAIN + TEST`, not
    `TRAIN + GAP + TEST`.
- CPCV prediction provenance still appears incorrect.
  - `src/models/wfo_engine.py` still writes split predictions into a single
    `all_y_hat` array and then reuses that array when reconstructing
    recombined paths.

### Likely Open And Needs A Direct Audit

- The ROE name mismatch still needs explicit end-to-end verification.
  - `scripts/edgar_8k_fetcher.py` still emits
    `roe_net_income_trailing_12m`.
  - `src/database/db_client.py` still stores `roe_net_income_ttm`.
  - The repo now has a partial mapping comment, but there is not yet a clear
    end-to-end guard proving the live path cannot silently drop the value.

### Already Partially Addressed

- The recommendation-layer simplification advocated by the Claude report is
  already partly in production.
  - The active recommendation layer is the simpler diversification-first path.
  - `ensemble_ridge_gbt_v18` is now the visible cross-check candidate from
    `v22`.
- Historical validation was already tightened in `v21-v24`.
  - The earlier narrow-window interpretation has been corrected.
  - The visible cross-check promotion now holds over the longer stitched
    history used in `v23`.

### Not The Right Immediate Next Step

- Replacing the current production path with a fully non-ML recommendation-only
  system is not the next code change.
  - That is now a governance question, not an urgent engineering task.
- Replacing `VOO` with `VTI` is not supported by the latest evidence.
  - `v24` showed that `VTI` adds raw history but does not improve the leading
    candidate enough to justify replacing `VOO`.
- A broad new feature search is not justified before the correctness issues are
  fixed.

## Recommended Enhancement Direction

The most valuable next cycle is:

1. fix monthly-index normalization
2. fix validation / diagnostic integrity
3. add explicit silent-failure guards
4. rerun the affected `v20-v24` comparisons on the corrected foundation
5. only then decide whether another prediction-research cycle is warranted

That means the next cycle should be an accuracy-through-correctness cycle, not
an accuracy-through-complexity cycle.

## Practical Next-Step Conclusion

The next enhancement program should be `v25`, with this order:

- `v25.0`: canonical monthly-index normalization
- `v25.1`: WFO + CPCV integrity fixes
- `v25.2`: schema / missingness / freshness guards
- `v25.3`: rerun the historically important studies affected by those fixes
- `v25.4`: governance closeout on whether prediction research should continue

See the full plan in [codex-v25-plan.md](/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/docs/plans/codex-v25-plan.md).

Execution results are now recorded in [V25_RESULTS_SUMMARY.md](/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/docs/results/V25_RESULTS_SUMMARY.md).
