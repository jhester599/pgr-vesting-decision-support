# Review 2026-09-25, step 6 — decision and tax layer, timing, ops (WP8 + WP10)

Findings F19, F20 (the ACTIONABLE mapping), F23, F24 and F26 (email gating,
mode fallback, concurrency, bootstraps, EDGAR User-Agent, CI network) in
[`REPO_REVIEW_2026-09-25.md`](REPO_REVIEW_2026-09-25.md). The read-only dry
run (F14) landed in step 1, the Monte Carlo volatility (F19) and TA features
(F24) in step 4, and the missing-CPCV fail-closed rule (F20) in step 5.

## 1. Live ACTIONABLE mapping (F20)

**Backtest of the exact live function.** `src/models/live_policy_backtest.py`
replays, at every OOS date of the realised-only panel
(`build_prequential_panel`), the live per-benchmark signal
(`classify_benchmark_signal`, now shared with `get_ensemble_signals`), the
live quality-weighted consensus (`build_shadow_consensus_table` with the
configured score column and lambda), the equal-weight IC, and
`decision_rendering.sell_pct_from_consensus`. Per-benchmark IC and quality
weights use only rows realised by that date. The mapping is scored as if the
gate had passed every month, because it is only used in ACTIONABLE months. A
decision keeps `1 − sell %` of the vesting tranche in PGR; its outcome is the
equal-weight mean relative return of that date.

The panel is the production ensemble's record as of 2026-09-21, exported
read-only from a DB copy (`scripts/export_oos_panel.py`; DB sha256 unchanged)
and committed as `tests/fixtures/live_mapping_oos_panel_2026-09-21.csv`:
186 dates (2010-09-30 → 2026-02-27) × 8 benchmarks.

| Policy (186 decisions) | Mean relative return kept |
|---|---|
| Always hold | +7.28 % |
| **New live mapping** | **+3.81 %** |
| Always sell 50 % | +3.64 % |
| Old live mapping | +3.53 % |

| Consensus | Old sell % | New sell % | n | Mean realised | Mean uplift vs 50 % (new) |
|---|---|---|---|---|---|
| NEUTRAL | 50 % | 50 % | 78 | +6.3 % | 0 |
| OUTPERFORM, forecast > 15 % | 25 % | 25 % | 10 | +15.6 % | +3.9 pp |
| OUTPERFORM, 5–15 % | 50 % | 50 % | 75 | +6.4 % | 0 |
| OUTPERFORM, ≤ 5 % | **75 %** | **50 %** | 22 | +9.5 % | 0 (was −2.4 pp) |
| UNDERPERFORM | 100 % | 100 % | 1 | +14.4 % | −7.2 pp |

**Change.** OUTPERFORM never sells more than the 50 % default: > 15 % → 25 %,
otherwise 50 %. A non-finite IC maps to 50 % (fail closed). Uplift over
always-50 %: **−0.11 pp → +0.17 pp** per decision. The review's figures
(+3.53 % / +4.14 % / +8.28 % over 168 dates) were on the pre-step-2 data.

The UNDERPERFORM → 100 % cell rests on one decision, which lost. It is kept:
one observation is not evidence either way, and the brief asked only that a
bullish signal never sell more than the default. The policy-regression test
fails if the mapping as a whole stops beating always-50 %.

The monthly report's "Decision Policy Backtest" now includes a "Live
ACTIONABLE mapping" row (it only scored `tiered_25_50_100` and friends).
The unused duplicate `monthly_decision._sell_pct_from_consensus` is gone.

No month in 2026 reaches ACTIONABLE (step 5), so this changes no committed
recommendation.

## 2. Tax (F19)

| Rule | Before | After |
|---|---|---|
| Breakeven | `+(S − L)/(1 − L)` = +21.25 %, read as the return needed to hold | `−g (S − L)/(1 − L)` on PGR's **absolute** return, g = (P − B)/P: holding to LTCG wins unless PGR falls more than 21.25 % (all-gain lot) |
| LTCG | `days > 365`; hold date vest + 366 days | long-term iff sold after `vest + relativedelta(years=1)`: first LTCG day is the day after the anniversary (2027-07-17 vest → 2028-07-18) |
| Wash sale | none; HOLD_FOR_LOSS sold at vest + 180 days (1–6 days from the next vest) | loss sale at vest + 6 months, moved past the ±30-day window of every scheduled vest; `optimize_sale` marks a loss lot inside the window `WASH_SALE` (loss disallowed, tax 0) and sells it last |
| Lot order | by total gain/loss | by gain/loss **per share** |
| Unvested lots | counted as held | dropped (`load_position_lots(as_of=)`, `optimize_sale`, `compute_position_summary`, holdings guidance, email) |
| Scenario choice | argmax of probability × proceeds (SELL_NOW on a +30 % forecast) | argmax of expected after-tax proceeds |
| Report text | relative forecast compared with the breakeven; "capital-loss harvesting 37 %" on a negative relative forecast | breakeven stated as a PGR price fall; the relative forecast is labelled relative, not a price or a tax loss; wash-sale warning |
| Scenario / MC drift | relative forecast × 2 used as PGR's price return | `config.TAX_SCENARIO_PGR_ANNUAL_RETURN` (default 0 %, i.e. no absolute view) |

The breakeven compares cash at the LTCG date and ignores what sale proceeds
earn meanwhile; the report says so.

## 3. Shadow layer (F24)

- **Path B** scores the decision row (`X_current`, the as-of feature row), not
  `iloc[-1]` of the target-joined frame (the last month whose outcome was
  known). The estimator is `make_path_b_model()`: median impute →
  `StandardScaler` → L2 logistic (C = 0.5), used for the final fit and inside
  every `TimeSeriesSplit` fold (gap 8), so the scaler never sees test rows.
- **Maturity.** `attach_matured_classifier_outcomes(..., as_of=)` recomputes
  `mature_on_date` (`forward_window_end` of the anchor) and
  `is_horizon_mature` on every call, attaches outcomes only to matured rows,
  and clears them on rows not matured by `as_of` (back-dated runs). Before,
  the flag was set once at write time (always False), so "Matured
  observations" stayed 0.
- **Direction-aware veto.** The `veto_regression_sell` overlay vetoes a
  sell-leaning ACTIONABLE month (> 50 %) when P(actionable sell) is below the
  threshold, and a hold-leaning one (< 50 %) when it is at or above it. It
  used to veto every ACTIONABLE month below the threshold, pushing bullish
  months up to 50 %. "Aligned" now reads the live sell % the same way.

## 4. Timing (F23, and the as-of part of F26)

Each EDGAR row (monthly 8-K and quarterly XBRL) enters the feature matrix on
the first business month-end on or after its `filing_date`
(`edgar_availability_dates`, `place_edgar_rows_by_filing_date`); the fixed
2-month lag remains only for a row without a filing date. The committed
tables: 265 monthly rows filed 9–29 days after month end (all now enter one
month earlier), 73 quarterly rows. The fixed lag had placed some 10-Ks before
they were filed (FY2024: filed 2025-03-03, used 2025-02-28), a look-ahead
that is now gone. `roe` (quarterly) is not a live input.

The as-of date is never later than today: from the 20th it is the last
business day on or before the 20th (a Saturday 20th gives Friday the 19th for
the 20th/21st/22nd runs alike; it used to give Monday the 22nd), and an
explicit `--as-of` in the future raises.

**Live effect (September 2026).** See section 7.

## 5. Ops (F26)

- `RECOMMENDATION_LAYER_MODE` outside `live_only` / `live_with_shadow` /
  `shadow_promoted` raises before any work (it fell back to the retired
  `shadow_promoted`).
- `monthly_decision.py` writes `generated=true` to `$GITHUB_OUTPUT` only after
  a new production report; verify, charts, commit and email are gated on it.
  The 21st/22nd fallback runs exit with `generated=false`.
- All eight DB-writing workflows share the concurrency group `db-writer`.
  `monthly_decision.yml` runs on completion of the 8-K fetch
  (`workflow_run`), with 21st/22nd fallback crons; the 20th cron is gone.
- `initial_fetch_prices.yml`, `initial_fetch_dividends.yml` and
  `post_initial_bootstrap.yml` are dispatch-only (their yearly crons would
  fire again in March 2027).
- `EDGAR_USER_AGENT` is set in `monthly_8k_fetch.yml` and
  `weekly_data_fetch.yml`. `config.get_edgar_user_agent()` raises
  `EdgarUserAgentError` when it is unset, blank, the old placeholder, or has
  no e-mail; the weekly job re-raises it instead of logging and continuing.
- CI smoke runs go through `scripts/ci_offline_smoke.py`: sockets refused,
  the SEC submissions index served as an empty canned response, any other URL
  an error. All four smoke commands pass offline (monthly dry run: exit 0,
  DB copy unchanged). `peer_fetch.py --dry-run` wrote to the DB (WAL header,
  schema); it now opens it read-only like the other dry runs.

## 6. Tests: failing before, passing after

New files: `test_live_mapping_wp8.py` (42), `test_tax_wp8.py` (24),
`test_shadow_layer_wp8.py` (15), `test_edgar_filing_timing_wp10.py` (9),
`test_ops_wp8.py` (18). Against unfixed `master` (69889ad) in a clean
worktree: **89 failed, 19 passed** (the passes are positive controls, e.g. an
OUTPERFORM with a > 15 % forecast already sold 25 %). On this branch all 108
pass.

The review's named tests fail on `master` by assertion:

| Finding | Test | Unfixed code |
|---|---|---|
| F20 | `test_bullish_signal_never_sells_more_than_the_default` (24 of 36 cases fail) | OUTPERFORM at ≤ 5 % sells 75 % |
| F20 | `test_live_mapping_policy_regression_uplift_vs_always_50` | on this branch with the harness but the old mapping: "live mapping loses −0.1102 % per decision"; on `master` the harness does not exist |
| F19 | `test_fully_appreciated_lot_breakeven_is_a_21_25_pct_fall` | +0.2125 instead of −0.2125 |
| F19 | `test_ltcg_starts_the_day_after_the_calendar_anniversary` (leap cases) | anniversary across 29 Feb counted as LTCG |
| F19 | `test_hold_to_ltcg_date_across_a_leap_year` | hold date 2028-07-17 (still short-term) |
| F19 | `test_sale_on_the_anniversary_is_taxed_short_term` | taxed as LTCG |
| F19 | `test_loss_lot_inside_the_vest_window_is_a_wash_sale` | loss taken inside the window |
| F19 | `test_hold_for_loss_sale_date_clears_the_next_vest_window` | 2027-07-18, a day after the vest |
| F19 | `test_stcg_lots_are_ordered_by_gain_per_share`, `..._loss_per_share` | total-gain order |
| F19 | `test_unvested_lot_cannot_be_sold`, `test_position_summary_excludes_unvested_lots` | unvested lot sold / counted |
| F19 | `test_positive_forecast_recommends_holding_to_ltcg` | SELL_NOW on +30 % |
| F24 | `test_matured_rows_are_recomputed_and_get_outcomes` | `[False, False]` forever |
| F24 | `test_veto_does_not_penalise_a_hold_leaning_month_...` | bullish month vetoed to 50 % |
| F23 | `test_first_feature_row_with_an_edgar_row_is_on_or_after_filing_and_within_a_month` | filed 2015-02-20, first used 2015-03-31 (39 days) |
| F26 | `test_unknown_recommendation_layer_mode_fails_fast` | falls back and runs |
| F26 | `test_bootstrap_workflows_are_dispatch_only`, `test_monthly_email_charts_and_commit_are_gated_on_generated`, `test_every_workflow_that_commits_the_db_shares_one_concurrency_group` | yearly crons; ungated steps; separate groups |

The remaining failures on `master` are the new APIs (`X_current`,
`as_of=`, `gain_fraction=`, `edgar_availability_dates`,
`ci_offline_smoke`, `_validate_layer_mode`, `_write_step_output`).

Existing tests changed because they encoded the old behaviour:

- `test_three_scenario_tax.py`: breakeven sign (−21.25 %), the stored
  breakeven is the lot's (g-scaled), and the recommendation is the highest
  expected proceeds, not probability × proceeds.
- `test_capital_gains.py`: the sell date moved from 2026-01-15 (four days
  before the January vest, so the loss lot is a wash sale) to 2026-04-15.
- `test_monthly_report_tax.py`: the breakeven row is −21.25 %; the
  "EXCEEDS breakeven" verdict for a +30 % relative forecast and the
  "capital-loss harvesting" note are gone.
- `test_monthly_decision_as_of_date.py`: a Saturday 20th gives Friday the
  19th (was Monday the 22nd, two days in the future).
- `test_edgar_client.py`: a missing User-Agent raises (was the placeholder).
- `test_classification_shadow.py`, `test_path_b_classifier.py`: the new
  `live_sell_pct` / `X_current` arguments.
- `test_v813_recommendation_mode.py`: the scenario note's wording.

## 7. September 2026 before and after

SEPTEMBER_REPLAY_PLACEHOLDER

## Judgement calls

- **Mapping.** The minimal change the brief asks for (bullish never above
  50 %). Tuning the other cells on the same 186 dates would be in-sample
  selection (F13).
- **Absolute return for the tax scenarios.** The models forecast relative
  returns only. Rather than invent a PGR price forecast, the scenarios and the
  Monte Carlo use a configurable absolute drift (default 0 %), and the
  breakeven is reported as the fall that would make selling now win.
- **Wash sale.** A loss inside the window is treated as fully disallowed (the
  replacement shares from a vest usually exceed the lot sold). The loss is
  deferred into the new shares' basis, not lost; the optimiser sells such
  lots last so they can be harvested outside the window.
- **Hold-leaning veto.** The classifier estimates P(actionable sell), so a
  high value contradicts holding more than the default; the veto now blocks
  that case too.
- **EDGAR User-Agent** in the workflows is the one the owner uses for EDGAR
  (the review and steps 3b/4c use it).

## Not done here (WP9 / later)

- `initial_fetch.py` ignores `--force`; `peer_bootstrap.yml` queries a
  nonexistent `price_date`; `post_initial_bootstrap.yml` runs
  `git add results/ || true`; `ci.yml` has no `permissions:` block; actions are
  pinned by tag.
- GitHub keeps one pending run per concurrency group, so a third queued
  DB writer cancels the waiting one (documented in `docs/workflows.md`).
- The TA shadow ledger's `is_horizon_mature` is still set at write time
  (only the classifier ledger has an outcome attacher).
- `rebalancer._check_stcg_boundary` still counts 365 days for its zone.
- Research scripts still use the fixed EDGAR lag (WP11).
