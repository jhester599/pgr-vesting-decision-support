# Independent verification of the 2026-09-25 review fixes — 2026-09-26

- **Scope:** every finding (F01–F35) of [`REPO_REVIEW_2026-09-25.md`](REPO_REVIEW_2026-09-25.md) and the work merged for it: CHANGELOG v171–v185 and PRs [#120](https://github.com/jhester599/pgr-vesting-decision-support/pull/120)–[#133](https://github.com/jhester599/pgr-vesting-decision-support/pull/133) (steps 1, 2, 3a, 3b, 4, 4b, 4c, 5, 6, 7, 9, 10, 11 and 12).
- **Commit verified:** `master` at `aae0be8` (Merge PR #133).
- **Committed DB:** `data/pgr_financials.db`, sha256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`. It was unchanged at the start and the end of this session. The review-time DB (`9887288`) is `e7531a34…8b38`.
- **Verifier:** I did not write any of the fixes. I treated every claim in the PRs, commit messages and CHANGELOG as unverified, and reproduced each one myself.
- **Safety:**
  - All work ran in scratch clones outside the repository, with a Python 3.11.15 venv (pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, xgboost 3.2.0).
  - DB reads used `?mode=ro&immutable=1` connections or copies.
  - Every monthly dry run ran under a wrapper that refuses all socket connections, with `--skip-fred` (Appendix A.4). No fetcher and no e-mail step ran.
  - No Alpha Vantage, FRED or EDGAR call was made.
  - The only file this PR adds besides the research plan is this report.

Plain-language terms used below:

- **OOS R² (out-of-sample R²):** how much smaller the forecast errors are than those of a naive guess (the average of the outcomes already known when the forecast is made). 0 % means no better than the naive guess; the production gate needs ≥ 2 %.
- **IC (information coefficient):** the rank correlation between forecasts and outcomes, from −1 to +1. The gate needs ≥ 0.07 on the equal-weight average across the 8 benchmark funds.
- **Hit rate / base rate:** hit rate is the share of correct "PGR beats / trails the fund" calls. The base rate is the hit rate of always saying "PGR beats the fund": 68–70 % over this history.
- **PT p-value:** Pesaran–Timmermann test that the up/down calls beat chance given the base rate; the gate needs p < 0.05.
- **ECE (expected calibration error):** the average gap between stated probabilities and observed frequencies. "Prequential" means each month is scored by a model fitted only on data available before it.
- **DK p-value:** a significance test that treats all benchmarks on one date as one observation (Driscoll–Kraay), so 8 correlated funds are not counted as 8 independent facts.

## 1. Summary for the owner

**Bottom line.** The fixes are real and they hold up.

- Of the 35 problems in the September 25 review:
  - 25 are fully fixed;
  - 8 are partly fixed (the rest is small or deferred);
  - 2 are not addressed:
    - F25 (research methods) was deferred to the research phase by the review's own plan;
    - F31 (unused model utilities) was never assigned to a work package;
  - none got worse.
- The full test suite passes, and each fix is protected by a test that fails if the fix is undone. I checked this for every step.
- The reorganisation of the repository changed no numbers: the September 2026 report comes out byte-for-byte identical before and after it.

**What was wrong.** Four kinds of problem:

1. **Bad inputs.**
   - The "daily" price table holds one price per week and ignores stock splits. So the "12-month momentum" input really measured about 58 months, and a split looked like a 75 % crash.
   - Two ETF splits (VOO 2013, VGT 2026) were missing, which put some past outcomes off by more than 100 percentage points.
   - Economic data such as interest rates and the VIX "fear index" was lagged twice, so it reached the model one to two months later than intended.
   - Dividends stopped updating in March.
   - Several numbers read from Progressive's monthly SEC filings were misread: losses stored as profits, percentages stored as fractions, and two months missing.
2. **Broken scorekeeping.**
   - The check "does the model beat a naive guess?" compared it with a guess that already knew part of the answer.
   - A second check (CPCV) could never pass.
   - Together they forced "defer to the default rule" every month, whatever the model said.
3. **Over-optimistic health reports.** Several settings were tuned on the same history that was then used to grade them.
4. **Operational risks.**
   - A "dry run" (meant as a test) silently rewrote the real database and reports.
   - The tax helper had a leap-year error and a sign error.

**What changed.**

- All of the above was repaired and re-tested.
- The database was rebuilt through scripted, reviewable migrations and repair scripts, not hand edits.
- The code was reorganised into an installable package, with one folder per research study.

**What the model and recommendation do differently now, and why.** I re-ran every monthly decision from February to September 2026 on today's code (section 9).

- **The action is unchanged.** Every month is still "DEFER-TO-TAX-DEFAULT: sell 50 % of the vest", as in the reports you received. The reason is different and honest now:
  - On the fair accuracy measures the model now looks modestly better than a naive guess. R² is +1 % to +6 % (it was reported as −0.1 % to −6 %). It passes the R² and IC gates in 6 of 8 months; July is marginal on IC, and August on both.
  - It fails the check that matters most for a sell/hold call: its up/down calls are right 60–64 % of the time, while simply assuming "PGR wins" is right 68–70 % of the time (PT p 0.36–0.62).
  - A model that cannot beat that rule should not move your vest decision, so it defers.
- **The model's lean moved in 4 of 8 months** (between NEUTRAL and UNDERPERFORM). It did not change the action.
- **An experimental "Path B" classifier now leans towards selling:** a 73 % chance that selling beats holding in September, where the old report said 49 %. It is a shadow signal, it is not validated, and by design it cannot change the recommendation.

**What risks remain.**

1. **ETF dividends are still missing after March 2026** for 20 of the 26 tickers tracked (funds and peer stocks), including 6 of the 8 comparison funds.
   - The fix is in the code, but the first automatic dividend refresh has not run yet. It is scheduled for Wednesday 2026-09-30 and uses the Alpha Vantage key stored in GitHub.
   - Until it runs, the latest training outcomes flatter PGR by up to 1.8 percentage points (bond funds) and about 0.6 points for stocks.
   - The monthly report's data-freshness table does not show this.
2. **The "80 % range" in the report is far too narrow.** It contains the actual outcome only about 43 % of the time.
3. **The report's "chance PGR outperforms" figure often contradicts the forecast.**
   - In September, VWO is forecast at −2.1 % yet shown with an 83 % chance of PGR outperforming.
   - The figure mostly reflects the 68 % base rate, not the model.
4. **The model's inputs and settings were chosen by earlier research (about 150 studies, March–April 2026) that ran on the broken data and scoring.**
   - None of that research has been redone yet; `docs/research/RERUN_PLAN_v200_claude.md` is the plan to redo it.
   - The research code's own R² calculation still has the old look-ahead problem for 6-month targets (new issue N2).
5. **Smaller items:**
   - a suspected 2024 change in Progressive's monthly reporting calendar that makes the premium-growth input noisy;
   - peer-company price and dividend quirks;
   - a few outdated docs;
   - 197 files still use an old import workaround.

**What needs a decision from you.**

1. **Make sure the dividend refresh has run before the 20 October decision.**
   - It is scheduled for Wednesday 30 September. Afterwards, check that the "Check price, split and dividend integrity" step of that run passed.
   - If it did not run, or still reports stale dividends, dispatch it: GitHub → Actions → "Weekly Data Accumulation" → Run workflow with `dividend_refresh: true`.
   - Otherwise October runs on stale ETF dividends.
2. **Approve (or trim) the research re-run plan v200–v210.**
   - It reserves the most recent 24 months as a one-time final test.
   - It changes nothing live; any promotion is a separate PR under `docs/model-governance.md`.
3. **Decide whether the monthly e-mail should keep showing the Path B sell probability, the 80 % ranges and the "chance PGR outperforms" figure** while they are unvalidated or miscalibrated. My suggestion: label them "experimental, not used", or hide them until research step v207 re-validates them.
4. **Optional:** remove or annotate the 6 old `[DRY RUN]` rows that are still in `artifacts/monthly_decisions/decision_log.md`.

## 2. Method

- **Location mapping.** The review cites the pre-restructure layout. For each finding I traced the cited code to its current file and function (`git log --follow`, and by function name for the `scripts/monthly_decision.py` split, which git does not detect as a rename). The current location is in the table in section 3; Appendix B maps the old files to the new ones.
- **Symptom checks.** Each check re-runs the review's evidence against the review-time state and today's state:
  - **Data:** the review-time DB (`9887288`) and today's DB, compared read-only.
  - **Features:** the production feature builder at both commits, each on its own DB (Appendix A.2), checked against my own hand calculation.
  - **Validation logic:** the same synthetic inputs through the review-time and today's code (Appendix A.3).
- **Tests.** The full suite (section 4.1); one revert per step in a scratch clone, with the named test run before and after (section 4.2); and the 24-mutation study (section 4.3).
- **Decisions.** Read-only dry runs for every committed as-of date, network blocked (section 9).

## 3. Verification table

PR links: [#120](https://github.com/jhester599/pgr-vesting-decision-support/pull/120) step 1, [#121](https://github.com/jhester599/pgr-vesting-decision-support/pull/121) step 2, [#122](https://github.com/jhester599/pgr-vesting-decision-support/pull/122) step 3a, [#123](https://github.com/jhester599/pgr-vesting-decision-support/pull/123) step 3b, [#124](https://github.com/jhester599/pgr-vesting-decision-support/pull/124) step 4, [#125](https://github.com/jhester599/pgr-vesting-decision-support/pull/125) step 4b, [#126](https://github.com/jhester599/pgr-vesting-decision-support/pull/126) step 4c, [#127](https://github.com/jhester599/pgr-vesting-decision-support/pull/127) step 5, [#128](https://github.com/jhester599/pgr-vesting-decision-support/pull/128) step 6, [#129](https://github.com/jhester599/pgr-vesting-decision-support/pull/129) step 7, [#130](https://github.com/jhester599/pgr-vesting-decision-support/pull/130) step 9, [#131](https://github.com/jhester599/pgr-vesting-decision-support/pull/131) step 10, [#132](https://github.com/jhester599/pgr-vesting-decision-support/pull/132) step 11, [#133](https://github.com/jhester599/pgr-vesting-decision-support/pull/133) step 12.

`q "<SQL>"` runs a query on a read-only DB (Appendix A.1). "Review DB" is `9887288:data/pgr_financials.db`; "now" is `aae0be8`. Status counts: 25 FIXED, 8 PARTIAL, 2 NOT ADDRESSED, 0 REGRESSED.

| Finding | Status | PR / commit | Where the code is now (review cited) | Evidence command | Result (review → now) |
|---|---|---|---|---|---|
| F01 weekly/unadjusted price features | FIXED | #124 `71854ba` | `src/processing/feature_engineering.py::build_feature_matrix` L366–372 (split-adjusted close), `_MOMENTUM_MONTHS`/`_VOL_WEEKS` L263–283; `src/processing/price_adjustment.py::split_adjusted_close` L37, `assert_weekly_bar_frequency` L146 (was `feature_engineering.py:166-175,253-286,1370-1378`) | A.2: build the matrix, compare with a hand-computed split-adjusted calendar return, 13-week vol ×√52 and 52-week high | `mom_12m` on the 2026-08-31 decision row: +1.281 → −0.115, equal to the hand value −0.115. 2006-05 `vol_63d`: 2.774 → 0.138. Every row since 2001-06 matches the hand values to ≤ 4e-15 (`mom_3m/6m/12m`, `vol_63d`, `high_52w`). Correlation of the old `mom_12m` with the true one: 0.516, as the review found. `daily_prices` keeps its name (the builder now asserts weekly bars). |
| F02 CPCV verdict never passes; CPCV is K-fold | FIXED | #127 `c8e6f66` | `src/models/wfo_engine.py::run_cpcv` L586, `cpcv_path_thresholds` L473, `_recombined_path_members` L559; `config/model.py` `CPCV_EMBARGO_SIZE` L82; `src/reporting/decision_rendering.py::evaluate_quality_gates` (was `wfo_engine.py:521-530,660-687`, `config/model.py:97`, `decision_rendering.py:50,65`) | A.3: `run_cpcv(ridge, 8 folds, 2 test)` on a near-perfect synthetic predictor | Review code: 8 "paths" (really folds), 8/8 positive, verdict **FAIL**. Now: 28 splits, 7 paths, each scoring all 240 rows once, 7/7, verdict **GOOD**. CPCV no longer gates ("diagnostic only"; Sept replay 4/7 MARGINAL). |
| F03 VOO 2013 split missing | FIXED | #121 `fd432f5`, `43e82d5` | `config/splits.py::KNOWN_SPLITS`; `src/processing/price_integrity.py`; `scripts/rebuild_relative_returns.py` (was `scripts/weekly_fetch.py:55-73`, `scripts/apply_split_history.py:48-65`, `multi_total_return.py:111-127`) | `q "select date, round(benchmark_return,3) from monthly_relative_returns where benchmark='VOO' and target_horizon=6 and date between '2013-04-01' and '2013-09-30'"` | Review DB: +1.13 … +1.32 (6 rows). Now: +0.065 … +0.160. `split_history` has `(VOO, 2013-10-24, 0.5)`. The jump guard finds 0 unexplained jumps (section 5). |
| F04 OOS-R² naive includes the target | FIXED (production) | #127 `c8e6f66` | `src/reporting/backtest_report.py::compute_oos_r_squared` L35; `src/models/prequential.py::prevailing_mean_forecast` L91 (was `backtest_report.py:63-65`) | A.3: oracle forecaster (forecast = true conditional mean) on overlapping 6-month targets | Review code: R² **−9.2 %**. Now: **+1.3 %**. Committed September R² −1.13 % → replay +2.86 %. The research harness still has the problem: new issue N2. |
| F05 VGT 2026 split missing | FIXED | #121 | as F03 | `q "select date, target_horizon, round(relative_return,3) from monthly_relative_returns where benchmark='VGT' and ((target_horizon=6 and date between '2025-10-01' and '2026-02-28') or (target_horizon=12 and date between '2025-04-01' and '2025-08-31'))"` | 6M: +0.77 … +0.91 → −0.01 … −0.40. 12M: +0.51 … +0.77 → −0.42 … −0.89. All 10 labels change sign. `(VGT, 2026-04-21, 8.0)` is in `split_history`. |
| F06 FRED lagged twice, duplicate months | FIXED | #122 `f72e0bd` | `src/ingestion/fred_loader.py::fetch_all_fred_macro` (`apply_publication_lags=False` L210), `to_monthly_observations` L172; `feature_engineering._apply_fred_lags` L109; `db_client.fred_month_label` L379; migration 005 (was `fred_loader.py:131,190-195`, `feature_engineering.py:102-122,1219-1222`) | `q "select series_id, month_end, value from fred_macro_monthly where series_id='VIXCLS' and month_end between '2020-02-01' and '2020-05-31'"`; A.2 lag identity | Stored VIXCLS for 2020-03 is 53.54 (the March close); the review DB had it under 2020-04-30. `vix` feature on the 2020-04-30 row: 40.11 (February, 2 months stale) → 53.54 (March, lag 1). `feature(M) = raw(M − lag)` holds on all 322 rows for `vix`, `nfci`, `yield_slope` and `credit_spread_hy` (0 mismatches). Duplicate (series, month) pairs: 576 → 0. |
| F07 stale PGR FRED series; silent NaN live feature | FIXED | #120 `b66bc01`, #122 | `fred_loader.production_fred_series` L191; `db_client.check_data_freshness` L518 (one row per ticker and per live series); `pgr_vds/decision/signal_generation.py::find_nan_live_features` L212 (was `weekly_fetch.py:105`, `monthly_decision.py:309`, `wfo_engine.py:409-416`, `db_client.py:304-387`) | `q "select series_id, max(month_end) from fred_macro_monthly group by series_id"`; A.2 NaN scan of live-model columns | Last stored month for `PCU5241265241261`: 2026-02 → 2026-08. `CUSR0000SETA02`/`SAM2`: 2026-03 → 2026-08. `TRFVOLUSM227NFWA`: 2026-03 → 2026-07. `rate_adequacy_gap_yoy` was NaN in every row 2026-04 … 09; now no live feature is NaN in any of them. The freshness check reports each live series against the month the decision row needs. |
| F08 dividends stale since 2026-03-26 | **PARTIAL** | #121 `fd432f5` | `src/ingestion/multi_dividend_loader.py::fetch_for_tickers` L193 (sleep + advisory retry); `db_client.check_dividend_freshness` L630; `scripts/weekly_fetch.py --dividend-refresh`; Wednesday cron in `weekly_data_fetch.yml`; `scripts/check_data_integrity.py` (was `weekly_fetch.py:162-175`, `multi_dividend_loader.py:208-224`, `initial_fetch_dividends.yml:19`) | `check_dividend_freshness(conn)` on a DB copy (section 5); A.7 bias estimate | The code is fixed and tested, and PGR is current (the July and October dividends arrived with the 2026-09-26 weekly run). **The data is still stale:** 20 of 26 tickers are STALE, including VOO, VXUS, VWO, VMBS, BND and VDE. No Wednesday refresh has run since the fix merged. The newest 6M targets flatter PGR by up to 1.82 pp (VMBS), 1.63 (BND) and about 0.6 (VOO, VXUS, VDE). The monthly report's freshness table does not show dividends (new issue N1). |
| F09 quarterly XBRL: Q4 annual, ROE 4× | FIXED | #123 `b71fdc8` | `src/ingestion/edgar_client.py` (Q4 = FY − 9M, TTM NI / average of 5 equities, earliest filed); migration 007 (was `edgar_client.py:197-220,365-371`) | `q "select period_end, roe, eps, filing_date from pgr_fundamentals_quarterly where period_end like '2018%'"` | Median Q4 ROE 0.736 → 0.185. 2018-12 ROE 0.967 → 0.245 (PGR reported 24.7 %). Q4-2018 EPS 4.45 → 0.44 (= 4.45 − 4.01). The always-NULL `pe_ratio`/`pb_ratio` columns are dropped and `filing_date` is added. 74 → 73 rows (2007-12 annual comparative removed). |
| F10 book yield % vs fraction | FIXED | #120, migration 004 | `pgr_vds/ingestion/edgar_monthly/parse.py` (no `/100`); `src/database/migrations/004_investment_book_yield_percent.py` (was `scripts/edgar_8k_fetcher.py:816-817`) | `q "select min(investment_book_yield), max(investment_book_yield), sum(investment_book_yield<1), count(investment_book_yield) from pgr_edgar_monthly where month_end>='2023-01-01'"` | Values below 1 since 2023: 33 of 40 → 0 of 44. Range 0.03–3.9 → 2.6–4.3. |
| F11 PIF growth: leap-year key, 2024 definition | FIXED | #123 | `pgr_vds/ingestion/edgar_monthly/derive.py::_prior_year_key` L19; `src/processing/pgr_edgar_derived.py::pif_sum` L100, `yoy_growth_by_period` L111 (was `scripts/edgar_8k_fetcher.py:1753-1765,1813-1823`, `feature_engineering.py:325-353`) | `q "select month_end, pif_total, round(pif_growth_yoy,3) from pgr_edgar_monthly where substr(month_end,6,2)='02' and substr(month_end,1,4) in ('2009','2013','2017','2021','2025')"` | Feb 2009/13/17/21/25: NaN → 0.038 / 0.020 / 0.060 / 0.102 / 0.182. 2024-12 … 2025-03: 0.314–0.317 → 0.182–0.185. Largest month-on-month `pif_total` jump since 2023: 13.5 % → 2.0 %. |
| F12 parenthesised negatives lose their sign | FIXED | #123 | `parse.py::_parse_number` L212, `_row_numbers` L236 (was `scripts/edgar_8k_fetcher.py:472-485`) | `q "select month_end, net_income, eps_diluted, income_before_income_taxes, total_net_realized_gains from pgr_edgar_monthly where month_end in ('2017-08-31','2018-10-31')"` | Aug-2017 NI +16.8 → −16.8, EPS +0.03 → −0.03, pretax +49.5 → −49.5, realized gains +11.7 → −11.7. Oct-2018 NI +31.7 → −31.7. Monthly NI summed per quarter equals XBRL NI in 73 of 73 quarters (max gap $0.5M). |
| F13 health inflated by in-sample choices | **PARTIAL** | #127 | `src/models/prequential.py` (shrinkage α and Ridge/GBT weights from realised rows); `src/models/robust_inference.py` (DK p); `decision_rendering.evaluate_quality_gates` (equal-weight IC, PT test); `pgr_vds/decision/health.py` (was `config/model.py:46`, `evaluation.py:451-495`, `consensus_shadow.py:45`, `monthly_decision.py:532-571,609-721`, `forecast_diagnostics.py:103-105`) | A.3 (always-same-sign predictor at a 68 % base rate); September replay `model_health` block | The review code rated an always-"PGR wins" predictor **ACTIONABLE / sell 100 %**; now **DEFER / 50 %**. Replay: the gate uses the equal-weight IC (0.0762, not 0.0822 quality-weighted), prequential ECE 0.154 (CI 0.106–0.233), trailing conformal coverage 42.7 %, pooled IC DK p 0.125, α re-chosen prequentially. **Left:** the quality weights that set the live direction are still fitted on the full OOS record. The v72–v78 choice of quality weighting, and v38's shrink-toward-zero rule, were selected in-sample and have not been re-run (step 5 deferred both to research). |
| F14 `--dry-run` mutates DB and artifacts | FIXED | #120 `b66bc01` | `pgr_vds/decision/pipeline.py::main` (read-only connection when `dry_run`); `pgr_vds/decision/artifacts.py::dry_run_output_dir`; `scripts/weekly_fetch.py` (was `monthly_decision.py:731-779,3039,3180,3402-3683,3576`) | A.4: `python cli/monthly_decision.py --dry-run --as-of <d> --skip-fred` in a scratch clone; sha256 of the DB and `git status` before and after | **Review code** (`9887288`, as-of 2026-04-02): DB `e7531a34…` → `0e8cbb37…`, 14 tracked files rewritten, the 2026-04 `aggregate_oos_r2` overwritten (−0.02626 → −0.03428, as the review found), a `model_retrain_log` row added. **Now:** 12 dry runs, all with DB `7c68efbd…` unchanged and 0 tracked changes: 8 months at `aae0be8`, 1 at `b609af6`, 3 at `69889ad`. Six old `[DRY RUN]` rows are still in the committed `decision_log.md`. |
| F15 per-share basis across the 2006 split | FIXED (production) | #124 | `src/processing/price_adjustment.py::restate_to_latest_share_basis` L249; `feature_engineering.build_feature_matrix_from_db` (was `feature_engineering.py:516-528,745-763,1191-1205,1227-1366`) | A.2: rows 2006-03 … 2007-08 | `book_value_per_share_growth_yoy` 2006-07 … 2007-06: −0.70 → +0.17 … +0.22. `pb_ratio` 2006-05/06: 0.842/0.782 → 3.305/3.120. Max \|Δlog P/B\| 1.39 → 0.21. `pgr_vs_kie_6m` 2006-05: −0.744 → −0.072. Research x-series targets are still on raw values (F25). |
| F16 YoY spans 13 months; 2024 fiscal-month change | **PARTIAL** | #123 | `pgr_edgar_derived.yoy_growth_by_period` L111 (was `scripts/edgar_8k_fetcher.py:2430,2442`) | `q "select month_end, round(npw_growth_yoy,4) from pgr_edgar_monthly where month_end like '2016-05%' or month_end like '2020-04%'"` | 2016-05: −0.152 → +0.1055. 2020-04: +0.272 → +0.0257 (both match EDGAR). **Not investigated:** the suspected 2024 fiscal-calendar change ("not investigated", step 3b report). Live-Ridge `npw_growth_yoy` has a standard deviation of 0.198 in 2024 against 0.076 in 2016–2023 (2024-03: +0.64). |
| F17 two releases missing; pagination unreachable | FIXED | #123 | `pgr_vds/ingestion/edgar_monthly/fetch.py::_fetch_submissions_page` L169 (flat pagination files); `parse.select_monthly_releases` L1849 (9.01-only 8-Ks with EX-99) (was `scripts/edgar_8k_fetcher.py:184-189,224-240`) | `q "select month_end, net_premiums_written, net_income, combined_ratio, book_value_per_share from pgr_edgar_monthly where month_end like '2015-05%' or month_end like '2019-04%'"` | 263 → 265 months; none missing from 2004-08 to 2026-08. 2015-05: NPW 1,581.4, NI 79.4, CR 94.3, BVPS 12.60. 2019-04: 3,669.8, 487.8, 87.4, 20.74. Both equal review Appendix A. |
| F18 equity/debt mis-parsed in 11 rows | FIXED | #123 | `parse.py` (whole-label equity/debt match; BVPS × shares when no equity line) (was the monthly parser field matching) | A.6 identity: `(equity − preferred) / (BVPS × shares) − 1` | Rows with equity = ROE %: 11 → 0. Rows off by more than 6 %: 11 → 0 (max 0.07 %). NULL equity: 3 → 0. |
| F19 tax and Monte Carlo errors | FIXED | #124 (MC vol), #128 `3a05d0c` | `src/tax/capital_gains.py::ltcg_eligible_date` L91, `compute_stcg_ltcg_breakeven` L583, `optimize_sale` L184; `src/tax/monte_carlo.py::estimate_annual_vol_weekly` L161; `pgr_vds/decision/tax_lots.py` (was `capital_gains.py:58,155-157,442-467,524,615,629-639`, `monte_carlo.py:125-145`) | `estimate_annual_vol_weekly(PGR closes ≤ 2026-09-21, splits)`; revert check (section 4.2) | MC annual vol 0.948 (review) → 0.264 (weekly split-adjusted returns × √52). Reverting the LTCG date to vest + 366 days fails 4 hand-computed tests. Breakeven, wash-sale window and lot-order tests pass (full suite). |
| F20 ACTIONABLE mapping; missing CPCV fails open | FIXED | #127, #128 | `decision_rendering.sell_pct_from_consensus` L16, `cpcv_available`; `src/models/live_policy_backtest.py` (was `decision_rendering.py:14-50`, `monthly_decision.py:406-410`) | A.3: CPCV = None with R² 3 %, IC 0.10, hit 68 %; OUTPERFORM with a +3 % forecast | Review code: **ACTIONABLE, sell 100 %** → now **DEFER, 50 %**. OUTPERFORM at +3 %: sell 75 % → 50 %. Reverting the mapping fails 28 policy tests. |
| F21 confidence always LOW | FIXED | #127 | `src/models/calibration.py::confidence_tier_from_probability` L417 (was `multi_benchmark_wfo.py:310-366`) | `signals.csv` of the replays | Committed: every benchmark had `prob_outperform` 0.5, `prediction_std` 0 and tier LOW. Replay: tiers vary (LOW/MODERATE/HIGH, e.g. Sept BND HIGH, GLD MODERATE) and `prediction_std` is gone. The calibrated probability still often disagrees with the forecast sign (new issue N3). |
| F22 target windows, duplicate bars, as-of truncation | FIXED | #121 | `src/processing/total_return.py::forward_window_end` L33; `db_client.upsert_prices(one_bar_per_week)` L707, `dedupe_weekly_price_bars` L806; `feature_engineering.truncate_relative_target_for_asof` L1514 (was `multi_total_return.py:115-127`, `feature_engineering.py:1454-1473`) | A.6 duplicate-week scan; `truncate_relative_target_for_asof(VOO 6M, as_of=2026-03-31)` | Ticker-weeks with two bars: 31 → 0. As-of 2026-03-31 keeps targets up to 2025-09-30 (window ends 2026-03-31 ≤ as-of). Reverting the business-month-end window fails 5 tests. |
| F23 fixed 2-month EDGAR lag | FIXED | #128 | `feature_engineering.edgar_availability_dates` L204, `place_edgar_rows_by_filing_date` L245 (was `config/features.py:237`, `feature_engineering.py:142-162`) | A.2: for every monthly row, find the first feature row carrying its `npw_growth_yoy` | January 2024 (filed 2024-02-14) enters the 2024-02-29 row (was 2024-03-29). 253 of 253 rows first appear at the first business month-end on or after `filing_date`. None earlier, none more than 31 days later. |
| F24 shadow-layer defects | FIXED | #124 (TA), #128 (Path B, maturity, veto) | `src/models/path_b_classifier.py::fit_path_b_classifier` L259; `src/models/classification_monitoring.py::attach_matured_classifier_outcomes` L33; `src/research/v118_utils.py` (direction-aware veto); `src/research/v160_ta_features.py::build_ta_feature_matrix` L218 (was `path_b_classifier.py:286`, `classification_shadow.py:765-803`, `classification_gate_overlay.py:108-112`, `v160_ta_features.py`) | Replay `monthly_summary.json` → `classification_shadow` | Path B now scores the decision row (`feature_anchor_date` 2026-08-31): P(actionable sell) 49.4 % (committed) → 73.2 %. TA: `test_price_features.py::test_ta_features_ignore_splits_on_weekly_bars` (split and pre-adjusted inputs agree; 2006-05 `ta_pgr_natr_63d` < 0.1; the review measured 0.195) passes. |
| F25 research conclusions and methods | **NOT ADDRESSED** (deferred to research, WP11) | — | `research/studies/v40…v59_*/…py`, `src/research/pb_vs_pe.py`, `src/research/x1_targets.py` (were under `results/research/`, `src/research/`) | `grep -rn "RidgeCV(" research/studies \| grep "cv=None"` | 17 call sites in 16 study scripts (v39–v42, v44, v47–v52, v54, v55, v57–v59) still use leave-one-out `RidgeCV(cv=None)`. Production `regularized_models.py:145` uses the inner time-series CV. No lint test forbids it. x-series targets still use raw prices. The pb_vs_pe inference is unchanged. Covered by the re-run plan. |
| F26 CI, workflows, config | **PARTIAL** | #128 (+ #120) | `.github/workflows/*.yml` (`db-writer` group; `monthly_decision.yml` on `workflow_run` of the 8-K fetch; bootstraps dispatch-only; `EDGAR_USER_AGENT` set); `config/api.py::get_edgar_user_agent` L45; `pgr_vds/decision/schedule.py::validate_layer_mode` L56, `resolve_as_of_date` L22; `artifacts.write_step_output` | `grep -n "group:\|cron\|EDGAR_USER_AGENT\|generated" .github/workflows/*.yml` | All five recommended fixes are in place. Every DB-writing workflow uses `group: db-writer`. E-mail is sent only when `generated == 'true'` and the commit succeeded. An unknown mode raises. The as-of date is never later than today. **Left (minor):** `peer_bootstrap.yml:119` still queries the nonexistent `price_date` column. `initial_fetch.py --force` is still ignored (`del force`). `ci.yml` has no `permissions:` block. |
| F27 other data integrity | **PARTIAL** | #122 (FRED part) | FRED: as F06. Peers/proxies: `daily_prices`, `daily_dividends`; `multi_total_return.py:202` (`proxy_fill` hard-coded 0) | `q "select substr(ex_date,1,4), count(*) from daily_dividends where ticker='CB' group by 1"`; FZROX proxy count | FRED duplicates 576 → 0; labels are business month-ends. **Not addressed:** CB is still spliced (7–9 dividends a year until 2015; weekly close 52.75 → 60.57 on 2006-12-29). TRV has 6 dividends in 2004. 895 FZROX VTI-proxy rows. `proxy_fill` is hard-coded. FRED is latest-vintage, with monthly-average GS2/5/10 mixed with month-end T10YIE/VIX. |
| F28 test gaps | FIXED | #121–#130 (step 9: `7032649`, `5bbb058`, `199c65d`) | `tests/` (unit / integration / research), `tests/repo_guard.py`, `scripts/checks/mutation_study.py` | Revert one fix per step (section 4.2); `scripts/checks/mutation_study.py` in scratch clones (section 4.3) | Every step's named test fails when its fix is reverted and passes when restored (17 of 17 reverts). One extra guard is untested (N6). Mutation study: 0 of 18 F28 mutations survive (review: 16 of 18), and 0 of 6 extras. |
| F29 hygiene | **PARTIAL** | #120, #129, #132 | `.gitignore`, `scripts/finalize_db.py`, `pyproject.toml`, `CLAUDE.md`, docs | `git check-ignore -v data/processed/pgr_edgar_cache.csv`; `od -An -tu1 -j18 -N2 data/pgr_financials.db`; `ruff check --select E,W .` | **Fixed:** reference CSVs can be committed; the DB header is in DELETE mode (1,1); no `-qq`; pandas pinned to `>=3.0,<4` with Python ≥ 3.11; CSV cache and data dictionary have 265 rows; `CLAUDE.md` points to `AGENTS.md`. **Left:** `AGENTS.md` still says "Python 3.10+". `docs/operations-runbook.md:144` names `build_db_health_report` (it is `get_db_health_report`). `docs/troubleshooting.md` lists `EMAIL_FROM`/`EMAIL_TO` (the workflows use `PREDICTION_EMAIL_FROM`/`_TO`). The CHANGELOG still skips PRs #116–#118. ROADMAP still says "combined_ratio NULL". `backlog.md` has 11 mojibake sequences. Unused config remains (`EDGAR_BASE_URL`, `EDGAR_PGR_CIK`, `ETF_LAUNCH_DATES`, `ETF_PROXY_MAP`, `BL_USE_BAYESIAN_VARIANCE`, `CUSR0000SETC01`). PEP 8 is not enforced: `ruff --select E,W` gives 1,915 violations at `9887288` and 1,556 now (same ruff 0.13.2 and config). |
| F30 production depends on `results/`; unversioned parquet | **PARTIAL** | #129 `e48b402`, #131 `02e3fd4` | `src/research/binary_classification.py` (was `results/research/v46_classification.py`); `config.V128_BENCHMARK_FEATURE_MAP_PATH`; `src/research/study_paths.py` | `git ls-files results \| grep -c '\.py$'`; `grep -rln "feature_matrix.parquet" research/studies` | 0 `.py` files under `results/`. Production imports nothing from `results/`. Importing `binary_classification` has no side effects (reverting that fails its test). **Left:** 16 x-series scripts still read the unversioned, gitignored `data/processed/feature_matrix.parquet`, and no study has a provenance file (0 of 113). |
| F31 unused or ineffective utilities | **NOT ADDRESSED** | — | `src/models/blp.py`, `src/models/drift_monitor.py` L45–50, `feature_engineering.apply_fracdiff` L1654 | `BLPModel().fit(P, y)` vs `.fit(P, 1 − y)` on synthetic data | Identical parameters for y and 1 − y (a 6.973, b 6.875, same weights and log-likelihood). The fit still ignores outcomes. The drift monitor still averages full-history ICs. fracdiff is still unusable (recorded as "found, not fixed" in v181). |
| F32 repository structure | **PARTIAL** | #129, #131, #132, #133 | `pyproject.toml`, `src/pgr_vds/`, `cli/`, `research/studies/`, `docs/history/`, `tests/{unit,integration,research}` | Section 6 | Phases 0–5 are done and behaviour is unchanged (September replay byte-identical). **Left:** 197 tracked files outside `tests/conftest.py` still edit `sys.path` (111 in `research/studies`, 54 in `tests/research`, 32 in `scripts/`, including 10 workflow-run scripts). The ratchet only blocks new ones. Only 2 of the 8 entry points in the review's target layout are in `cli/` (`monthly_decision`, `edgar_monthly_fetch`). The older packages are still under `src/` (not `src/pgr_vds/`). |
| F33 EDGAR history rewritten, no provenance | FIXED | #123 | migration 006 (`pgr_edgar_filing_parses`, `pgr_edgar_monthly_raw`, append-only triggers, 3 views); `db_client.upsert_pgr_edgar_monthly` L1118; `pgr_vds/ingestion/edgar_monthly/load.py::load_from_csv` L203 (was `db_client.py:709-780`) | Re-run `load_from_csv` on a DB copy and hash every table; A.5 per-table diff of the 2026-09-25 8-K bot commit | `load_from_csv` inserts 0 rows and changes no table (the review saw 77 cells change). The scheduled 8-K run after the repair (`529afee`) changed no table content. Accessions: 28 undashed → 0. 265 parses under `8k-html/2026-09-25`. |
| F34 2025-09 combined ratio | FIXED | #123 | `parse.py` (ratio rows read decimal cells only) | `q "select combined_ratio, loss_lae_ratio, expense_ratio from pgr_edgar_monthly where month_end like '2025-09%'"` | CR 88.7 → 100.4; ER 23.0 → 34.7 (as filed). |
| F35 live fetch never writes `roe_net_income_ttm` | FIXED | #120 | `db_client._normalise_edgar_monthly_record` L1103–1110 | `q "select month_end, roe_net_income_ttm from pgr_edgar_monthly where month_end >= '2026-02-01'"` | NULL 2026-02 … 08 → 34.7, 35.0, 35.1, 36.0, 34.7, 34.2, 33.4. Reverting the mapping fails 2 round-trip tests. |

## 4. Tests

### 4.1 Full suite

`python -m pytest -o addopts="--tb=short" -q`, in a clean scratch clone of `aae0be8`:

```text
2494 passed, 1 skipped, 136 warnings in 542.04s (0:09:02)
exit code: 0
```

- The count matches the CHANGELOG v185 claim (2494 passed, 1 skipped).
- **DB unchanged by the run:** sha256 `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d` before and after.
- **Tree unchanged:** `git status --porcelain --untracked-files=no` was empty afterwards. Only ignored `__pycache__/` and `.pytest_cache/` folders appeared, plus the `pgr_vds.egg-info/` from my `pip install -e .`.

### 4.2 Each fix has a test that catches it (revert one fix per step)

Method:

- In a scratch clone of `aae0be8`, undo one fix (a 1–3 line edit restoring the pre-fix behaviour).
- Run its named test file and record the result.
- Restore with `git reset --hard && git clean -fd` and run the same file again.

| Step | Finding | Revert applied | Named test | Reverted | Restored |
|---|---|---|---|---|---|
| 1 | F35 | drop the `roe_net_income_trailing_12m` → `roe_net_income_ttm` mapping | `tests/unit/ingestion/test_edgar_monthly_units_and_keys.py` | 2 failed, 4 passed | 6 passed |
| 2 | F03 | remove VOO 2013 from `KNOWN_SPLITS` | `tests/unit/ingestion/test_split_registry.py` | 2 failed, 10 passed | 12 passed |
| 2 | F22 | window end back to `t + DateOffset(months=h)` | `tests/unit/processing/test_target_windows.py` | 5 failed, 4 passed | 9 passed |
| 3a | F06 | `fetch_all_fred_macro(apply_publication_lags=True)` (lag before storing) | `tests/integration/pipeline/test_fred_pipeline.py` | 2 failed, 15 passed | 17 passed |
| 3b | F12 | `_parse_number` drops the sign of `(x)` | `tests/unit/ingestion/test_edgar_monthly_parser_repair.py` | 10 failed, 22 passed | 32 passed |
| 4 | F01 | price features on raw closes | `tests/unit/processing/test_price_features.py` | 6 failed, 13 passed | 19 passed |
| 4b | chart unit check | drop the "repurchase > 15 % of shares" stop | `tests/unit/scripts/test_repurchase_timeseries_charts.py` | 1 failed, 9 passed | 10 passed |
| 4c | 2006-05 buybacks | `apply_pgr_edgar_supplements` returns without applying | `tests/unit/ingestion/test_split_month_buybacks.py` | 2 failed, 7 passed | 9 passed |
| 5 | F04 | naive back to the expanding mean including the scored target | `tests/integration/pipeline/test_validation_gating.py` | 1 failed, 29 passed | 30 passed |
| 6 | F20 | OUTPERFORM ≤ 5 % sells 75 % again | `tests/unit/models/test_live_policy_backtest.py` | 28 failed, 14 passed | 42 passed |
| 6 | F19 | LTCG date back to vest + 366 days | `tests/unit/tax/test_tax_hand_computed.py` | 4 failed, 20 passed | 24 passed |
| 7 | F30 | global `warnings.filterwarnings('ignore')` at import of `binary_classification` | `tests/integration/repo/test_restructure_phase2.py` | 1 failed, 14 passed | 15 passed |
| 9 | F28 (M11) | consensus quality weights stop clipping negative IC | `tests/integration/pipeline/test_mutation_kills.py` | 1 failed, 16 passed | 17 passed |
| 9 | v182 | WFO minimum back to one test window | `tests/unit/models/test_wfo_min_rows.py` | 12 failed, 2 passed | 14 passed |
| 10 | F32 phase 3 | `git mv` v38's script back to `results/research/` | `tests/integration/repo/test_restructure_phase3.py` | 3 failed, 56 passed | 59 passed |
| 11 | F29/F32 phase 4 | `CLAUDE.md` back to a full copy of `AGENTS.md` | `tests/integration/repo/test_restructure_phase4.py` | 1 failed, 25 passed | 26 passed |
| 12 | F32 phase 5 | remove the `src.pgr_vds` import guard | `tests/integration/repo/test_restructure_phase5.py` | 1 failed, 16 passed | 17 passed |

- **Gap found.** Removing step 4b's other guard ("market cap ≠ price × shares" in `capital_return_data.verify_monthly_frame`) fails no test (19 passed). The chart tests check the frames the code builds, never an inconsistent frame (new issue N6).
- **Harness caveat.** Steps 1–9, 11 and 12 ran with `PYTHONPATH=<clone>/src:<clone>`. Step 10 ran with the clone root only. With `src/` first on the path, `src/research` shadows the top-level `research/` namespace package and 14 phase-3 tests fail even on unmodified code. That is a harness artefact, recorded as new issue N7.

### 4.3 Mutation study (F28)

`scripts/checks/mutation_study.py` re-applies the review's 18 production-formula mutations (M01–M18) and 6 extra mutations aimed at the formerly vacuous tests (X19–X24). Each one runs its related test files with `pytest -x` in a scratch clone of `aae0be8`.

- To save time the 24 ran in three scratch clones: M01–M08, M09–M12 and M13–X24. Each run restored its clone afterwards.
- **Result: 0 of 18 F28 mutations survive, and 0 of 6 extras.** Step 9's claim is reproduced; the review had 16 of 18 survivors.

| Mutation | Result | First test that failed (under `tests/`) |
|---|---|---|
| M01_edgar_placement_removed | KILLED | `unit/processing/test_edgar_filing_timing.py::test_first_feature_row_with_an_edgar_row_is_on_or_after_filing_and_within_a_month` |
| M02_roe_placement_removed | KILLED | `integration/pipeline/test_mutation_kills.py::test_m02_quarterly_roe_enters_on_or_after_its_filing_date` |
| M03_combined_ratio_ttm_window_3 | KILLED | `integration/pipeline/test_mutation_kills.py::test_m03_combined_ratio_ttm_is_a_twelve_month_mean` |
| M04_bvps_yoy_pct_change_1 | KILLED | `unit/processing/test_price_features.py::test_bvps_growth_yoy_is_continuous_across_2006_split` |
| M05_drip_value_over_price | KILLED | `unit/processing/test_drip_closed_form.py::TestFlatPriceQuarterlyDividend::test_position_series_matches_product_formula` |
| M06_targets_drop_splits | KILLED | `unit/processing/test_drip_closed_form.py::TestFourForOneSplit::test_db_pipeline_split_is_zero_and_missing_split_is_not` |
| M07_wfo_max_train_size_none | KILLED | `unit/models/test_property_wfo_temporal.py::test_folds_are_embargoed_bounded_and_ordered` |
| M08_live_refit_full_history | KILLED | `unit/models/test_property_wfo_temporal.py::test_live_refit_uses_only_the_recent_window` |
| M09_cpcv_purged_size_0 | KILLED | `integration/pipeline/test_validation_gating.py::test_cpcv_purges_the_horizon_and_embargoes_at_least_two_rows` |
| M10_consensus_lambda_swap | KILLED | `integration/pipeline/test_mutation_kills.py::test_m10_lambda_weights_the_quality_share_not_the_equal_share` |
| M11_consensus_ic_clip_removed | KILLED | `integration/pipeline/test_mutation_kills.py::test_m11_negative_ic_is_clipped_to_zero_quality` |
| M12_consensus_lambda_clip_removed | KILLED | `integration/pipeline/test_mutation_kills.py::test_m12_lambda_outside_unit_interval_is_clipped` |
| M13_underperform_sells_75 | KILLED | `integration/pipeline/test_mutation_kills.py::test_m13_sell_mapping_table` |
| M14_cpcv_fail_gate_dropped | KILLED | `integration/pipeline/test_validation_gating.py::test_missing_or_unknown_cpcv_does_not_permit_actionable` |
| M15_conformal_no_finite_sample | KILLED | `integration/pipeline/test_mutation_kills.py::test_m15_split_conformal_uses_the_finite_sample_quantile` |
| M16_aci_sign_flip | KILLED | `integration/pipeline/test_mutation_kills.py::test_m16_aci_narrows_when_covered_and_widens_after_misses` |
| M17_relative_return_sign_flip | KILLED | `unit/processing/test_multi_total_return.py::TestBuildRelativeReturnTargets::test_relative_return_is_pgr_minus_etf` |
| M18_ltcg_365_days | KILLED | `unit/tax/test_capital_gains.py::TestPositionSummary::test_is_ltcg_eligible_boundary` |
| X19_fred_ffill_to_bfill_lookahead | KILLED | `unit/processing/test_fred_features.py::TestFredFeaturesInMatrix::test_fred_features_use_only_past_data` |
| X20_insurance_cpi_mom_1m | KILLED | `unit/processing/test_pgr_fred_features.py::TestPgrFredFeatures::test_insurance_cpi_mom3m_formula` |
| X21_vmt_yoy_1m | KILLED | `unit/processing/test_pgr_fred_features.py::TestPgrFredFeatures::test_vmt_yoy_is_finite` |
| X22_cr_acceleration_diff_1 | KILLED | `unit/processing/test_feature_engineering_macro_predictors.py::TestCrAcceleration::test_cr_acceleration_is_3period_diff_of_ttm` |
| X23_valuation_ttm_avail_no_rolling_max | KILLED | `unit/processing/test_valuation_multiples.py::test_availability_date_covers_fills_in_ttm_window` |
| X24_fracdiff_never_qualifies | KILLED | `unit/processing/test_fracdiff.py::TestApplyFracdiff::test_memory_preserved_via_correlation` |

## 5. Data-integrity sweep (current DB)

All checks ran on a read-only copy of today's DB (Appendix A.6).

| Check | Result |
|---|---|
| Weekly close ratio outside [0.6, 1.7] with no `split_history` row within ±7 days and not a reviewed move | **0 unexplained**, from the production guard and an independent scan. All 19 large moves are splits (CB, FZROX, KIE, PGR ×2, SCHD, VGT, VOO, VTI, VWO) or the 9 reviewed HIG 2008–09 crisis weeks. |
| Looser screen, ratio outside [0.75, 1.33] away from splits (context) | HIG 13 (2008-10 … 2009-05 and 2020-03), ALL and KIE 2008-10-10, VDE 2020-03-13: genuine market moves |
| Duplicate bars per ticker per ISO week | 0 (review: 31) |
| Weekly gaps > 9 days since 2010 (non-proxy) | none |
| Duplicate (series, month) FRED rows | 0 of 7,574; every label is a business month-end (review: 576 duplicates) |
| FRED interior month gaps | `CUSR0000SAM2` misses 2025-10 (a gap in the FRED series itself; the 5-month fill limit bridges it). `SP500_PRICE_TO_BOOK_MULTPL` (research-only) misses 136 months. |
| Revenue − expenses = pretax (±$0.2M) | 0 of 265 violations |
| CR = loss ratio + expense ratio (±0.15) | 0 of 265 |
| Equity − preferred ≈ BVPS × shares (±6 %) | 0 of 265 (max 0.07 %) |
| Monthly NI summed per quarter vs XBRL quarterly NI (±$1M) | 0 of 73 quarters off (max $0.5M, 2024Q4) |
| Month coverage | EDGAR 2004-08 … 2026-08, none missing. Targets: 21 benchmarks with no interior gaps; the latest 6M target is 2026-02-27 and the latest 12M is 2025-08-29. |
| Net-loss months | 11 months have NI < 0 (2008-08/09/12, 2017-08, 2018-10, 2021-08/09, 2022-04/06/09, 2023-03), consistent with the sign fix |
| Dividend freshness (`check_dividend_freshness`) | **20 of 26 tickers STALE:** VOO, VTI, VIG, VWO (last ex-date Dec 2025), VXUS, VEA, VDE, VFH, VGT, VHT, VIS, VPU, VNQ, KIE, SCHD, ALL (Mar 2026), BND, BNDX, VCIT, VMBS (Mar 2026, monthly payers). PGR, DBC, CB, HIG and TRV are OK; GLD pays none. |
| FRED and price freshness (`check_data_freshness`, reference 2026-09-21) | OK. Every live series covers the month its lag needs; prices are 3 days old. The check has no dividend row (N1). |
| NaN live features in the decision rows 2026-04 … 2026-09 | none (review: `rate_adequacy_gap_yoy` NaN from 2026-04) |

The newest 6M targets are biased toward PGR by the missing ETF dividends (Appendix A.7, missing payments proxied by the same payments a year earlier):

| Benchmark | Last ex-date | Payments missing to 2026-09-25 | Max bias in one 6M target (pp) | Mean over the 2025-09 … 2026-02 targets (pp) |
|---|---|---:|---:|---:|
| VOO | 2025-12-22 | 2 | 0.57 | 0.44 |
| VXUS | 2026-03-20 | 2 | 0.56 | 0.28 |
| VWO | 2025-12-19 | 3 | 0.32 | 0.20 |
| VMBS | 2026-03-02 | 6 | 1.82 | 0.92 |
| BND | 2026-03-02 | 6 | 1.63 | 0.81 |
| GLD | none | 0 | 0.00 | 0.00 |
| DBC | 2025-12-22 | 0 | 0.00 | 0.00 |
| VDE | 2026-03-24 | 2 | 0.61 | 0.31 |

## 6. Restructure checks (steps 7, 10, 11, 12)

- **No behaviour change.**
  - I replayed the September 2026 decision (as-of 2026-09-21, dry run, `--skip-fred`, network blocked) at `b609af6`, the last commit before the restructure (step 6 merged), and at `aae0be8` (today). Both used the same DB, `7c68efbd…`.
  - 10 of the 11 output files are byte-identical: `benchmark_quality.csv`, `classification_shadow.csv`, `consensus_shadow.csv`, `dashboard.html`, `decision_overlays.csv`, `diagnostic.md`, `monthly_summary.json`, `recommendation.md`, `signals.csv` and `plots/calibration_curve.png`.
  - The 11th, `run_manifest.json`, differs only in `git_sha`, `run_timestamp_utc` and `script_name` (`scripts/monthly_decision.py` → `cli/monthly_decision.py`).
  - This covers steps 7, 9 (the v182 WFO minimum), 10, 11 and 12.
- **No `.py` files under `results/`:** 0 tracked (`git ls-files results | grep -c '\.py$'`).
- **Every study folder registered:** `python research/tools/registry.py` → `[registry] 113 studies, 0 problems`.
- **Link checker:** `python scripts/checks/check_doc_links.py` → `[doc-links] 340 files, 0 broken links`.
- **Import smoke test:** `tests/integration/pipeline/test_entrypoint_imports.py` passes in the full suite. Every workflow entry point imports in a fresh interpreter.
- **`sys.path` edits outside `tests/conftest.py`: not met.**
  - `check_sys_path_edits.py` reports `0 new edits, 0 stale allowlist entries`. It is a ratchet, and the allowlist still lists 197 files.
  - An independent grep finds 197 tracked `.py` files editing `sys.path` outside `tests/conftest.py`: 111 study scripts, 54 research tests and 32 `scripts/` files.
  - 10 of the 32 run in workflows: `weekly_fetch`, `peer_fetch`, `initial_fetch`, `bootstrap`, `verify_monthly_outputs`, `capital_return_charts`, `repurchase_timeseries_charts`, `finalize_db`, `check_data_integrity` and `ci_offline_smoke`.
  - `src/`, `config/`, `cli/` and `dashboard/` have none.

## 7. Regressions and side effects

- **Committed artifacts changed after the review.** Each change is explained by a step:
  - DB: steps 1, 2, 3a, 3b, 4c and 5, plus the scheduled weekly and 8-K runs. Per-table content hashes across those commits are in Appendix A.5.
  - `data/processed/pgr_edgar_cache.csv` and `pgr_valuation_monthly.csv`: steps 3b and 4c.
  - The 12 monthly charts: steps 4b and 4c.
  - Two detail CSVs removed: step 10.
  - `decision_log.md`: its header now names `cli/monthly_decision.py`.
  - No research output under `research/studies/*/outputs/` changed content. They were moved, not regenerated.
- **The 2026-09-26 weekly bot run changed 84 target rows by ≤ 0.00044.** It added PGR's July dividend and refreshed two month-to-date FRED values (T10Y2Y, T10YIE for 2026-09). This is normal operation.
- **GitHub Actions on `master` since the review.**
  - CI passed on every merge from #119 to #133, except #125, whose run was cancelled by the #126 merge 8 minutes later (`cancel-in-progress: true`); #126's run passed.
  - The scheduled runs on the fixed code all passed: Weekly Data Accumulation (2026-09-26), Monthly 8-K Fetch (2026-09-25) and the Drift-Based Retrain Trigger.
  - **Not yet run on the new code:**
    - the Monthly Decision Report (next run after the 20 October 8-K fetch, now through `workflow_run` and `cli/monthly_decision.py`);
    - the Wednesday dividend refresh (first on 2026-09-30);
    - the Sunday peer fetch (2026-09-27).
  - Watch these first production runs.
- **Workflows.** No workflow lost a step. The new conditions are intentional:
  - The chart step is now fail-hard, not `continue-on-error`.
  - Verify, charts, commit and e-mail run only on `generated == 'true'`.
  - `drift_retrain_trigger.yml` keeps its own concurrency group. It has `contents: read` and never commits the DB. So the `model_retrain_log` row its header promises "for governance" is never persisted, as before the fixes.
- **Docs that contradict the new behaviour.**
  - The header and footer of the committed `artifacts/monthly_decisions/decision_log.md` still say:
    - runs happen on "the first business day after the 20th if the 20th falls on a weekend" (step 6 changed this to the last business day on or before the 20th);
    - outputs go to `results/monthly_decisions/`;
    - sell % is "Kelly-based";
    - "UNDERPERFORM + Sell % ≥ 75 %".
  - The F29 leftovers listed in the table.
  - The report footer still says "Generated by `scripts/monthly_decision.py`" (kept for byte-identical output; step 12 notes it).
- **New K-fold, shuffle or full-sample scaling: none found in production code.**
  - `grep -rnE "KFold|shuffle=True|cross_val_score|StratifiedKFold" src cli config` finds no production use.
  - CPCV is still computed, now as a labelled diagnostic that cannot gate.
  - Path B's new `StandardScaler` is inside the per-fold pipeline.
  - The research studies still contain the review's LOO `RidgeCV(cv=None)` (F25).
- **`sys.path` hacks reintroduced:** none. The ratchet reports 0 new edits, and the allowlist shrank from 227 to 197.

## 8. New issues found

| # | Severity | Issue | Evidence | Suggested fix |
|---|---|---|---|---|
| N1 | High | Stale ETF dividends are invisible to the monthly decision. `check_data_freshness` reports OK while 6 of 8 live benchmarks' dividend feeds are STALE. Stale dividends only fail the weekly job after a Wednesday refresh, and none has run yet. | Section 5; newest 6M targets biased toward PGR by up to 1.8 pp | Dispatch the dividend refresh now. Add one row per live benchmark from `check_dividend_freshness` to `check_data_freshness` and the report's freshness table, and make a stale live benchmark a manifest warning (or a fail) in the monthly run. |
| N2 | Medium (research) | The research harness still scores R² with look-ahead. `src/research/v37_utils.compute_metrics` calls `compute_oos_r_squared(y_hat, y_true)` with the default `horizon_months=1` on an integer index, so on 6M targets the naive uses 5 unrealised targets. `pool_metrics` concatenates benchmarks, so one benchmark's history counts as the next one's past. | Oracle forecaster on overlapping 6M targets: honest R² +4.9 %, `compute_metrics` −2.4 %. The same series concatenated with a shifted copy scores +13.7 %. | Give every research metric an explicit horizon, dates and a per-benchmark naive. This is step v200 of the re-run plan (`src/pgr_vds/research_lib/`). |
| N3 | Medium | The calibrated "P(outperform)" contradicts the forecast sign. Platt calibration on a 68 % base rate pulls every probability up. | Sept replay: VWO forecast −2.1 % with P(outperform) 83 %; consensus UNDERPERFORM with a mean P(outperform) of 65.8 %. All 8 months show a mean of 61–70 % while the consensus is NEUTRAL or UNDERPERFORM. | Report the probability relative to the base rate, or hide it in the owner report until research v207 re-validates the calibration. |
| N4 | Medium | The conformal "80 %" intervals under-cover badly. | Replayed trailing coverage 40.6–49.0 % in every month from 2026-02 to 2026-09 (target 80 %) | Widen them with prequential residual quantiles, or drop them from the owner report (research v205/v207). |
| N5 | Medium | The suspected 2024 fiscal-month change (F16) is uninvestigated and affects a live Ridge input. | `npw_growth_yoy` standard deviation is 0.198 in 2024 against 0.076 in 2016–2023; the NPE/NPW first-month pattern changes in 2024 | Confirm with PGR's 2023/2024 10-K. Use trailing-12-month or quarterly NPW growth (research v203). |
| N6 | Low | Step 4b's "market cap ≠ price × shares" guard is untested. | Removing the `raise` leaves all 19 chart tests passing | Add a test that feeds `verify_monthly_frame` an inconsistent frame. |
| N7 | Low | Name clash: `src/research` (regular package) shadows the top-level `research/` namespace package whenever `src/` is on `sys.path`. | 14 phase-3 tests fail with `ModuleNotFoundError: research.tools` under `PYTHONPATH=src:.` | Move `src/research` to `src/pgr_vds/research_lib` (planned in v200), or add `research/__init__.py`. |
| N8 | Low | The committed decision log contradicts the current rules and still holds 6 `[DRY RUN]` rows. | Section 7 | Update the header template. Annotate or remove the dry-run rows in a separate, reviewed commit. |
| N9 | Low | Plumbing leftovers from F26/F27/F29/F31. | Verification table | Batch them into one hygiene PR. |
| N10 | Medium | The governance health baseline is stale. `docs/model-governance.md` § "Health baseline at 2026-09-21" (described as the baseline later months are compared with) still shows step-5 numbers. Step 6's filing-date placement (F23) changed them. Decision 0006's replay table still lists 2026-03 and 2026-04 as MONITORING-ONLY. | September on `aae0be8` (same DB): R² +2.86 % (doc +5.08 %), equal-weight IC 0.0762 (0.0706), pooled IC 0.102 with DK p 0.125 (0.130, p 0.030), PT p 0.391 (0.255), CPCV 4/7 MARGINAL (5/7 GOOD). 2026-03/04 are now DEFER (PT p 0.383). | Refresh the baseline table from a post-step-6 replay, and add a dated note to 0006 (history stays as written). |

## 9. Decision impact, 2026-02 to 2026-09

**Method.**

- One read-only dry run per committed as-of date on today's code (`aae0be8`) and a copy of today's DB (`7c68efbd…`), network blocked (Appendix A.4). The DB hash was unchanged after every run.
- "Committed" is the `monthly_summary.json` written at the time.
- "Step 5" is `docs/reviews/2026-09-25_step5_rebaseline_rows.csv`.

**Definitions.**

- **Committed columns.** The IC is the quality-weighted mean IC (the gate at the time), and the hit rate is the equal-weight mean of the per-benchmark hit rates.
- **Step-5 and "now" columns.**
  - The IC is the equal-weight mean IC (the gate now).
  - The hit rate is the pooled prequential hit rate, shown against the base rate.
  - PT is the Pesaran–Timmermann p-value.
  - "pooled IC (DK p)" is the pooled rank IC with its date-clustered p-value.

| Month (as-of) | Committed at the time | Step-5 re-baseline | Now (`aae0be8`) |
|---|---|---|---|
| 2026-02 (02-28) | DEFER, 50 %, UNDERPERFORM; R² −6.0 %; IC 0.142; hit 66.9 % | DEFER, 50 %, NEUTRAL; R² +6.8 %; IC 0.105; hit 64.3 % vs 70.0 %; PT 0.18 | DEFER, 50 %, UNDERPERFORM; R² +3.0 %; IC 0.101; pooled IC 0.126 (DK p 0.049); hit 64.1 % vs 70.0 %; PT 0.36 |
| 2026-03 (03-31) | DEFER, 50 %, NEUTRAL; R² −4.5 %; IC 0.155; hit 66.5 % | MONITORING-ONLY, 50 %, UNDERPERFORM; R² +7.6 %; IC 0.114; hit 65.4 % vs 69.6 %; PT 0.095 | DEFER, 50 %, NEUTRAL; R² +5.7 %; IC 0.106; pooled IC 0.132 (0.083); hit 63.8 % vs 69.6 %; PT 0.38 |
| 2026-04 (04-22) | DEFER, 50 %, NEUTRAL; R² −2.6 %; IC 0.143; hit 66.2 % | MONITORING-ONLY, 50 %, UNDERPERFORM; R² +7.6 %; IC 0.114; hit 65.4 % vs 69.6 %; PT 0.095 | DEFER, 50 %, NEUTRAL; R² +5.7 %; IC 0.106; pooled IC 0.132 (0.083); hit 63.8 % vs 69.6 %; PT 0.38 |
| 2026-05 (05-22) | DEFER, 50 %, NEUTRAL; R² −3.5 %; IC 0.103; hit 63.9 % | DEFER, 50 %, UNDERPERFORM; R² +6.4 %; IC 0.092; hit 63.9 % vs 69.4 %; PT 0.21 | DEFER, 50 %, UNDERPERFORM; R² +3.0 %; IC 0.088; pooled IC 0.108 (0.159); hit 63.1 % vs 69.4 %; PT 0.46 |
| 2026-06 (06-22) | DEFER, 50 %, NEUTRAL; R² −2.1 %; IC 0.098; hit 63.7 % | DEFER, 50 %, UNDERPERFORM; R² +3.8 %; IC 0.091; hit 62.9 % vs 68.8 %; PT 0.35 | DEFER, 50 %, UNDERPERFORM; R² +3.5 %; IC 0.073; pooled IC 0.087 (0.245); hit 61.6 % vs 68.8 %; PT 0.58 |
| 2026-07 (07-22) | DEFER, 50 %, NEUTRAL; R² −2.8 %; IC 0.099; hit 64.9 % | DEFER, 50 %, NEUTRAL; R² +4.0 %; IC 0.076; hit 63.4 % vs 68.3 %; PT 0.22 | DEFER, 50 %, NEUTRAL; R² +5.1 %; IC 0.070 (MARGINAL); pooled IC 0.090 (0.240); hit 61.7 % vs 68.3 %; PT 0.50 |
| 2026-08 (08-20) | DEFER, 50 %, UNDERPERFORM; R² −0.1 %; IC 0.121; hit 65.4 % | DEFER, 50 %, UNDERPERFORM; R² +5.0 %; IC 0.088; hit 62.7 % vs 68.2 %; PT 0.34 | DEFER, 50 %, NEUTRAL; R² +1.2 % (MARGINAL); IC 0.049 (MARGINAL); pooled IC 0.073 (0.292); hit 60.5 % vs 68.2 %; PT 0.62 |
| 2026-09 (09-21) | DEFER, 50 %, NEUTRAL; R² −1.1 %; IC 0.113; hit 66.2 % | DEFER, 50 %, NEUTRAL; R² +5.1 %; IC 0.071; hit 62.8 % vs 68.1 %; PT 0.26 | DEFER, 50 %, UNDERPERFORM; R² +2.9 %; IC 0.076; pooled IC 0.102 (0.125); hit 62.8 % vs 68.1 %; PT 0.39 |

Across all months, the "now" runs have:

- shrinkage α 0.50 (re-chosen prequentially);
- prequential ECE 0.144–0.193;
- trailing conformal coverage 40.6–49.0 %;
- CPCV 4/7 (MARGINAL, diagnostic only);
- a directional-skill gate that FAILS every month.

The as-of 2026-03-31 and 2026-04-22 rows are identical by construction. The April run's latest feature row is 2026-03-31 (the month-end on or before the as-of date), and as-of truncation keeps the same targets.

What explains each difference.

- **Committed → step 5 (every month).** R² turned positive because the naive guess no longer contains the target (F04), and the inputs and targets were repaired (F01, F03/F05, F06/F07, F10–F12, F16–F18, F22, F34, F35). The IC gate moved to the lower equal-weight IC and the hit-rate gate became the PT test (F13). CPCV stopped forcing FAIL (F02).
- **Step 5 → now (every month).** Only step 6's EDGAR filing-date placement (F23) moves these numbers. I replayed the step-5 merge commit (`69889ad`) on today's DB for 2026-02, 03 and 08, and it reproduces the step-5 CSV exactly (e.g. 2026-03: MONITORING-ONLY, UNDERPERFORM, R² +7.6 %, IC 0.114, PT p 0.095). So the 2026-09-26 data update moved nothing material, and the restructure is byte-identical (section 6).

Month by month:

- **2026-02.**
  - Committed → now: the action and the UNDERPERFORM lean are unchanged; R² −6.0 % → +3.0 % (F04 plus data repairs).
  - Step-5 NEUTRAL → now UNDERPERFORM: under F23, VOO moves NEUTRAL → UNDERPERFORM (−0.5 % → −3.3 %) and VMBS OUTPERFORM → NEUTRAL.
  - The equal-weight consensus stays NEUTRAL; the quality weights (F13 residual) make the lean UNDERPERFORM.
- **2026-03.**
  - Step 5 had MONITORING-ONLY because the PT p (0.095) was just under the 0.10 FAIL line. F23 raises it to 0.38, so the month is DEFER again.
  - The lean moves UNDERPERFORM → NEUTRAL: VXUS and VWO move UNDERPERFORM → NEUTRAL, while VMBS and BND move NEUTRAL → OUTPERFORM.
- **2026-04.** Same as 2026-03: it uses the same 2026-03-31 feature row and the same targets.
- **2026-05.** The lean NEUTRAL → UNDERPERFORM comes from the data repairs (already in step 5). F23 lowers R² from +6.4 % to +3.0 % and the IC from 0.092 to 0.088.
- **2026-06.** NEUTRAL → UNDERPERFORM, as in May. F23 lowers the IC from 0.091 to 0.073.
- **2026-07.** NEUTRAL throughout. F23 lowers the IC from 0.076 to 0.070, and the IC gate becomes MARGINAL.
- **2026-08.** UNDERPERFORM (committed and step 5) → NEUTRAL.
  - Under F23, BND and DBC move UNDERPERFORM → NEUTRAL and VXUS moves NEUTRAL → UNDERPERFORM. The mean forecast moves from −4.6 % to −2.3 %.
  - R² (+1.2 %) and IC (0.049) fall to MARGINAL.
- **2026-09.** NEUTRAL → UNDERPERFORM: F23 adds VOO and VWO to the underperform side, as the step 6 report says.
  - R² +5.1 % → +2.9 %; pooled IC 0.130 → 0.102.
  - The equal-weight consensus is NEUTRAL (mean −1.0 %); the quality-weighted lean is UNDERPERFORM.
- **Mode, every month.** DEFER-TO-TAX-DEFAULT at 50 %, as committed. The binding reason changed: before, the leaky R² and the impossible CPCV forced it; now the directional-skill test fails on its merits (PT p 0.36–0.62).

## Appendix A — Commands and scripts

All scripts ran from a scratch clone with the package installed (`pip install -e ".[dev]"`), against DB copies outside the repository.

**A.1 Read-only query helper**

```bash
q() { python -c 'import sqlite3,sys
c=sqlite3.connect("file:data/pgr_financials.db?mode=ro&immutable=1",uri=True)
for r in c.execute(sys.argv[1]): print(r)' "$1"; }
```

**A.2 Feature checks (F01, F06, F07, F15, F23).** Build the production matrix read-only, with the cache redirected, at each commit on its own DB. For the review-time build, put the review-time clone first on `PYTHONPATH`.

```python
import sqlite3, pandas as pd, numpy as np
import src.processing.feature_engineering as fe
fe._PROCESSED_PATH = "/tmp/fm_cache.parquet"            # keep the cache out of the repo
con = sqlite3.connect("file:data/pgr_financials.db?mode=ro&immutable=1", uri=True)
fm = fe.build_feature_matrix_from_db(con, force_refresh=True)
# Hand calculation: divide pre-split closes by each split ratio, then take calendar-month returns
px = pd.read_sql("select date, close from daily_prices where ticker='PGR' and proxy_fill=0 order by date",
                 con, parse_dates=["date"]).set_index("date").close
for d, r in pd.read_sql("select split_date, split_ratio from split_history where ticker='PGR'",
                        con, parse_dates=["split_date"]).itertuples(index=False):
    px[px.index < d] /= r
close_on = lambda t: px[px.index <= t].iloc[-1]
mom12 = lambda t: close_on(t) / close_on(t - pd.DateOffset(months=12) + pd.offsets.MonthEnd(0)) - 1
vol13 = lambda t: np.log(px[px.index <= t]).diff().dropna().iloc[-13:].std() * np.sqrt(52)
print(fm.loc["2026-08-31", ["mom_12m", "vol_63d"]], mom12(pd.Timestamp("2026-08-31")), vol13(pd.Timestamp("2026-08-31")))
# F06 lag identity: fm["vix"] at month M equals VIXCLS stored for month M-1 (NFCI: M-2)
# F07: [c for c in config.MODEL_FEATURE_OVERRIDES["ridge"] + config.MODEL_FEATURE_OVERRIDES["gbt"] if fm[c].iloc[-1] != fm[c].iloc[-1]]
# F23: for each pgr_edgar_monthly row, BMonthEnd().rollforward(filing_date) must be the first row carrying its npw_growth_yoy
```

**A.3 Validation logic (F02, F04, F13, F20).** Run once at `9887288` and once at `aae0be8`.

```python
import numpy as np, pandas as pd
import src.models.wfo_engine as w, src.reporting.backtest_report as br, src.reporting.decision_rendering as dr
rng = np.random.default_rng(0); idx = pd.date_range("2000-01-31", periods=240, freq="ME"); x = rng.normal(size=240)
X = pd.DataFrame({"f1": x, "f2": rng.normal(size=240)}, index=idx)
y = pd.Series(0.9 * x + 0.01 * rng.normal(size=240), index=idx, name="target")
r = w.run_cpcv(X, y, model_type="ridge", target_horizon_months=6, n_folds=8, n_test_folds=2)
print(r.n_paths, len(r.path_ics), r.n_positive_paths, r.stability_verdict)      # F02
eps = rng.normal(scale=0.05, size=406); mu = 0.02 + 0.03 * np.sin(np.arange(400) / 9.0)
yy = np.array([mu[i] + eps[i:i + 6].sum() for i in range(400)]); d = pd.date_range("1990-01-31", periods=400, freq="ME")
print(br.compute_oos_r_squared(pd.Series(mu, index=d), pd.Series(yy, index=d), horizon_months=6))  # F04 (drop the kwarg at 9887288)
h = {"oos_r2": 0.03, "agg_hit": 0.68, "constant_rule_hit_rate": 0.68, "pt_p_value": float("nan")}
print(dr.determine_recommendation_mode("UNDERPERFORM", -0.05, 0.10, 0.68, h, None))          # F20: CPCV missing
print(dr.sell_pct_from_consensus("OUTPERFORM", 0.03, 0.10))                                    # F20: mapping
```

**A.4 Offline dry runs (F14, section 6, section 9).** `offline_run.py`:

```python
import runpy, socket, sys
def _blocked(*a, **k): raise RuntimeError("network blocked by verifier")
socket.socket.connect = socket.socket.connect_ex = socket.create_connection = _blocked
script = sys.argv[1]; sys.argv = sys.argv[1:]; runpy.run_path(script, run_name="__main__")
```

Run from a scratch clone, with the clone root on `PYTHONPATH`:

```bash
sha256sum data/pgr_financials.db
python offline_run.py cli/monthly_decision.py --dry-run --as-of 2026-09-21 --skip-fred
sha256sum data/pgr_financials.db
git status --porcelain --untracked-files=no
```

At `b609af6` and `9887288` the entry point is `scripts/monthly_decision.py`. Outputs are in `results/dry_run/monthly_decisions/YYYY-MM/` (or `results/monthly_decisions/` at `9887288`, which is the F14 bug).

**A.5 Per-table DB diffs across commits.** `git show <c>:data/pgr_financials.db > <c>.db` for each commit, then a sha256 of every table's rows ordered by all columns:

| Commit | Tables whose content changed (rows before → after) |
|---|---|
| step 1 `dc0e24a` | `pgr_edgar_monthly` (263 → 263), `schema_migrations` (3 → 4) |
| step 2 `34835d5` | `daily_prices` (29,797 → 29,766), `monthly_relative_returns` (9,658 → 9,658), `split_history` (9 → 11) |
| step 3a `15644cd` | `fred_macro_monthly` (8,079 → 7,574), `schema_migrations` (4 → 5) |
| step 3b `89dc3e8` | `pgr_edgar_filing_parses` (0 → 265), `pgr_edgar_monthly` (263 → 265), `pgr_edgar_monthly_raw` (0 → 15,323), `pgr_fundamentals_quarterly` (74 → 73), `schema_migrations` (5 → 7) |
| bot 8-K fetch `529afee` | none |
| step 4 `8c3b2bc` | none |
| steps 4b/4c `c089124` | `pgr_edgar_filing_parses` (265 → 266), `pgr_edgar_monthly` (265 → 265), `pgr_edgar_monthly_raw` (15,323 → 15,329) |
| bot weekly `a31ca5f` | `api_request_log`, `daily_dividends` (2,389 → 2,391), `daily_prices` (29,766 → 29,788), `fred_macro_monthly` (2 values), `ingestion_metadata`, `monthly_relative_returns` (84 rows, ≤ 0.00044) |
| step 5 `69889ad` | `model_performance_log` (8 → 8), `schema_migrations` (7 → 8) |

**A.6 Integrity sweep.**

- `src.processing.price_integrity.find_unexplained_price_jumps(con)` and `find_duplicate_week_bars(con)`.
- `db_client.check_dividend_freshness(con)` and `db_client.check_data_freshness(con, date(2026, 9, 21))`.
- Plus these SQL identities:
  - `total_revenues − total_expenses − income_before_income_taxes`;
  - `loss_lae_ratio + expense_ratio − combined_ratio`;
  - `(shareholders_equity − 493.9·[2018-03…2024-01]) / (book_value_per_share · common_shares_outstanding) − 1`;
  - monthly `net_income` summed by calendar quarter against `pgr_fundamentals_quarterly.net_income / 1e6`.

**A.7 Dividend-staleness bias.**

- For each benchmark, take the dividends paid one year earlier, shift them by one year, and keep those after the last stored ex-date and on or before 2026-09-25.
- Divide each by the close on its shifted date.
- Sum them within each 6M target window starting 2025-09-30 … 2026-02-27.

## Appendix B — Where the review's cited files are now

| Review cited | Now |
|---|---|
| `scripts/monthly_decision.py` (4,080 lines) | `cli/monthly_decision.py` (argument parsing) + `src/pgr_vds/decision/{schedule,refresh,signal_generation,health,tax_lots,portfolio,rendering,recommendation_report,diagnostic_report,artifacts,pipeline}.py` (step 12; not a git rename) |
| `scripts/edgar_8k_fetcher.py` | `src/pgr_vds/ingestion/edgar_monthly/{fetch,parse,derive,load}.py` + `cli/edgar_monthly_fetch.py` (`git log --follow` shows `R062 → parse.py`) |
| `src/ingestion/edgar_8k_fetcher.py`, `src/ingestion/pgr_monthly_loader.py` | deleted (step 12) |
| `results/research/v46_classification.py` | `src/research/binary_classification.py` (`R084`, step 7) |
| `results/research/<id>_*.py` and outputs | `research/studies/<id>_<slug>/` (step 10, 705 renames) |
| `results/v9 … v28/` | `research/legacy/` |
| `results/monthly_decisions/`, `results/v14/shadow_reviews/`, `results/research/pgr_*.png`, `data/fetch_status.md` | `artifacts/monthly_decisions/`, `artifacts/shadow_reviews/`, `artifacts/charts/`, `artifacts/ops/fetch_status.md` (step 7) |
| `docs/{plans,superpowers,closeouts,results,archive}/` | `docs/history/…` (step 11) |
| `tests/test_*.py` (flat) | `tests/{unit,integration,research}/…` (step 11) |
| `pytest.ini`, `mypy.ini`, `ruff.toml`, `requirements*.txt`, `constraints-dev.txt` | `pyproject.toml` (steps 7 and 11) |
| `claude.md` | `CLAUDE.md` (imports `AGENTS.md`) |
| Everything else cited (`src/processing/*`, `src/models/*`, `src/reporting/*`, `src/tax/*`, `src/ingestion/*`, `src/database/db_client.py`, `config/*.py`, `scripts/weekly_fetch.py`, `.github/workflows/*`) | same path; the line numbers moved (current lines are in the table in section 3) |
