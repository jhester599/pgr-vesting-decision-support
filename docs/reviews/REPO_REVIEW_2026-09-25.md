# Repository Review — 2026-09-25

- **Scope:** read-only review of `master` at `c948d0f` (Merge PR #118), on branch `review/2026-09-25`.
- **Rules audited against:** `AGENTS.md` (identical copy in `claude.md`).
- **Method:**
  - 12 parallel review agents, one per review area, plus direct verification by the orchestrator.
  - Every finding below was reproduced with a command or script. Headline numbers were re-checked by the orchestrator against the DB or the code.
  - All database work used read-only or immutable connections, or copies outside the repo.
  - SEC EDGAR was used only for the permitted values (Appendix A).
- **Test suite (full, isolated clone):** `1965 passed, 2 skipped, 421 warnings in 326.33s (0:05:26)`, pytest exit code **0**.
  - The tracked DB was byte-identical afterwards: sha256 `e7531a34…8b38` before and after.
  - Appendix B has the details.

Status labels:
- **CONFIRMED** means reproduced with the command or script cited.
- **SUSPECTED** means reasoned from evidence, with the missing confirmation stated.

## 1. Executive summary

The live recommendation has been **DEFER-TO-TAX-DEFAULT / sell 50 %** in all 8 committed months (2026-02 … 2026-09). Several agents replayed the September run exactly and applied each fix below in turn. No single fix changes September's sell %. That is mostly because two validation bugs (F02, F04) keep the system locked in DEFER whatever the model says. The model inputs, targets and health metrics are nevertheless materially wrong.

The issues that matter most:

1. **F01 — The "daily" price table is weekly and split-unadjusted, but the live features treat rows as trading days.**
   - `mom_3m`, `mom_6m` and `mom_12m` are really 14.5-, 29- and 58-month returns. `vol_63d` covers 63 weeks and is annualised by √252.
   - Live `mom_12m` for the latest decision row is **+1.28 against a true 12-month return of −0.10**.
   - Correcting these four live features, which appear in both Ridge and GBT, drops pooled IC from 0.165 to 0.104. The quality-weighted IC falls below its 0.07 gate. VOO and VXUS forecasts flip sign, and the consensus flips on 3 of 6 past vest dates.
2. **F02 — The CPCV gate can never pass.**
   - The thresholds (19 GOOD / 9 MARGINAL) assume 28 paths, but C(8,2) produces 7.
   - The path loop also iterates folds instead of paths.
   - Every month is therefore forced into DEFER. The gate itself is a combinatorial K-fold, which AGENTS.md prohibits.
3. **F04 — The OOS-R² gate compares the model against a naive benchmark that includes the value being predicted.** Pooled R² is reported as −1.13 %, while look-ahead-free benchmarks give **+10.8 % to +13.9 %**. The binding R² gate is therefore wrong in the conservative direction.
4. **F03 and F05 — Two share splits are missing from `split_history`: VOO 1-for-2 (2013-10-25) and VGT about 8-for-1 (2026-04-24).**
   - They corrupt 28 target rows by 106–140 return points. All 10 VGT labels change sign; VGT has weight 0.20 in the investable classifier pool.
   - With VOO corrected, the VOO CPCV IC moves from −0.054 to +0.221.
5. **F06 and F07 — The FRED pipeline is broken.**
   - Publication lags are applied twice, and also by row over duplicate month-end rows, so macro features are 2 months stale (NFCI 3–4).
   - The four PGR-specific series have not been refreshed since 2026-02/03.
   - The live GBT feature `rate_adequacy_gap_yoy` has been silently median-imputed since April. The freshness check still reports OK.
6. **F10 and F11 — Two more live GBT features are corrupted.**
   - `investment_book_yield` mixes percent and fraction units.
   - `pif_growth_yoy` is NaN every February after a leap year and inflated by a 2024 `pif_total` definition change.
7. **F08 — Dividends stopped updating on 2026-03-26.**
   - The weekly PGR dividend call fails silently, and ETF dividends refresh only once a year.
   - Relative-return targets are biased toward PGR by up to about 2 pp today.
   - If the January 2027 special dividend (last year $13.60, about 6 %) is missed, the bias grows to about 6 pp.
8. **F12, F09, F18, F16 and F17 — EDGAR parsing defects, verified against filings.**
   - Parenthesised negatives lose their sign. The Aug-2017 and Oct-2018 net losses are stored as profits.
   - Q4 fundamentals are annual figures, so Q4 ROE is 4× too high.
   - Equity and debt are mis-parsed in 11 rows.
   - Two monthly releases were never ingested. They exist on EDGAR, and the fetcher's pagination cannot reach them.
   - YoY values span 13 months at the gaps.
9. **F13 — Reported model health is inflated by in-sample choices.** The shrinkage α, quality weights, ECE and conformal coverage are all fitted on the same OOS period they report on. The hit-rate gate passes on base rate alone.
10. **F14 — `--dry-run` writes to the DB and overwrites committed monthly artifacts.** CI, the README and the runbook all tell people to run it.
11. **F33 — The EDGAR monthly table has no provenance and is silently rewritten.**
    - Each monthly job re-parses 24 months. The upsert overwrites `filing_date` and `accession_number` and COALESCEs each value column, so one row can mix CSV and live-parser values.
    - The live parse overwrote the 2025-09 combined ratio: 88.7 in the DB vs 100.4 as filed (F34).
    - Re-running the documented `load_from_csv` step changes 77 cells.

Test adequacy: 16 of 18 targeted mutations of production formulas survived the tests (F28). Covered: EDGAR lag, TTM window, DRIP reinvestment, split handling, consensus weights, sell-% mapping, conformal quantile and WFO window.

## 2. Findings table

Severity scale:
- **Critical:** can flip or lock a production decision or its validation.
- **High:** corrupts a live model input, target or validation metric.
- **Medium:** research-only, shadow-only, or a small quantified production effect.
- **Low:** docs, hygiene or robustness.

| ID | Sev | Area | Location (file:line) | Status | One-line impact |
|---|---|---|---|---|---|
| F01 | Critical | Features / data | `src/processing/feature_engineering.py:166-175,253-286,1370-1378`; `src/ingestion/multi_ticker_loader.py:43` | CONFIRMED | Live momentum and vol features measure 14–58-month windows on unadjusted weekly prices; `mom_12m` +1.28 vs true −0.10 |
| F02 | Critical | Validation | `src/models/wfo_engine.py:521-530,660-687`; `config/model.py:97`; `src/reporting/decision_rendering.py:50,65` | CONFIRMED | CPCV verdict always FAIL, so every month is forced to DEFER; CPCV also violates the no-K-fold rule |
| F03 | Critical | Targets | `scripts/weekly_fetch.py:55-73`; `scripts/apply_split_history.py:48-65`; `src/processing/multi_total_return.py:111-127` | CONFIRMED (data); split itself SUSPECTED | VOO 2013 split missing: 18 targets off by 106–134 pts; VOO CPCV IC −0.054 vs +0.221 |
| F04 | Critical | Validation | `src/reporting/backtest_report.py:63-65` | CONFIRMED | OOS-R² naive benchmark uses the current target: R² −1.13 % vs +10.8 to +13.9 % look-ahead-free |
| F05 | High | Targets | same as F03 | CONFIRMED (data) | VGT 2026 split missing: 10 labels flipped, in the current shadow-classifier training window |
| F06 | High | Data / timing | `src/ingestion/fred_loader.py:131,190-195`; `feature_engineering.py:102-122,1219-1222` | CONFIRMED | FRED lagged twice and row-shifted over duplicate month rows; macro features 2–4 months stale |
| F07 | High | Data / decision | `scripts/weekly_fetch.py:105`; `scripts/monthly_decision.py:309`; `src/models/wfo_engine.py:409-416`; `src/database/db_client.py:304-387` | CONFIRMED | Live GBT feature NaN since 2026-04, silently imputed; freshness check blind to it |
| F08 | High | Data / targets | `scripts/weekly_fetch.py:162-175`; `src/ingestion/multi_dividend_loader.py:208-224`; `.github/workflows/initial_fetch_dividends.yml:19` | CONFIRMED (staleness); cause SUSPECTED | No dividends since 2026-03-26; targets biased to PGR by up to ~2 pp; ~6 pp risk in Jan 2027 |
| F09 | Medium | EDGAR data | `src/ingestion/edgar_client.py:197-220,365-371`; `feature_engineering.py:1180-1189` | CONFIRMED (EDGAR) | Q4 rows are annual; ROE 4× too high Feb–Apr each year; pe/pb always NULL |
| F10 | High | EDGAR / features | `scripts/edgar_8k_fetcher.py:816-817`; `feature_engineering.py:700-706` | CONFIRMED | Live GBT `investment_book_yield` mixes 3.8 and 0.038 units |
| F11 | High | EDGAR / features | `scripts/edgar_8k_fetcher.py:1753-1765,1813-1823`; `feature_engineering.py:325-353` | CONFIRMED | Live GBT `pif_growth_yoy` NaN in Feb 2009/13/17/21/25; 2024 PIF definition break; mean forecast −3.71 % → −2.71 % when fixed |
| F12 | Medium | EDGAR data | `scripts/edgar_8k_fetcher.py` (monthly HTML value parser) | CONFIRMED (EDGAR) | Net-loss months stored as profits; TTM EPS +$0.06/+$0.12 for 24 months; realized gains sign-flipped |
| F13 | High | Validation | `config/model.py:46`; `src/models/evaluation.py:451-495`; `src/models/consensus_shadow.py:45`; `scripts/monthly_decision.py:532-571,609-721`; `decision_rendering.py:49`; `src/models/forecast_diagnostics.py:103-105` | CONFIRMED | Health metrics fitted in-sample; July 2026 IC gate passed only because of quality weighting; p-values ~600× overstated |
| F14 | High | Decision / ops | `scripts/monthly_decision.py:731-779,3039,3180,3402-3683,3576` | CONFIRMED | `--dry-run` rewrites the DB row and 10 committed files and appends ledgers; 6 `[DRY RUN]` rows already committed |
| F15 | Medium | Per-share | `feature_engineering.py:516-528,745-763,1191-1205,1227-1366` | CONFIRMED | 2006 split: live-Ridge BVPS growth −70 % for 12 months; P/B 0.84/0.78; buyback yield 4×; synthetic spreads |
| F16 | Medium | EDGAR / windows | `scripts/edgar_8k_fetcher.py:2430,2442` | CONFIRMED (EDGAR) | `npw_growth_yoy` (live Ridge) 2016-05 −15.2 % vs true +10.6 %; 2020-04 +27.2 % vs +2.6 % |
| F17 | Medium | EDGAR data | `scripts/edgar_8k_fetcher.py:224-240`; `feature_engineering.py:314-317` | CONFIRMED (EDGAR) | 2015-05 and 2019-04 releases never ingested; older filings unreachable; forward-filled duplicate months |
| F18 | Medium | EDGAR data | monthly parser field matching (`scripts/edgar_8k_fetcher.py`) | CONFIRMED (EDGAR) | 11 rows store ROE % as equity and debt/capital % as debt; ratio features reach 135,964 |
| F19 | Medium | Tax | `src/tax/capital_gains.py:58,155-157,442-467,524,615,629-639`; `src/tax/monte_carlo.py:125-145`; `scripts/monthly_decision.py:1184-1301` | CONFIRMED | Breakeven sign inverted; leap-year LTCG off by one day; no wash-sale handling; MC vol 3.6× too high |
| F20 | Medium | Decision | `src/reporting/decision_rendering.py:14-50`; `scripts/monthly_decision.py:406-410` | CONFIRMED | ACTIONABLE mapping loses to always-50 % OOS; a missing CPCV fails open |
| F21 | Medium | Decision | `src/models/multi_benchmark_wfo.py:310-366` | CONFIRMED | P(outperform) always 0.5 and confidence always LOW since BayesianRidge retired |
| F22 | Medium | Targets | `multi_total_return.py:115-127`; `feature_engineering.py:1454-1473`; `db_client.upsert_prices` | CONFIRMED | Weekly windows 175–190 days; 31 partial-week duplicate bars; as-of truncation keeps a post-as-of target |
| F23 | Medium | Timing | `config/features.py:237`; `feature_engineering.py:142-162` | CONFIRMED | Fixed 2-month lag has no look-ahead but discards one month on every row; placing rows by filing date flips consensus to UNDERPERFORM |
| F24 | Medium | Shadow layer | `src/models/path_b_classifier.py:286`; `classification_shadow.py:765-803`; `classification_gate_overlay.py:108-112`; `src/research/v160_ta_features.py` | CONFIRMED | Path B scores a stale row (0.4825 vs 0.5805); monitoring never matures; TA features on weekly unadjusted prices |
| F25 | Medium | Research | `src/research/pb_vs_pe.py:217-359`; `src/research/x1_targets.py`; `results/research/v39_ridge_alpha.py:63` (and v40–v59) | CONFIRMED | P/B-vs-P/E conclusions overstated; x-series targets on raw prices; LOO RidgeCV in v39–v59 |
| F26 | Medium | CI / ops | `.github/workflows/*.yml`; `config/api.py:34-41`; `config/model.py:163-171` | CONFIRMED (config); email duplicates SUSPECTED | Unserialised DB writers; yearly "one-time" bootstraps; placeholder EDGAR User-Agent; env-typo falls back to a retired mode |
| F27 | Medium | Data | `fred_macro_monthly`, `daily_prices` (CB), `src/research/v19.py` | CONFIRMED | Duplicate FRED month rows (64 per series); CB spliced from two companies; FRED latest-vintage |
| F28 | Medium | Tests | `tests/` (see section) | CONFIRMED | No weekly, split, gap or leap fixtures; several vacuous or leak-asserting tests |
| F29 | Low | Hygiene | `.gitignore:15-21`; DB header; `pytest.ini`; `requirements.txt`; docs | CONFIRMED | Ineffective ignore negations; WAL sidecars not ignored; `-qq` hides CI summary; unpinned pandas 3; docs drift |
| F30 | Low | Research / infra | `results/research/v46_classification.py`; `src/research/v66_utils.py:15`; `feature_engineering.py:1059` | CONFIRMED | Production imports code from `results/` that silences warnings; unversioned parquet cache; tests write into the repo |
| F31 | Low | Models | `feature_engineering.py:1480-1669`; `src/models/blp.py:194`; `src/models/drift_monitor.py:45` | CONFIRMED | Fracdiff unusable and full-sample; BLP fit ignores outcomes; drift monitor cannot see recent drift |
| F32 | Low | Structure | whole repo | CONFIRMED | Four result layouts, code in `results/`, 214 `sys.path` hacks, no package metadata (section 5) |
| F33 | High | EDGAR data | `src/database/db_client.py:709-780`; `.github/workflows/monthly_8k_fetch.yml` | CONFIRMED | No point-in-time values or provenance; rows mix CSV and live parses; `load_from_csv` rerun changes 77 cells |
| F34 | Medium | EDGAR data | `scripts/edgar_8k_fetcher.py:937-947,1087-1092` | CONFIRMED (internal identity) | 2025-09 CR 88.7 vs 100.4 filed: live-Ridge `combined_ratio_ttm` understated 0.975 in the current row |
| F35 | Medium | EDGAR data | `scripts/edgar_8k_fetcher.py:988,1728` vs `db_client.py:831` | CONFIRMED | Live fetch never writes `roe_net_income_ttm` (key mismatch): NULL 2026-02 → 08 |

## 3. Findings in detail

For each finding: evidence, impact, root cause, fix, and the test that would catch it. Scratch scripts cited are in the temp directory listed at the end.

### F01 — Weekly, unadjusted prices fed to trading-day windows (Critical, CONFIRMED)

**Evidence**
- `daily_prices` holds 52–53 rows per year for every one of 27 tickers, with a median gap of 7 days. Source is Alpha Vantage `TIME_SERIES_WEEKLY` (`multi_ticker_loader.py:43`). The loader docstring explains that daily data needs a premium subscription.
- Prices are unadjusted. PGR weekly closes: 2006-05-12 = 107.20, 2006-05-19 = 27.24.
- `build_feature_matrix` uses row windows: `shift(63/126/252)`, `rolling(21/63)` × √252 and a 252-row high (`feature_engineering.py:166-175,253-286`).
- `build_feature_matrix_from_db` passes the raw closes straight through (`:1370-1378`).
- The production callers are `scripts/monthly_decision.py:343,1328`, `classification_shadow.py:424,595` and `cross_check.py:185`.

Orchestrator reproduction on a DB copy (`orch/feature_matrix_orch.csv`):

| Feature | Production | True value | Correlation with correct definition |
|---|---|---|---|
| `mom_12m` (2026-09-30) | +1.257 | −0.123 (split-adjusted, calendar 12 months) | 0.516 |
| `mom_3m` | — | — | 0.394 |
| `vol_63d` vs 13-week vol | — | — | 0.140 |

- **At the 2006 split:** `mom_12m` = −0.80, `vol_63d` = 2.77, `high_52w` = 0.155.
- **Split contamination by feature (months affected):** `mom_3m` 30, `mom_6m` 58, `mom_12m` 78 (2004-09 → 2011-02), `vol_63d` 30, `high_52w` 107. Source: area-2 `feature_split_audit.py`.

**Impact**
- The affected live features are Ridge `mom_12m` and `vol_63d`, and GBT `mom_3m`, `mom_6m`, `mom_12m` and `vol_63d`.
- Agents replayed the September decision with these features recomputed correctly (area 7 `t8_price_feature_fix.py`, area 4 `decision_impact.py`):

| Metric | Production features | Corrected features |
|---|---|---|
| Pooled IC | 0.165 | 0.104 |
| Hit rate | 66.6 % | 62.2 % |
| Quality-weighted IC | 0.113 | 0.052 (fails the 0.07 gate) |
| VOO forecast | +3.0 % | −1.9 % |
| VXUS forecast | −3.5 % | +0.5 % |
| VMBS signal | OUTPERFORM | NEUTRAL |

- Across 6 past vest dates the consensus flips 3 times: 2026-01, 2025-07 and 2024-07.
- September stays DEFER because of F02 and F04.
- Contaminated 2002–2011 rows sit in the training windows of 12–13 walk-forward folds for VWO, GLD and VDE.
- The research that selected these features (v18/v20) used the mis-specified versions.
- The same defect affects:
  - Monte Carlo volatility (F19);
  - the TA shadow features (F24);
  - `pgr_vs_kie_6m`, `pgr_vs_peers_6m` and `pgr_vs_vfh_6m`, which reach 8–10 SD outliers at splits (F15).

**Root cause**
- The data source switched from daily to weekly. The v2 plan specified `TIME_SERIES_DAILY` (`docs/archive/claude-plan-v2.md:100`).
- The feature code, its names and 20 docstrings kept daily semantics.
- There is no split adjustment before feature computation.

**Fix**
- Build one split-adjusted close series: `close × share_basis_factor / latest`. Keep raw closes only for DRIP.
- Compute momentum on month-end closes with calendar offsets.
- Compute volatility from 13 weekly returns × √52, and the 52-week high over 52 rows.
- Alternatively, accumulate real daily bars (for example a weekly `TIME_SERIES_DAILY` compact fetch).
- Assert the bar frequency inside the builder.
- Rename `daily_prices`, or add a `frequency` column.
- Then re-run the feature-selection research.

**Test that would catch it:** a weekly synthetic series with a known 12-month return and a 4:1 split. Assert that `mom_12m` equals the split-adjusted calendar return and that no feature is discontinuous across the split.

### F02 — CPCV verdict can never pass, and CPCV is a K-fold (Critical, CONFIRMED)

**Evidence**
- `wfo_engine.py:523-530` sets `good = config.DIAG_CPCV_MIN_POSITIVE_PATHS` (19) and `marginal = 19 // 2` (9).
- skfolio `CombinatorialPurgedCV(8,2)` gives 28 splits but `n_test_paths = 7`, with `recombined_paths` shape (8, 7).
- The loop at `:660-687` iterates rows (folds), not columns (paths), and adds both test folds per split. It yields 8 pooled "fold" ICs of about 322 observations each.
- All 8 committed `diagnostic.md` files show "CPCV Positive Paths 1/7 (12.5 %) ❌", e.g. `results/monthly_decisions/2026-09/diagnostic.md:20`. The report's "≥ 5/7" threshold text does not match the code.
- In replays, 7/7 positive paths still returns FAIL (area 2 `replay_decision.py` B/D/E; area 7 `t4_cpcv.py`).

**Impact**
- `determine_recommendation_mode` (`decision_rendering.py:60-73`) returns DEFER-TO-TAX-DEFAULT every month regardless of model quality.
- CPCV trains on data after the test blocks: 48 of 56 test blocks have later training rows. It is a combinatorial K-fold used as a production gate, contrary to AGENTS.md "MUST NEVER use K-Fold".
- The embargo is 0 (`wfo_engine.py:589-594`).

**Fix**
- Preferred: demote CPCV to diagnostic-only and gate on walk-forward metrics.
- If it is kept: iterate `recombined_paths.T`, take fold i from split `rp[i, p]` only, and scale the threshold to `ceil(19/28 × n_paths)`.
- Set `embargo_size ≥ 2`.

**Test that would catch it:** for 8 folds and 2 test folds, assert 7 paths each covering every row exactly once, and assert a perfect predictor gets GOOD.

### F03 — VOO 1-for-2 reverse split missing (Critical, CONFIRMED in data; corporate action SUSPECTED)

**Evidence**
- Orchestrator query: VOO weekly closes 2013-10-18 = 79.87 and 2013-10-25 = 161.20. VTI moved +0.8 % that week.
- The VOO/VTI price ratio moves from 0.878 to 1.758. The dividend per share goes from 0.393 to 0.914. Volume halves.
- There is no VOO row in `split_history`. It has 9 rows, all internally correct: PGR 1992/2002/2006, CB 2006, VTI/VWO/FZROX 2008, KIE 2017, SCHD 2024.
- Stored 6M VOO rows 2013-04-30 → 2013-09-30 have `benchmark_return` of +1.13 to +1.32 and `relative_return` of −1.14 to −1.30.
- Area 1b and area 6 independently reproduced all 9,658 stored target rows exactly (max diff 1.8e-15). So the target code is correct and the missing split row is the only defect.

**Impact**
- 18 VOO rows (6 at 6M, 12 at 12M) are wrong by 106–134 return points. They are 70.8 % of VOO's 6M target sum of squares.
- They sit in the training windows of 6 of 19 VOO folds.

| VOO metric | As stored | Corrected |
|---|---|---|
| Walk-forward IC | −0.097 | −0.026 |
| Newey-West IC | −0.049 | +0.042 |
| Representative CPCV mean-path IC | −0.054 | +0.221 (7/7 positive) |
| Aggregate OOS R² | −1.13 % | −0.82 % |
| Path B P(actionable sell) | 49.4 % | 62.1 % (VOO and VGT fixed) |

- Other consumers: the redeploy scorer's `corr_to_pgr` for VOO (0.121 vs 0.351) and `commodity_equity_momentum` for 2013-10 → 2014-03 (−1.24 vs −0.12).

**Root cause:** splits come from three hand-maintained lists, and nothing ingests or detects them.

**Fix**
- Add `("VOO", "2013-10-24", 0.5)` to one canonical source, after confirming the date with the issuer notice.
- Ingest split coefficients, e.g. from AV's adjusted series or `SPLITS` endpoint.
- Add a jump guard, then rebuild `monthly_relative_returns`.

**Test that would catch it:** every weekly close ratio outside [0.6, 1.7] must be explained by a `split_history` row within ±7 days. This fails today exactly on VOO 2013-10-25 and VGT 2026-04-24.

### F04 — OOS-R² naive benchmark contains the target it is compared with (Critical, CONFIRMED)

**Evidence**
- `backtest_report.py:64` computes `y_naive = [mean(y_true[:i]) for i in 1..n]`. So the naive value for row i−1 includes `y_true[i−1]` itself. Toy check: `y = [1, −1, 3]` gives naive `[1, 0, 1]`.
- The pooled series also leaks the 5 overlapping 6M targets and the other benchmarks' same-date values, which share the PGR leg.
- Area 7 (`t7b_r2_decomp.py`) rebuilt the September quality table exactly and compared benchmarks:

| Naive benchmark | Pooled OOS R² |
|---|---|
| Repo definition | −1.13 % |
| Excluding the current row | +1.2 % |
| Mean of targets realised by t | +10.8 % |
| Campbell–Thompson prevailing mean | +13.9 % |
| Zero | +14.2 % |

**Impact**
- The R² ≥ 2 % gate has failed every month (−6.0 % to −0.1 %). Every look-ahead-free benchmark clears it.
- With F02 also fixed, September would have been evaluated for ACTIONABLE or MONITORING-ONLY rather than DEFER.
- An August flip is SUSPECTED only: the replay consensus differed from the committed report.

**Fix:** set `naive_t` to the mean of targets whose window ended on or before t, including training history, per benchmark. Store the corrected value in `model_performance_log.aggregate_oos_r2`.

**Test that would catch it:** on a simulated overlapping series where the forecast equals the true conditional mean, R² must be > 0, and the naive must never use a target realised after t.

### F05 — VGT ~8-for-1 split missing (High, CONFIRMED in data; ratio SUSPECTED)

**Evidence**
- Orchestrator query: VGT 2026-04-17 = 805.58 and 2026-04-24 = 104.16. Volume went from 3.3M to 17.2M. There is no split row.
- 6M rows 2025-10-31 → 2026-02-27 are stored at +0.77 to +0.91 but are −0.01 to −0.31 corrected.
- 12M rows 2025-04 → 2025-08 are stored at +0.51 to +0.77 but are −0.42 to −0.89 corrected.

**Impact**
- All 10 labels flip. They are the newest training rows for VGT, which has weight 0.20 in the investable classifier pool and Path B, and they affect every report since May 2026.
- VGT is not in the live regression universe.

**Fix and test:** same as F03. Also regenerate the VGT shadow history.

### F06 — FRED lags applied twice, and by row over duplicate month rows (High, CONFIRMED)

**Evidence**
- `fred_loader.py:131` defaults to `apply_publication_lags=True` before storing. It is called that way from `weekly_fetch.py:116`, `monthly_decision.py:309` and `initial_fetch.py:66`.
- `feature_engineering.py:1219-1222` lags again. `duration_rate_shock_3m` (`:717-721`) lags a third time.
- Orchestrator query:
  - VIXCLS stored at 2020-04-30 is 53.54, which is the 2020-03-31 close.
  - GS10 stored at 2020-04-30 is 0.87, the March-2020 average.
  - Rows exist for both 2020-05-29 and 2020-05-31.
- 64 months (2008-05 → 2026-02) are stored twice per macro series, at the business and the calendar month-end. NFCI values differ in all 64.
- `_apply_fred_lags` shifts by rows (`:117-121`).
- Effective lags measured by pushing marker values through the production path (area 3 `c1_fred_tracer.py`):
  - VIX, rates and spreads: 2 months (configured 1).
  - NFCI: 3–4 months (configured 2).
  - Vehicle miles: 4 months (configured 2).

**Impact**
- There is no look-ahead, but inputs are stale. The September decision row carries June VIX, rates and spreads, and April NFCI.
- Lagging once as configured changes 1,778 of 4,275 feature cells. The mean forecast moves from −3.71 % to −1.69 %, and GLD goes from UNDERPERFORM to NEUTRAL.
- The committed v134 lag sweep was run on the pre-lagged DB, so its "lag 0" was really lag 1.
- The freshness line shows "FRED 2026-09-30, 0 days" on a 2026-09-20 run.

**Fix**
- Store raw observation dates.
- Lag once, by calendar period, at feature time.
- Collapse to one row per series-month, and delete the legacy calendar-month-end rows written by `src/research/v19.py`.
- Refetch the history, then re-run v134.

**Test that would catch it:** a mocked series through fetch → store → build must show `feature(M) = raw(M − lag)`, and no (series, month) pair may have more than one row.

### F07 — PGR-specific FRED series never refreshed; a live feature has been silently imputed since April (High, CONFIRMED)

**Evidence**
- Orchestrator query: `PCU5241265241261` last stored 2026-02-28; `CUSR0000SETA02`, `CUSR0000SAM2` and `TRFVOLUSM227NFWA` last stored 2026-03-31.
- Only `initial_fetch.py` fetches `FRED_SERIES_PGR`, and its workflow runs yearly (cron `0 14 25 3 *`).
- `rate_adequacy_gap_yoy` (live GBT) is NaN from the 2026-04-30 row onward. It is imputed to the training median at `wfo_engine.py:409-416`.
- The freshness check uses `MAX(date)` across the whole table (`db_client.py:304-387`). A DB copy with PGR prices deleted after June still reports OK (area 8 `freshness_check.out`).

**Impact**
- One of 13 GBT inputs has been a constant in every live prediction since the May report, and nothing flags it.
- Moving that feature from its p50 to its p95 moves VOO from +3.04 % to +5.61 %.
- Separately, the research series `CUSR0000SETE` ends in 2017, and the Multpl series end 2026-04.

**Fix**
- Fetch `FRED_SERIES_PGR` in the weekly and monthly jobs.
- Check freshness per ticker and per live-feature series.
- Flag or downgrade the run when any live feature is NaN in the decision row.

**Test that would catch it:** a fixture with one stale series must return WARNING, and the decision row must have no NaN live features.

### F08 — Dividend feeds stale since 2026-03-26 (High; staleness CONFIRMED, cause SUSPECTED)

**Evidence**
- The last PGR ex-date is 2026-04-02, so the July 2026 $0.10 dividend is missing.
- `DIVIDENDS/PGR` is logged in `api_request_log` every week, yet `ingestion_metadata` stays at 2026-03-26.
- The same pattern shows on the first peer call (ALL). CB, HIG and TRV, called after 13-second sleeps, update fine.
- ETF dividends are refreshed only by `initial_fetch_dividends.yml`, which runs yearly. `weekly_fetch.py:16-17` claims "quarterly ETF dividend refreshes"; none exist.
- Likely cause: the first dividend call fires straight after the price batch, AV returns an "Information" advisory, and the loader skips it (`multi_dividend_loader.py:208-224`). Confirm by searching the Actions logs for `[av-advisory] … 'PGR'`.

**Impact**
- 162 target rows understate benchmark returns (biased toward PGR, i.e. against selling).
- Worst cases: VCIT 1.95 pp; primary benchmarks BND 1.54, VMBS 1.73, VDE 0.75 and VOO 0.59 pp. All are in the final refit window.
- If the January 2027 PGR annual dividend is missed, every PGR window spanning it loses about 6 pp.

**Fix**
- Sleep before the first call of each batch, and retry advisories with backoff.
- Schedule a monthly ETF dividend refresh.
- Fail the workflow when dividends lag prices by more than one payment interval.

**Test that would catch it:** per ticker, `max(ex_date)` must be at least the last price date minus 1.5 × the ticker's usual payment interval. Also, the first dividend call must follow a sleep (mocked).

### F09 — `pgr_fundamentals_quarterly`: Q4 rows are annual, ROE is 4×, pe/pb are always NULL (Medium, CONFIRMED against EDGAR)

**Evidence**
- 0 of 74 rows have `pe_ratio` or `pb_ratio`.
- Median by quarter:

| Quarter | ROE | EPS |
|---|---|---|
| Mar | 0.199 | 0.43–0.65 |
| Jun | 0.179 | 0.43–0.65 |
| Sep | 0.163 | 0.43–0.65 |
| Dec | 0.736 | 2.16 |

- `_extract_flow_concept` keeps 10-K facts of 320–380 days (`edgar_client.py:197-201`). ROE is computed as `net_income × 4 / equity` (`:369-371`).
- EDGAR check: FY2018 NI $2,615.3M and equity $10,821.8M (10-K `0000080661-19-000008`) give 0.9667, exactly the DB value. PGR reported trailing-12-month ROE of 24.7 % (8-K `0000080661-19-000003`).
- Q1–Q3 are defined differently again: annualised single-quarter NI over ending equity, vs PGR's trailing-12-month NI over average equity. Q3-2017 is 0.0965 vs the reported 16.3 %.
- `.sort_values(["filed", …]).last()` (`:219-220`) keeps the latest comparative, so the value is not point-in-time. Weekend quarter-ends add a third month of lag in 19 of 74 quarters.

**Impact**
- The `roe` feature is 1.325 for 2025-02 → 04 and 1.492 for 2026-02 → 04, against about 0.30–0.39 in other months.
- It is not a live Ridge/GBT feature. It is used by the v128 shadow feature map (BND), `candidate_model_bakeoff`, `v87_utils` and `redeploy_buckets`.

**Fix**
- Derive Q4 as FY minus 9M, as `valuation_multiples._quarterly_eps` already does for EPS.
- Define ROE as trailing-12-month NI over average equity, or use the 8-K `roe_net_income_ttm`.
- Keep the earliest-filed value.
- Drop the NULL columns or populate them.

**Test that would catch it:** a companyfacts fixture containing 10-K annual and 10-Q year-to-date facts must give Q4 EPS = FY − 9M and ROE within [−0.5, 0.6]. The current fixture in `tests/test_edgar_client.py` has no annual facts, so the bug passes.

### F10 — Live GBT `investment_book_yield` mixes percent and fraction (High, CONFIRMED)

**Evidence**
- `scripts/edgar_8k_fetcher.py:816-817` divides by 100 "to match CSV convention", but the CSV stores percent (2024-04 = 3.8).
- The COALESCE upsert keeps both conventions. Orchestrator query: 33 of 40 non-null values since 2023 are below 1 (min 0.03, max 3.9).

**Impact**
- The live value is 0.042, while the GBT training window mixes 0.03–0.04 with 2.6–3.9.
- Fixing moves predictions by up to 1.25 pp (e.g. VOO +3.0 % → +1.8 %), with no signal flips (area 5 `impact_V2.json`).

**Fix:** remove the `/100`, and re-derive the affected rows.

**Test that would catch it:** all non-null values must lie in [0.5, 10].

### F11 — Live GBT `pif_growth_yoy`: leap-year key bug and PIF definition break (High, CONFIRMED)

**Evidence**
- `_prior_year_key` (`:1753-1765`) maps 2025-02-28 to 2024-02-28. The stored key is 2024-02-29, so February YoY is NaN after every leap year. Orchestrator query: 2021-02 and 2025-02 are NaN.
- `pif_total` jumps +13.5 % in 2024-04, dips in 2024-09, then jumps again (property PIF included). Stored growth reads about 0.31 in 2024-12 → 2025-03 against about 0.18 on a consistent definition. 19 rows are affected.
- Also affected: `pif_growth_acceleration`, `npw_per_pif_yoy`, `gainshare_est` and the direct-channel share features.

**Impact**
- 2021-02 and 2025-02 fall in live training windows and get median-imputed.
- Fixing moves the mean forecast from −3.71 % to −2.71 % (area 5 `impact_V3.json`).

**Fix**
- Compute PIF growth in the feature builder from one consistent definition, with calendar-period arithmetic: key on year-month, not the date string.
- Record the definition change.

**Test that would catch it:** February 2025 must be non-NaN, and no month-over-month `pif_total` jump may exceed 5 % without an annotation.

### F12 — Parser drops the sign of parenthesised negatives (Medium, CONFIRMED against EDGAR)

**Evidence**

| Filing | EDGAR shows | DB stores |
|---|---|---|
| Aug-2017 8-K `0000080661-17-000060` | NI **$(16.8)**, EPS **$(0.03)**, pretax (49.5), tax (29.8), realized gains (11.7) | +16.8 / +0.03 / … |
| Oct-2018 8-K `0000080661-18-000052` | NI $(31.7), EPS $(0.06), pretax (69.4), tax (32.3), realized gains (259.9) | all positive |
| Q3-2017 8-K `0000080661-17-000063` | Sep realized gains $(13.9) | +13.9 |
| Q4-2018 8-K `0000080661-19-000003` | Dec realized gains $(330.8) | +330.8 |

- With the signs restored, the monthly net income sums reconcile exactly to XBRL:
  - Q3-2017: 199.3 − 16.8 + 41.5 = **224.0** (10-Q `0000080661-17-000066`).
  - Q4-2018: −31.7 + 242.4 + 54.0 = **264.7** (FY `0000080661-19-000008` minus 9M `0000080661-18-000050`).
  - This explains the reported "+0.05 Q3-2017 / +0.12 Q4-2018" EPS gaps. The review's initial hypothesis of quarter-close true-ups was wrong and is retracted.
- Area 11 reconciled 72 quarters. Only these two quarters show a gap equal to twice the monthly value.
- Q4-2020 (NI −6.4, EPS −0.03) remains unexplained.
- Area 1a scanned every row with accounting identities (revenue − components, revenue − expense vs pretax, OCI vs ΔNUG). The same sign loss appears in:
  - `total_net_realized_gains`: 38 months, 2004-12 → 2020-03;
  - `provision_for_income_taxes`: 7 months;
  - `roe_net_income_ttm`: 2009-04 → 07;
  - `net_unrealized_gains_fixed`: all of 2018;
  - `total_comprehensive_income` / `comprehensive_eps_diluted`: 6 months.
  - Also, `total_comprehensive_income` is about 0 for 2009-08 → 2010-05 (identity implies $70–200M).
- The in-repo `_parse_number` (`scripts/edgar_8k_fetcher.py:472-485`) returns +56.7 for a split-cell "(56.7" and for Unicode "−56.7". The historical rows came from an external extractor that is not in the repo.

**Impact**
- 24 months of trailing-12-month EPS are too high: +$0.06 for 2017-08 → 2018-07 and +$0.12 for 2018-10 → 2019-09. P/E is understated 1.4–2.6 %.
- This affects `pe_ratio` (not live) and `pb_vs_pe`/valuation research (correlations move ≤ 0.004).
- `realized_gain_to_net_income_ratio` has the wrong sign in affected months.

**Fix:** parse `(x)` as −x in the monthly value parser, then re-parse or backfill every row.

**Test that would catch it:** a monthly-vs-XBRL quarterly reconciliation with |ΔNI| < $1M and |ΔEPS| ≤ 0.02, and a revenue − expense = pretax identity check per row.

### F13 — Reported health inflated by in-sample selection (High, CONFIRMED)

**Evidence** (areas 3, 7 and 8)
- **Shrinkage α = 0.50** was chosen on the same OOS history (`results/research/v38_shrinkage.py:103`):

| Configuration | Pooled R² |
|---|---|
| Production (fixed α = 0.50, full-OOS weights) | −1.13 % |
| No shrinkage | −10.58 % |
| Past-folds-only α and weights | −9.16 % (IC 0.127) |
| Prequential α only | −7.6 % |

- **Quality weights** come from the same OOS record they gate (`consensus_shadow.py:45`). In July 2026 the equal-weight mean IC was 0.0686, below the 0.07 gate, and the quality-weighted IC was 0.0993, which passed. The inflation is +0.010 to +0.031 in every month. In 2026-02 and 2026-08 the consensus is NEUTRAL (equal weight) vs UNDERPERFORM (quality weighted).
- **ECE** is computed on the same rows the calibrator was fitted to (`monthly_decision.py:532-571`): 0.0335 in-sample vs 0.121 prequential. The 2026-05 and 2026-06 logged ECE values lie below their own CI lower bounds.
- **Conformal coverage** is in-sample (`conformal.py:217,341`): reported 89.2 %; causal trailing-12 coverage is 41.7 %. The DB column `conformal_empirical_coverage` stores the trailing value.
- **Hit-rate gate:** 66.6 % vs a 68.1 % positive base rate. The model does not beat base rate on 6 of 8 benchmarks.
- **Pooled IC p-value:** 2.5e-5 using Newey-West lag 5 over 1,224 date-sorted rows. Driscoll–Kraay by date gives p = 0.015.

**Impact:** the gates and report tell the reader the model is healthier than an honest estimate. This matters as soon as F02 and F04 are fixed.

**Fix**
- Select α and weights prequentially.
- Gate on equal-weight or pooled IC.
- Gate hit rate on excess over max(p, 1−p) or a Pesaran–Timmermann test.
- Report only prequential ECE and trailing coverage, and cluster significance by date.

**Test that would catch it:** recompute the aggregate metric without any later-fold statistic and assert equality.

### F14 — `--dry-run` mutates the DB and committed artifacts (High, CONFIRMED)

**Evidence** (area 8, run only inside a scratch clone with SMTP and API variables blanked)
- `python scripts/monthly_decision.py --as-of 2026-04-02 --dry-run --skip-fred`, the CI smoke command, changed the clone's DB sha from e7531a… to 86494ab…
  - It overwrote the 2026-04-30 `model_performance_log` row (oos_r2 −0.02626 → −0.03428).
  - It added a `model_retrain_log` row.
  - It rewrote all 10 files in `results/monthly_decisions/2026-04/`, whose manifest says `artifact_classification: production`.
  - It appended to `decision_log.md` and to two shadow ledgers.
- Code paths: `_record_model_health_snapshot` (`:731-779`, called at `:3180` with no `dry_run` guard), `:3576`, and artifact writes at `:3402-3683`. The idempotency check is skipped in dry-run (`:3039`).
- `weekly_fetch.py` logs API requests and seeds splits in dry-run.
- The committed `decision_log.md` already contains 6 `[DRY RUN]` rows (orchestrator grep).

**Impact**
- CI is ephemeral, but the README, runbook and CONTRIBUTING tell people to run this locally. Doing so corrupts production history, including the drift monitor's input.

**Fix**
- Open the DB read-only in dry-run, write to `results/dry_run/`, skip ledgers, and add a `dry_run` flag to the manifest.

**Test that would catch it:** run `main(dry_run=True)` in a temp repo and assert the DB hash and all tracked files are unchanged.

### F15 — Per-share basis mixed across the 2006 split (Medium, CONFIRMED)

These rows are in historical folds only; no contaminated row is in any final refit.

| Feature | Location | Error |
|---|---|---|
| `book_value_per_share_growth_yoy` (live Ridge) | `feature_engineering.py:516-528` | −0.696 to −0.709 for 2006-07 → 2007-06 vs +0.165 to +0.218 corrected; in 5 folds each for GLD and VDE |
| `pb_ratio` | `:1191-1205` | 2006-05 = 0.842 and 2006-06 = 0.782 vs 3.369 and 3.130 corrected; the P/E path (`:1162-1178`) is correct |
| `buyback_yield` | `:745-763` | 4× overstated in 2006-05/06 |
| `pgr_vs_kie_6m`, `pgr_vs_peers_6m`, `pgr_vs_vfh_6m`, `commodity_equity_momentum` | `:1227-1366` | Up to 0.71–1.15, also at the KIE 2017, CB 2006 and VOO 2013 splits |

- Research: x1/x9/x12/x17 compute BVPS and price targets on raw values. The committed `x12_bvps_discontinuities.csv` records the 2006 split as an unexplained −74.9 % capital event.

**Fix:** route every per-share series through `share_basis_factor` before `pct_change` or division, carrying the factor through `_apply_edgar_lag`.

**Test that would catch it:** |Δlog P/B| < 0.3 and |BVPS YoY| < 0.5 across 2006-05.

### F16 — YoY fields span 13 months at the gaps; possible fiscal-month change (Medium, CONFIRMED; 2024 cause SUSPECTED)

**Evidence**
- `load_from_csv` pre-fills row-based `pct_change(12)` (`scripts/edgar_8k_fetcher.py:2430,2442`). `_compute_derived_fields` (`:1813-1838`) overwrites only where a prior-year key exists.
- `npw_growth_yoy`:

| Row | Stored | Compared against | True value (vs EDGAR) | Error |
|---|---|---|---|---|
| 2016-05 | −0.1524 | 2015-04 (2,062.7) | +0.1055 (May-2015 NPW 1,581.4) | −25.8 pp |
| 2020-04 | +0.2718 | 2019-03 | +0.0257 (Apr-2019 NPW 3,669.8) | +24.6 pp |

- `unearned_premium_growth_yoy` has the same two rows wrong.
- In the feature matrix these land at 2016-07 and 2020-06 after the lag.
- Separately, NPE/NPW first-month-of-quarter ratios of 1.13–1.25× through 2023 are flat from 2024. This suggests 5-week fiscal months ended in 2024, making monthly YoY swing between −0.06 and +0.64 in 2024. SUSPECTED; PGR's 2023 10-K would confirm.

**Impact**
- The two gap rows are 2.5-SD errors in a live Ridge feature; they sit in historical folds only.
- Using trailing-12-month NPW growth instead flips GLD to OUTPERFORM (sensitivity only).

**Fix:** compute YoY by calendar period with NaN when the base month is missing, and prefer trailing-12-month or quarterly flow growth.

**Test that would catch it:** a `load_from_csv` fixture with a missing month must return NaN, not a 13-month change, 12 months later.

### F17 — Two monthly releases never ingested; older filings unreachable (Medium, CONFIRMED against EDGAR)

**Evidence**
- The DB has 263 of 265 months. Missing are 2015-05 and 2019-04. Both exist on EDGAR:
  - 8-K `0000080661-15-000034`, filed 2015-06-17, item "9.01" only.
  - 8-K `0000080661-19-000027`, filed 2019-05-15, items "5.07, 7.01, 9.01".
- The item filter keeps only filings listing 7.01 or 2.02 (`scripts/edgar_8k_fetcher.py:184-189`), so the 9.01-only May-2015 release is dropped.
- The fetcher reads `page_data["filings"]["recent"]` from pagination files (`scripts/edgar_8k_fetcher.py:237`). The EDGAR file `CIK0000080661-submissions-001.json` is flat, with top-level `accessionNumber`, `form` and similar keys. It holds 2,000 filings from 2009-02-19 to 2022-05-16, including both missing 8-Ks. So pagination collects nothing, and history before the "recent" block depends entirely on the CSV backfill.
- Forward-filling the gap months duplicates a month inside trailing-12-month windows. `combined_ratio_ttm` is off by 0.45 points in 24 rows, and `investment_income_growth_yoy` at 2015-07 is 0.006 vs 0.126 (area 4 `edgar_gap_truth.py`).

**Fix**
- Read pagination files as flat arrays.
- Backfill the two months (values in Appendix A).
- Compute on a complete calendar grid with NaN for missing months.

**Test that would catch it:** month coverage must be complete after 2004-08, and the pagination parser must accept the flat format.

### F18 — Equity and debt mis-parsed in 11 rows (Medium, CONFIRMED against EDGAR)

**Evidence**
- In 11 rows `shareholders_equity == roe_net_income_ttm` and `debt == debt_to_total_capital`: 2004-12, 2005-01/04/05/08 and 2008-09 → 2009-02 (orchestrator query).
- The Dec-2004 release (`0000950152-05-000341`, full `.txt`) has **no** total-equity line. It shows only "Book value per share $25.73 | Return on average shareholders’ equity 30.0 % | Debt to total capital ratio 19.9 %".
- The FY2004 10-K EX-13 (`0000950152-05-001650`) gives total shareholders’ equity **$5,155.4M**, against 30.0 in the DB.
- The FY2005 10-K (`0000950152-06-001512`) gives $6,107.5M, which matches the DB.
- The Oct-2004 drop to $4,819M is genuine: a 16.9M-share Dutch auction.
- Also: null equity in 2005-02, 2005-03 and 2007-08, and `total_investments` 2004-08 = −2.0.

**Impact:** research and shadow ratio features explode. `pgr_premium_to_surplus` reaches +135,964 and −45,242, `unrealized_gain_pct_equity` 2,213, and `buyback_yield` 9.4.

**Fix**
- Anchor the equity match to "Total shareholders’ equity", or compute it as BVPS × shares.
- Validate each row against BVPS × shares within 1 %, allowing for preferred stock in 2018-03 → 2024-01.
- Winsorise ratio features.

**Test that would catch it:** `abs(equity / (bvps × shares) − 1) < 0.06` for every row.

### F19 — Tax and Monte Carlo logic errors (Medium, CONFIRMED)

These run only when a lot file exists. That means local runs; the 2026-02/03 reports contain them.

- **Breakeven sign inverted** (`capital_gains.py:442-467`).
  - For a fully appreciated lot, holding to LTCG wins unless PGR falls more than 21.25 %, but the code reports 21.25 % as the return needed to hold.
  - A negative relative forecast prints "capital-loss harvesting 37 %" (Aug/Sep 2026). Relative returns are also used as absolute PGR returns (`monthly_decision.py:1184-1239,1283,1294`).
- **Leap years** (`:58`): `days > 365` treats the anniversary as LTCG across Feb 29 (vests 2027-07-17 and 2028-01-19). The "hold to LTCG" date is vest + 366 days (`:524`).
- **Wash sales:** HOLD_FOR_LOSS sells at vest + 180 days (`:615`), which is 1–6 days from the next vest. There is no wash-sale logic anywhere.
- **Other defects:**
  - Lots are sorted by total rather than per-share gain (`:155-157`).
  - The scenario "utility" picks SELL_NOW on a +30 % forecast (`:629-639`).
  - Future (unvested) lots are treated as held.
- **Monte Carlo volatility** (`monte_carlo.py:125-145`) is 0.948 vs about 0.266 correct, because of weekly × √252 plus the split weeks. The 2026-03 report's "P(HOLD beats sell) 30.8 %" would be about 50.9 %.

**Fix:** absolute-return threshold −g·(S−L)/(1−L); `relativedelta(years=1)` plus one day for LTCG; a wash-sale window check against the vest schedule; adjusted weekly returns × √52.

**Test that would catch it:** hand-computed cases for each rule (leap-year anniversary, wash-sale window, a fully appreciated breakeven).

### F20 — ACTIONABLE mapping unvalidated; missing CPCV fails open (Medium, CONFIRMED)

- Area 8 backtested the exact live mapping over 168 OOS dates (2012-03 → 2026-02), assuming the gate passes:

| Policy | Mean relative return per decision |
|---|---|
| Live ACTIONABLE mapping | +3.53 % |
| Always sell 50 % | +4.14 % |
| Always hold | +8.28 % |

- An OUTPERFORM consensus with prediction ≤ 5 % sells 75 %, which is more than the neutral 50 %.
- The report's "Decision Policy Backtest" evaluates a different policy (`tiered_25_50_100`).
- A CPCV exception gives None/UNKNOWN (`monthly_decision.py:406-410`), which passes the gate (`decision_rendering.py:42,50`). R² 3 %, IC 0.10, hit 60 % and CPCV None gives ACTIONABLE, sell 100 %.

**Fix**
- Backtest the exact live function before promotion.
- Never sell more than the default on a bullish signal.
- Treat UNKNOWN as DEFER.

**Test that would catch it:** a policy-regression fixture requiring uplift ≥ 0; CPCV None must not yield ACTIONABLE.

### F21 — Confidence is a constant (Medium, CONFIRMED)

- `prediction_std` comes only from BayesianRidge, which has been retired (`multi_benchmark_wfo.py:310-366`).
- All 8 months × 8 benchmarks have `prob_outperform = 0.5`, `prediction_std = 0` and tier LOW.
- The report text still cites "BayesianRidge posteriors" (`:1830`).
- Calibrated probabilities often contradict the forecast sign (VWO −3.98 % with P = 85.6 %). They are hidden in the redeploy table but still size allocations (`redeploy_portfolio.py:200-203`).

**Fix:** derive uncertainty from OOS residuals or drop the field.

**Test that would catch it:** tier values across a run must not all be identical.

### F22 — Target windows and as-of truncation (Medium, CONFIRMED)

- **Windows:** end prices are the last weekly close on or before t + DateOffset (`multi_total_return.py:115-127`).
  - 6M windows run 175–190 days, and 96 of 316 end one week early.
  - Compared with month-end-aligned windows the mean absolute difference is 0.010–0.022, up to 0.32, with 2–11 sign flips per benchmark.
- **Partial-week duplicates:** 31 ticker-weeks have two bars: 2026-03-24 Tuesday (18 tickers), 2026-03-30 (the peers) and 2026-04-30 Thursday (9 ETFs). PGR and benchmark windows then end on different dates in 18 rows (errors 0.17–1.65 pp).
- **As-of truncation** (`feature_engineering.py:1454-1473`) cuts at the as-of month-end minus h. So 127 of 129 backdated monthly runs keep one target that ends 1–11 days after the as-of date. `tests/test_asof_target_truncation.py:19` asserts this leaky behaviour. Live runs are unaffected.
- **Backtest timing:** the realised window is ~1 month offset from the model target (`backtest_engine.py:229-240`). SUSPECTED.

**Fix**
- End windows at the last bar on or before `BMonthEnd(t + h)`.
- Keep at most one bar per ticker-ISO-week.
- Truncate targets on the window end date ≤ as-of.

**Test that would catch it:** one row per ticker-week; every retained target's end date ≤ as-of.

### F23 — Fixed 2-month EDGAR lag (Medium, CONFIRMED)

- Actual `filing_date − month_end` is 9–29 days (median 15). None exceed 61 days, so there is no look-ahead.
- Every one of 263 rows lands one month later than necessary. For example, January 2024 data filed 2024-02-14 enters the 2024-03-29 row.
- Placing rows by actual filing date changes 1,208 feature cells and flips the September consensus from NEUTRAL to UNDERPERFORM (VWO joins; 4 of 8 benchmarks). The mode stays DEFER.
- The v142 study chose lag 2 by in-sample R² among lags 0–3. Lag 0 is itself a look-ahead, so an availability rule was treated as a tunable setting.
- The `config/features.py:235` comment says 8-Ks are "filed within the same month". All 263 were filed the following month.

**Fix:** place each row at the first decision date on or after `filing_date` (the column exists).

**Test that would catch it:** every EDGAR row's first feature row is ≥ `filing_date` and within one month of it.

### F24 — Shadow-layer defects (Medium, CONFIRMED)

- **Path B stale row:** `fit_path_b_classifier` scores `iloc[-1]` of the target-joined frame, i.e. 2026-02-27 (in-sample). The reported 0.4825 should be 0.5805 on the 2026-08-31 row. There is also no scaler before the C = 0.5 logistic.
- **Monitoring never matures:** `is_horizon_mature` is set once at creation and never updated. `diagnostic.md` will show "Matured observations 0" forever, blocking the governance promotion rule.
- **Veto gate:** penalises hold-leaning ACTIONABLE months (`classification_gate_overlay.py:108-112`). "Aligned" ignores direction.
- **TA features** (`v160_ta_features.py`): 126/63-row windows on weekly unadjusted OHLCV. `ta_pgr_natr_63d` is 0.195 vs 0.037 at 2006-05, and `ta_ratio_roc_6m_vwo` errors reach 1.46.

**Fix:** pass the current row explicitly; recompute maturity at attach time; use a direction-aware veto; apply the F01 treatment to TA features.

### F25 — Research conclusions and methods (Medium, CONFIRMED)

Reproducibility is good: `pgr_valuation_monthly.csv` and all `results/research/pb_vs_pe/*` regenerate byte-identically. The problems are in the conclusions and methods.

- **P/B vs P/E OOS R² comes from 2023+.** roe_gap vs market is +0.067 over the full sample, −0.006 before 2023 and +0.184 from 2023. Clark–West p = 0.069.
- **Inference ignores overlap and multiple testing.** The Hodrick-1B t for roe_gap is −1.40 vs the committed NW −2.60. Rotating 10 signals × 2 targets gives |IC| ≥ 0.463 in 27.5 % of shifts. About 494 combinations were tried, and roe_gap was added after the first results.
- **"P/B predicted nothing" is a pooling artefact.** The within-era IC is +0.551 (2004–14) and +0.220 (2015+).
- **HTML summary details are wrong:** "today" uses the April S&P P/E, and the tercile labels are wrong.
- **x-series targets** use raw price `shift(-h)` across splits, ignore dividends, and mix ME and BME dates (`x1_targets.py:24-95`). `build_special_dividend_targets` misses December specials.
- **v39–v59 tune Ridge with `RidgeCV(cv=None)`**, which is leave-one-out and violates AGENTS.md (`v39_ridge_alpha.py:63`). The chosen α is 3.7 vs 110 under the production time-series inner CV.
- **Research holdout contamination:** 48 target rows cross the holdout start (`v37_utils.py:118`).

**Fix:** per-period and Clark–West reporting; Hodrick or non-overlapping inference with family-wise correction; DRIP-based BME x-targets; a lint test forbidding `cv=None`.

### F26 — CI, workflows and config risks (Medium; mostly CONFIRMED)

- **Serialisation:** DB-writing workflows each have their own concurrency group and push straight to master. The 8-K job `pull --rebase`s a binary DB, and scheduler lag of up to about 3 hours makes overlaps real. The docs (`docs/workflows.md:102-106`) claim otherwise.
- **"One-time" bootstraps** still fire every March (`initial_fetch_prices.yml:16`, `initial_fetch_dividends.yml:19`, `post_initial_bootstrap.yml:43`).
  - 2027-03-25 collides with the 8-K job, and 2027-03-26 exceeds the AV 25/day budget.
  - `--force` is ignored (`initial_fetch.py:87`).
  - `peer_bootstrap.yml:108` queries a nonexistent `price_date` column.
  - `post_initial_bootstrap.yml:108` runs `git add results/ || true`.
- **EDGAR access:** no workflow sets `EDGAR_USER_AGENT`, so SEC requests use `contact@example.com` (`config/api.py:34-41`). The CI "dry run" of `edgar_8k_fetcher.py` still downloads the submissions list and filings.
- **Duplicate emails:** the email step runs even when `_already_ran` short-circuits, so the email is probably re-sent on the 21st/22nd (SUSPECTED; check the Actions logs).
- **Mode fallback:** an invalid `RECOMMENDATION_LAYER_MODE` (e.g. `live-only`) falls back to the retired `shadow_promoted` mode (`config/model.py:163-171`, `monthly_decision.py:3027-3033`).
- **As-of on weekends:** weekend runs produce an as-of later than the run date (June and September 2026).
- **Secrets:** none are committed, and `.env` is ignored. Query-string API keys appear in logged HTTP errors locally. Actions are pinned by tag, and `ci.yml` has no `permissions:` block.

**Fix:** one shared `db-writer` concurrency group, with the monthly decision run after the 8-K fetch; dispatch-only bootstraps; a required EDGAR User-Agent; fail fast on an unknown mode; gate the email on a `generated=true` output.

### F27 — Other data-integrity items (Medium, CONFIRMED)

- `fred_macro_monthly` has duplicate business/calendar month-end rows (64 per macro series; the NFCI copies disagree). Values are latest-vintage, not point-in-time (NFCI, CPIs, PPIs and VMT are revision-prone; SUSPECTED magnitude).
- Aggregation is mixed: GS2/5/10 are monthly averages, while T10YIE, VIX and the spreads are month-end values. So `real_rate_10y` subtracts a point value from an average.
- Peer CB is spliced: old Chubb before 2006-12, ACE-like levels after, and 8 dividends a year until 2015. TRV has 3 overlapping dividends in 2004.
- FZROX has 895 orphan VTI-proxy rows. `proxy_fill` is hard-coded to 0 (`multi_total_return.py:192`).
- `fetch_status.md` mislabels real fetches as "SKIPPED". A missing 2026-08-07 weekly run went unnoticed.

### F28 — Test suite gaps (Medium, CONFIRMED)

- The suite passes (section 1), but no test uses weekly bars, splits with a matching price drop, missing EDGAR months, duplicate FRED months, leap-year keys, or 10-K annual XBRL facts. So F01, F03, F06, F09, F11, F16 and F17 all pass CI.
- `tests/test_total_return.py`'s split test uses flat $100 prices, so it expects a +100 % "return" at the split.
- Weak or vacuous tests:
  - `tests/test_asof_target_truncation.py:19` asserts the leaky behaviour (F22).
  - `tests/test_fracdiff.py:84-108` passes vacuously: every candidate is skipped on monthly-length series.
  - `tests/test_valuation_multiples.py::test_availability_date_covers_fills_in_ttm_window` cannot fail. Mutating `valuation_multiples.py:301` leaves all 12 tests passing.
- **Repo writes:** 50 tests write `data/processed/feature_matrix.parquet` into the working tree (`feature_engineering.py:1059`) because none patches `_PROCESSED_PATH`.
- **Real-DB opens:** 4 research tests open the real `data/pgr_financials.db` (`test_research_v154…`, `v155…`, `v156…`). The clone's DB was not modified.
- **Mutation study** (area 9, `mutate.py`, clone only). **16 of 18 mutations survived** the targeted tests; only the relative-return sign flip and LTCG `>=365` were caught. Survivors include:
  - removing the EDGAR lag in `build_feature_matrix_from_db` (`:1140`) and in the ROE path (`:1185`);
  - `combined_ratio_ttm` window 12 → 3 (`:317`);
  - BVPS YoY `pct_change(12)` → `(1)` (`:525`);
  - DRIP `new_shares = evt_value/div_price` (`total_return.py:112`);
  - dropping all splits from targets (`multi_total_return.py:64`);
  - `max_train_size=None` and the full-history refit (`wfo_engine.py:273,403`);
  - CPCV `purged_size=0` (`:592`);
  - swapping the consensus λ-mix and removing the IC clip (`consensus_shadow.py:39,46,112`);
  - UNDERPERFORM selling 75 % (`decision_rendering.py:29`);
  - dropping the CPCV FAIL gate (`:50`);
  - removing the conformal finite-sample correction (`conformal.py:105`);
  - flipping the ACI update sign (`:222`).
- **Ineffective property tests:** `test_property_wfo_temporal.py` and `test_property_return_calculations.py` never call production code; parts of `test_property_feature_engineering.py` and `test_property_tax_boundaries.py` are tautologies.
- **Mirror tests:** `test_no_future_leakage_in_momentum` computes its expected value with the same `shift(252)` as the code.
- **Stored-artifact tests:** about 45 research test files only read committed CSVs.
- **Weak fixtures:** the companyfacts fixture asserts Q4 = full year (`tests/test_edgar_client.py:141-147`); `tests/test_capital_gains.py:237-248` and `test_three_scenario_tax.py:131` lock in the leap-year LTCG error.

**Fix:** one fixture set per data defect class (weekly+split series, gapped EDGAR months, duplicated FRED months, leap-year months, companyfacts with annual facts). Invert the as-of assertion. Monkeypatch or remove the parquet side effect.

### F29 — Repository hygiene (Low, CONFIRMED)

- **`.gitignore:15-21`:** `data/processed/` is ignored as a directory, so the `!data/processed/*.csv` negations cannot work. `git check-ignore -v` reports `.gitignore:16` for every CSV, and a scratch-repo simulation confirmed a new `pgr_edgar_cache.csv` cannot be added. Use `data/processed/*` followed by the negations.
- **WAL mode:** the committed DB header is in WAL mode (bytes 18–19 = 2), so any read creates `-wal`/`-shm` sidecars in `data/`, which are not ignored. Checkpoint and set `journal_mode=DELETE` before committing, and ignore `*.db-wal` and `*.db-shm`.
- **`pytest.ini`:** `addopts` already has `-q`, and CI runs `pytest -q`, which gives `-qq`. That suppresses pytest's final summary line in CI logs. The full suite runs in about 5.5 minutes, not more than 15.
- **Unpinned dependencies:** pandas ≥ 2.1 now resolves to 3.0.6. `pct_change` without `fill_method` differs between versions: 6 `pif_total` rows at `src/ingestion/edgar_8k_fetcher.py:505` and `pgr_monthly_loader.py:150`, neither on a production path. Pandas 3 needs Python 3.11+, while the README and AGENTS say 3.10+. Add a lock or constraints for runtime deps.
- **Data dictionary** (`docs/PGR_EDGAR_CACHE_DATA_DICTIONARY.md`):
  - It matches the CSV (256 rows to 2026-01), but the DB has 263 rows to 2026-08.
  - The gaps, the 8 DB-only derived columns and the column renames are undocumented.
  - Share counts are in millions, not thousands.
  - It wrongly says no candidate features are live.
- **CSV cache:** `data/processed/pgr_edgar_cache.csv` ends 2026-01, seven months behind the DB. Re-running `load_from_csv` would recompute the derived YoY fields with row-based windows.
- **Stale docs:**
  - `operations-runbook.md:62` names `build_db_health_report`; the real function is `get_db_health_report`.
  - `troubleshooting.md:17-18` names the wrong email secrets.
  - The CHANGELOG stops at v170; PRs #116–#118 are missing.
  - ROADMAP says "combined_ratio NULL for all quarterly rows"; there are 0 NULLs.
  - `backlog.md` has mojibake.
  - The v2 plan still references daily prices.
- **`claude.md` is lowercase:** Claude Code looks for `CLAUDE.md`.
- **Unused config:** `EDGAR_BASE_URL`, `EDGAR_PGR_CIK`, `ETF_LAUNCH_DATES`, `ETF_PROXY_MAP` and `BL_USE_BAYESIAN_VARIANCE`. `BLP_*` is sized for 4 models, and `FRED_SERIES_LAGS` lists the removed `CUSR0000SETC01`.
- **PEP 8:** AGENTS.md requires strict PEP 8, but `ruff` only checks fatal errors; `ruff --select E,W` reports 869 violations.

### F30 — Production depends on code and caches under `results/` (Low, CONFIRMED)

- The import chain is `scripts/monthly_decision.py:84` → `classification_shadow` → `src/research/v66_utils.py:15` → `results/research/v46_classification.py`.
  - That import installs global warning filters, silencing ConvergenceWarning and All-NaN warnings.
  - It also calls `sys.stdout.reconfigure` and `sys.path.insert`.
- `src/models/v129_feature_map.py` reads `results/research/v128_benchmark_feature_map.csv` at runtime.
- Sixteen `scripts/research/x*` scripts read the gitignored `data/processed/feature_matrix.parquet`, which is written as a side effect of every build.
  - Rerunning x1/x12 on a fresh cache changes x12's model winners.
  - The committed x1 used a matrix ending 2026-04-30.

**Fix:** move `compute_binary_metrics` into `src/models`; remove import-time side effects; pin research inputs by as-of date with a provenance file (DB hash, code hash, max date).

### F31 — Unused or ineffective model utilities (Low, CONFIRMED)

- **Fractional differencing** (`feature_engineering.py:1480-1669`) needs 927–4,076 lags at the 1e-5 threshold, so on 60–323 monthly points it outputs a single value. d is also selected on the full sample.
- **`BLPModel.fit`** returns identical parameters for y and 1−y (`blp.py:194`).
- **Black–Litterman diagnostic** has failed in 8 of 8 months (views have the wrong sign and horizon).
- **Drift monitor's "rolling 12M IC"** averages full-history ICs (`drift_monitor.py:45`), so it cannot detect drift. The retrain trigger has never fired in 18 rows.
- **Feature-importance section** looks up `("historical_mean", "VTI")` and silently falls back to VOO ridge (`monthly_decision.py:3494`).

### F32 — Repository structure (Low, CONFIRMED)

See section 5.

### F33 — EDGAR monthly history rewritten without provenance (High, CONFIRMED)

**Evidence**
- `db_client.py:709-712` overwrites `filing_date`, `filing_type` and `accession_number` unconditionally. `:713-780` COALESCEs every value column, so the latest parse wins column by column.
- `monthly_8k_fetch.yml` re-parses the last 24 months on the 20th and 25th and commits the DB.
- Rows mix sources:
  - 2023-04 → 09 carry the CSV accession but live-parsed FTE returns and fractional book yield;
  - 2024-04 → 2026-01 carry a live accession but the CSV's ROE.
- Accession formats are mixed: 235 dashed, 28 undashed (orchestrator query: `000008066125000126`).
- Re-running `load_from_csv` on a copy, as the runbook documents and `db_client.py:235-239` recommends, changes 77 cells:
  - it restores the 2025-09 combined ratio;
  - it fixes some book-yield rows;
  - it corrupts 7 `pif_growth_yoy` rows and changes 18 gainshare values.

**Impact:** no reproducible point-in-time history. Backtests use whichever parse ran last. Neither restatements nor parser regressions can be audited.

**Fix:** an append-only raw table keyed by (accession, field) with parser version and fetched-at; a view exposing first-reported and latest values; derived fields recomputed over the full table; no CSV re-seed over newer rows.

**Test that would catch it:** each row's accession matches the source of every column, and `load_from_csv` is idempotent against a live-populated DB.

### F34 — 2025-09 combined ratio overwritten by the live parser (Medium, CONFIRMED by identity; EDGAR confirmation pending)

**Evidence**
- The DB has CR 88.7 and expense ratio 23.0; the CSV has 100.4 and 34.7 (orchestrator query).
- The DB row contradicts itself: `total_expenses` includes a $1,000M policyholder-credit line, and net income is $305M vs about $1,100M in neighbouring months.
- The CSV convention includes such credits: 2020-04/05 expense ratios are 35.4 and 38.6.
- Accession `0000080661-25-000126` would confirm this; it was not fetched, because it is outside the permitted EDGAR scope.

**Impact:** live-Ridge `combined_ratio_ttm` is understated by 0.975 (0.23 σ) in feature months 2025-11 → 2026-10, including the current row. Mean prediction moves ±0.36 pp.

**Fix:** restore 100.4, and validate CR against (LAE + acquisition + other underwriting + other expense) / NPE within 2 points (`scripts/edgar_8k_fetcher.py:937-947,1087-1092`).

**Test that would catch it:** the same identity check on every row.

### F35 — Live fetch never writes `roe_net_income_ttm` (Medium, CONFIRMED)

**Evidence**
- The parser emits the key `roe_net_income_trailing_12m` (`scripts/edgar_8k_fetcher.py:988,1728`), but the upsert reads `roe_net_income_ttm` (`db_client.py:831`). Only the CSV loader maps the name (`:2387`).
- Orchestrator query: NULL for 2026-02 → 2026-08; 2026-01 = 35.1.

**Impact:** `roe_net_income_ttm` and `roe_trend` are median-imputed at the end of the series. Consumers are the elasticnet and bayesian_ridge override lists, `cross_check` and research; none is a primary live model.

**Fix:** map the key in the upsert.

**Test that would catch it:** parse → upsert → read back every field.

## 4. Prioritised fix plan (independent work packages)

Each package can be done in its own session. Dependencies are noted. Re-baseline research (WP11) only after the data packages land.

| WP | Title | Findings | Depends on | Notes |
|---|---|---|---|---|
| WP1 | Split ingestion and target rebuild | F03, F05, F22 (dup bars, windows) | — | Canonical split source, VOO/VGT rows, jump guard test, one-bar-per-week upsert, BME-aligned windows, rebuild `monthly_relative_returns` |
| WP2 | Price-feature rewrite | F01, F15, F19 (MC vol), F24 (TA) | WP1 | Split-adjusted close helper; calendar-month momentum; 13-week vol × √52; per-share basis for BVPS and P/B; frequency assertion; rename `daily_prices` |
| WP3 | FRED pipeline | F06, F07, F27 (FRED part) | — | Store raw; lag once by period; dedupe month rows; fetch `FRED_SERIES_PGR` weekly; per-series freshness; NaN-in-live-row guard |
| WP4 | Dividend feeds | F08 | — | Sleep and backoff, ETF monthly refresh, freshness gate; then rerun the WP1 target build |
| WP5 | EDGAR monthly parser and table repair | F10, F11, F12, F16, F17, F18, F33, F34, F35, F29 (CSV/dictionary) | — | Negative parsing, equity/debt anchors, book-yield units, leap-year key, PIF definition, calendar YoY, flat pagination, backfill 2015-05/2019-04, re-sync CSV and dictionary |
| WP6 | Quarterly XBRL fundamentals | F09 | — | Q4 = FY − 9M, TTM/avg-equity ROE, earliest-filed values, drop NULL columns |
| WP7 | Validation and gating | F02, F04, F13, F20 (CPCV None), F21 | best after WP2–WP5 for re-baselining | Remove or fix CPCV; look-ahead-free naive; honest hit-rate gate; prequential weights, α, calibration and conformal; clustered significance |
| WP8 | Decision and tax layer | F14, F19, F20 (mapping), F24, F26 (email, mode) | WP7 for the mapping backtest | Read-only dry-run; tax fixes; policy backtest of the live mapping; Path B and monitoring fixes; email gating |
| WP9 | CI, workflows and hygiene | F26, F29 | — | Shared DB concurrency; dispatch-only bootstraps; EDGAR User-Agent; `.gitignore`/WAL; dependency pins; `pytest.ini`; docs corrections |
| WP10 | Timing | F23, F22 (as-of) | WP5 | Filing-date placement; as-of truncation on window end; never set as-of later than today |
| WP11 | Research re-baseline | F25, F30, re-run v134/v142/v38/v128 | WP1–WP6, WP10 | Honest inference; x-series on DRIP series; provenance files |
| WP12 | Test hardening | F28 | alongside each WP | Defect-class fixtures; invert leaky asserts; remove repo writes |
| WP13 | Repository restructure | F32 | Phase 0 anytime; later phases after WP7/WP8 | See section 5 |

Suggested order: WP1 → WP4 → WP3 → WP5/WP6 (in parallel) → WP2 → WP10 → WP7 → WP8. WP9 and WP12 run throughout. WP11 and WP13 come last. Expect the reported OOS metrics, and possibly the recommendation mode, to change once WP7 lands. Document the new baseline in `docs/model-governance.md`.

## 5. Repository structure review and tidy-up plan

### What is there today (tracked files at c948d0f)

| Top level | Files | Size | Role today |
|---|---:|---:|---|
| `src/` | 134 | 1.2 MB | Library code: 10 sub-packages, including `src/research` (55 modules) |
| `scripts/` | 53 | 0.7 MB | Production CLIs, one-off experiments, and `scripts/research/` (25 x-series) |
| `results/` | 672 | 27.3 MB | Production artifacts, research outputs, 86 `.py` files, charts |
| `tests/` | 229 | 1.2 MB | Flat folder, three naming schemes |
| `docs/` | 193 | 2.2 MB | Operator docs plus five parallel history trees |
| `archive/` | 18 | 0.3 MB | Retired scripts and tests (separate from `docs/archive/`) |
| `config/` | 5 | 37 KB | Python package re-exported with `from .x import *` |
| `data/` | 4 | 5.0 MB | Committed SQLite DB (WAL mode), two reference CSVs, a run log (`fetch_status.md`) |
| `dashboard/` | 3 | 26 KB | Streamlit app |
| root | 13 files | | README, CHANGELOG (78 KB), ROADMAP, CONTRIBUTING, AGENTS.md and `claude.md` (identical), 4 requirements/constraints files, `pytest.ini`, `mypy.ini`, `ruff.toml` |

### Where the methodologies collide

1. **Four ways to file a research result:**
   - top-level version folders `results/v9/ … results/v28/` (17 folders);
   - flat version-prefixed files in `results/research/` (`v37_* … v165_*`, `x1_* … x24_*`, `bl01_*`; about 150 study prefixes, 357 files in one directory);
   - a per-study sub-folder (`results/research/pb_vs_pe/`, the newest style);
   - summaries in `docs/results/V9..V29_RESULTS_SUMMARY.md`.
2. **Code lives in `results/`.** There are 86 Python files there, and `results/research/__init__.py` makes it an importable package. 31 files in `tests/`, `src/` and `scripts/` import from it, and production does too (F30). Output folders are therefore not safe to regenerate or prune.
3. **Production writes into research and versioned paths.**
   - `monthly_decision.yml:107` commits `results/research/pgr_*.png`.
   - Shadow reviews accumulate under `results/v14/shadow_reviews/<date>.md`.
   - `post_initial_bootstrap.yml:108` runs `git add results/ || true`.
4. **Version numbers are used as module names:** `src/research/v11.py … v29.py`, `v37_utils.py … v160_ta_features.py`, `x1_targets.py … x24_indicator_contract.py`.
   - 9 are two-line `from X import *` shims: `benchmark_sets`, `diversification`, `evaluation`, `policy_metrics`, `v11`, `v12`, `v22`, `v27` and `v29`.
   - 39 files import them; `src.research.v11` has zero importers.
5. **Scripts serve three audiences.**
   - 9 of 53 are production or ops entry points referenced by workflows: `weekly_fetch`, `peer_fetch`, `edgar_8k_fetcher`, `monthly_decision`, `verify_monthly_outputs`, `initial_fetch`, `bootstrap`, `capital_return_charts`, `repurchase_timeseries_charts`.
   - 18 top-level scripts are manual utilities or experiments that no workflow calls, e.g. `feature_ablation.py`, `candidate_model_bakeoff.py`, `v27_redeploy_portfolio_study.py`, `migrate_v1_to_v2.py`.
   - 25 are in `scripts/research/`.
   - `scripts/edgar_8k_fetcher.py` (2,584 lines, live) and `src/ingestion/edgar_8k_fetcher.py` (a diverged, test-only copy) share a name.
   - `scripts/monthly_decision.py` is a 3,715-line module exempted from mypy.
6. **The code is not an installable package.** There is no `pyproject.toml` or `setup.py`, and 214 files manipulate `sys.path` (82 in `results/`, 68 in `tests/`, 50 in `scripts/`, 14 in `archive/`). Tool config is spread across five files.
7. **Plans and history live in five-plus places:** `docs/plans/` (v8–v34), `docs/superpowers/plans/` (v37+), `docs/closeouts/`, `docs/results/`, and `docs/archive/` (plus `history/`). Add to that `ROADMAP.md`, `docs/research/backlog.md` and a 78 KB `CHANGELOG.md`. Root `archive/` is yet another "old stuff" location.
8. **Tests are flat, with three naming schemes:** 125 `test_<topic>.py`, 75 `test_research_vNNN_*.py` and 27 `test_vNN_*.py`. Production and research tests run together; research harness tests dominate the slowest 25.
9. **`data/` mixes kinds of things:** a binary DB, reference CSVs force-added past an ineffective ignore rule, and an append-only run log written by workflows.
10. **Large regenerable files are committed**, adding up to more than 20 MB of history:
    - `results/v9/classifier_feature_selection_detail_20260403.csv` (9.7 MB);
    - `results/research/v128_regularized_selection_detail.csv` (2.7 MB);
    - `v162_ta_broad_screen_detail.csv` (2.2 MB).

### Target layout (recommended)

```text
pgr-vesting-decision-support/
├── pyproject.toml            # package metadata, deps (+ constraints/lock), pytest/ruff/mypy config
├── README.md  CHANGELOG.md  ROADMAP.md  CONTRIBUTING.md  AGENTS.md  CLAUDE.md (points to AGENTS.md)
├── src/pgr_vds/              # one installable package (pip install -e .)
│   ├── config/               # was top-level config/
│   ├── ingestion/ database/ processing/ models/ portfolio/ tax/ reporting/ backtest/ visualization/
│   ├── decision/             # logic split out of scripts/monthly_decision.py
│   └── research_lib/         # reusable research helpers with descriptive names (no vNNN module names)
├── cli/                      # production + ops entry points only (thin wrappers)
│   ├── weekly_fetch.py  peer_fetch.py  edgar_monthly_fetch.py  monthly_decision.py
│   └── verify_monthly_outputs.py  initial_fetch.py  bootstrap.py  charts.py
├── research/
│   ├── README.md             # study index: id, question, date, data vintage (DB sha256), status, promoted-to
│   ├── studies/<id>_<slug>/  # e.g. v128_feature_search/, x15_pb_regime_overlay/, pb_vs_pe/
│   │   ├── run.py            # from scripts/research, results/research/*.py, scripts/*experiments*
│   │   ├── outputs/          # summary CSV/MD/JSON only; *_detail.csv not committed
│   │   └── README.md         # question, command, conclusion, link to closeout
│   └── legacy/               # results/v9..v28, read-only as-is
├── artifacts/                # production outputs only (written by workflows)
│   ├── monthly_decisions/YYYY-MM/
│   ├── shadow_reviews/       # was results/v14/shadow_reviews
│   ├── charts/               # was results/research/pgr_*.png
│   └── ops/                  # was data/fetch_status.md
├── data/                     # DB + reference CSVs + README (sources, vintage)
├── docs/
│   ├── (operator docs)       # architecture, runbook, workflows, governance, data dictionary, output guide
│   ├── decisions/            # one file per promotion decision (ADR style)
│   ├── reviews/              # this report and future audits
│   └── history/              # plans/, superpowers/, closeouts/, results/, archive/ under one index
└── tests/
    ├── unit/  integration/   # mirror src/pgr_vds
    └── research/             # study tests, marker `research`, separate CI job
```

### Migration plan (phased; each phase ships on its own)

- **Phase 0: safety nets before any file move.** These address why the 2026-04-19 hygiene review deferred moves.
  - Add `pyproject.toml` and `pip install -e .`.
  - Add a link checker for active docs.
  - Add an import smoke test that imports every production entry point.
  - Add a CI rule that fails on new `sys.path` edits outside `tests/conftest.py`.
  - Record the repo tree in a manifest so moves can be verified.
- **Phase 1: separate production artifacts.** Small change, high value.
  - Move `results/monthly_decisions/`, `results/v14/shadow_reviews/`, the monthly `pgr_*.png` charts and `data/fetch_status.md` under `artifacts/`.
  - Centralise these paths as constants in `config`.
  - Narrow workflow `git add` to exact paths; drop `git add results/ || true`.
  - Update `docs/workflows.md` and `docs/artifact-policy.md`.
- **Phase 2: de-version the library.**
  - Delete `src/research/v11.py` (no importers).
  - Rewrite the 39 importing files to use the real modules, then delete the other 8 shims.
  - Move `results/research/*.py` files that production or tests import (e.g. `v46_classification.py`) into `src/pgr_vds/research_lib/` with descriptive names, and remove their import-time side effects.
- **Phase 3: one folder per study.**
  - `git mv` each study's script and outputs into `research/studies/<id>_<slug>/`. This preserves history.
  - Generate `research/README.md` from a small registry (id, date, question, promoted?).
  - Stop committing multi-MB `*_detail.csv` files: write them to a gitignored `outputs/detail/`, or use Git LFS if they must be kept.
- **Phase 4: docs and tests.**
  - Merge the five history trees into `docs/history/` behind one index; the Phase 0 link checker makes this safe.
  - Split `tests/` into `unit/`, `integration/` and `research/`, and run research tests in a separate CI job.
  - Replace `claude.md` with a `CLAUDE.md` that points to `AGENTS.md`.
  - Consolidate `pytest.ini`, `mypy.ini`, `ruff.toml` and the requirements files into `pyproject.toml`.
- **Phase 5: split the monoliths.**
  - Break `scripts/monthly_decision.py` into `src/pgr_vds/decision/` modules behind a thin CLI, and remove its mypy exemption.
  - Merge the two `edgar_8k_fetcher.py` files into `src/pgr_vds/ingestion/edgar_monthly/` (fetch, parse, derive, load), with the CSV loader as one function.

Naming rules to adopt going forward:
- Module names are descriptive, with no version numbers.
- Study IDs appear only in `research/studies/<id>_<slug>/` and in the study registry.
- Production paths never contain a version or study ID.
- Every generated folder has a README naming the workflow or script that writes it.

## 6. What was not reviewed, and why

- **Interrupted agents.** All 12 agents were paused twice by an API spend limit and then resumed.
  - Area 9 ran 18 targeted mutations, not the full transitive test closure, and covered about 100 research test files only by pattern.
  - Area 1a did not check the signs of `fte_return_*` (no identity exists) and did not test the live HTML parser against real exhibits.
- **Mutation coverage of other modules:** dashboard, email, backtest, Black–Litterman and Kelly were not mutation-tested.
- **External confirmations not obtained** (no network access beyond the permitted EDGAR values):
  - issuer notices for the VOO and VGT splits (dates and ratios);
  - ALFRED vintages (FRED revisions);
  - Alpha Vantage responses behind the dividend failures;
  - GitHub Actions logs (duplicate emails, rejected pushes);
  - PGR's 2024 fiscal-calendar change (F16);
  - the Q4-2020 NI/EPS gap.
- **Full backtest reruns** after fixes, and a combined "all fixes" production replay. Each fix was replayed individually. The combined feature matrix was built, but the model was not run on it, because of CPU contention and the spend limit.
- **Areas reviewed lightly or not at all:**
  - the Streamlit dashboard and email HTML (spot checks only);
  - `drift_analyzer.py` and the `redeploy_buckets` ranking;
  - `archive/`;
  - exhaustive grep of `results/research/*.py` for K-fold, shuffle or full-sample scaling beyond the v39–v59 RidgeCV finding;
  - the v-series studies behind live settings, other than v38 (F13), v134 (F06) and v142 (F23).
- **Git history of the switch to weekly prices.** The clone is shallow (grafted at `d080355`, 2026-05-22), and fetching was not permitted.

---

## Appendix A — EDGAR sources used (User-Agent "Jeff Hester jeffrey.r.hester@gmail.com", cached, ≤ 4 req/s)

| Value verified | Accession | URL |
|---|---|---|
| May-2015 monthly results (missing month): NPW 1,581.4; NPE 1,543.4; NI 79.4; EPS 0.13; CR 94.3; BVPS 12.60; TTM ROE 18.2 %; avg diluted shares 589.3 | 0000080661-15-000034 | https://www.sec.gov/Archives/edgar/data/80661/000008066115000034/exhibit99may2015earningsre.htm |
| Apr-2019 monthly results (missing month): NPW 3,669.8; NPE 3,348.3; NI 487.8; EPS 0.83; CR 87.4; BVPS 20.74; TTM ROE 28.8 % | 0000080661-19-000027 | https://www.sec.gov/Archives/edgar/data/80661/000008066119000027/pgr20190430exhibit99earnin.htm |
| Filing listings for the gap windows | — | https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000080661&type=8-K&dateb=20150630&owner=include&count=10&output=atom (and `dateb=20190531`) |
| Pagination file format (flat; contains both missing 8-Ks) | — | https://data.sec.gov/submissions/CIK0000080661-submissions-001.json |
| Dec-2004 release: no equity line; BVPS 25.73, ROE 30.0 %, debt/capital 19.9 %, 200.4M shares | 0000950152-05-000341 | https://www.sec.gov/Archives/edgar/data/80661/000095015205000341/0000950152-05-000341.txt |
| Jan-2005 release: BVPS 26.18, ROE 26.4 %, debt/capital 19.7 % | 0000950152-05-001310 | https://www.sec.gov/Archives/edgar/data/80661/000095015205001310/l12207aexv99.htm |
| FY2004 total shareholders’ equity 5,155.4 (2003: 5,030.6); Dutch-auction note | 0000950152-05-001650 | https://www.sec.gov/Archives/edgar/data/80661/000095015205001650/l12357aexv13.htm |
| FY2005 total shareholders’ equity 6,107.5 | 0000950152-06-001512 | https://www.sec.gov/Archives/edgar/data/80661/000095015206001512/l17994aexv13.htm |
| Q3-2017 EPS basic 0.39 / diluted 0.38, NI 224.0; equity 9-30-2017 9,289.4 | 0000080661-17-000066 | https://data.sec.gov/api/xbrl/companyconcept/CIK0000080661/us-gaap/EarningsPerShareBasic.json (also `…Diluted.json`, `NetIncomeLoss.json`, `StockholdersEquity.json`) |
| Sep-2017 month NI 41.5 / EPS 0.07, Q3 NI 224.0 / EPS 0.38; realized gains (13.9); TTM ROE 16.3 % | 0000080661-17-000063 | https://www.sec.gov/Archives/edgar/data/80661/000008066117000063/exhibit99september2017earn.htm |
| Aug-2017 NI (16.8), EPS (0.03), pretax (49.5) | 0000080661-17-000060 | https://www.sec.gov/Archives/edgar/data/80661/000008066117000060/0000080661-17-000060.txt |
| FY2018 EPS 4.45 / 4.42, NI 2,615.3; equity 12-31-2018 10,821.8 | 0000080661-19-000008 | companyconcept URLs above |
| 9M-2018 EPS 4.01 / 3.98, NI 2,350.6 | 0000080661-18-000050 | companyconcept URLs above |
| Oct-2018 NI (31.7), EPS (0.06), pretax (69.4), realized gains (259.9) | 0000080661-18-000052 | https://www.sec.gov/Archives/edgar/data/80661/000008066118000052/0000080661-18-000052.txt |
| Dec-2018 month NI 54.0 / EPS 0.09, Q4 NI 264.7 / EPS 0.44; realized gains (330.8); TTM ROE 24.7 % | 0000080661-19-000003 | https://www.sec.gov/Archives/edgar/data/80661/000008066119000003/pgr20181231exhibit99earnin.htm |

Notes:
- The index page for `0000950152-05-000341` returned HTTP 503 once and was not retried; the full-submission `.txt` was used instead.
- Two reachability probes at session start were fetched once and not re-requested.

## Appendix B — Test suite and safety record

- **Environment:** `.venv` (gitignored), Python 3.11.15, pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, xgboost 3.2.0, pytest 9.0.2.
- **Run 1:** `python -m pytest -q -rfEs` gives effective `-qq`. Exit code 0; no summary line printed.
- **Run 2:** `python -m pytest -o addopts="--tb=short" -q -rfEs`, isolated `--no-hardlinks` clone.
  - Result: `1965 passed, 2 skipped, 421 warnings in 326.33s (0:05:26)`, exit code 0.
  - Skips: `tests/test_classification_shadow.py:279` (integration, needs DB) and `tests/test_v45_features.py:259` (`combined_ratio_ttm` not produced).
- **After both runs:** no tracked changes in the clone; DB sha256 unchanged. Ignored files written: `.hypothesis/`, `.pytest_cache/` and `data/processed/feature_matrix.parquet` (F28).
- **In the real repo:**
  - Read-only opens of the WAL-mode DB created `data/pgr_financials.db-wal` (0 bytes) and `-shm`. They were removed once no process held them.
  - A read-only feature build by one agent (working directory = repo) left the gitignored `data/processed/feature_matrix.parquet`.
  - The tracked DB hash was `e7531a34…8b38` throughout.
