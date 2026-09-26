# Independent verification — 2026-09-26

## Owner summary

The repairs correct real errors in stock-split returns, price windows, accounting fields, macro-data timing, tax calculations and model evaluation. I reproduced the repaired values rather than accepting the CHANGELOG's claims. The current database has no unexplained weekly price jumps, duplicate macro months, missing PGR monthly releases or failed accounting identities under the checked tolerances. All 73 comparable quarters reconcile monthly net income with the stored quarterly XBRL facts. The latest production feature row has no missing live inputs.

The recommendations remain cautious. Replaying February through September 2026 gives **DEFER-TO-TAX-DEFAULT and 50% in every month**, although forecasts and consensus labels change. September's out-of-sample R² is +2.86%: the sum of squared prediction errors is 2.86% lower than predicting the average return that was actually known at each forecast date. This replaces a misleading old comparison that included the answer being predicted. September's directional hit rate is 62.83%, but always predicting the more common direction achieves 68.14%. That explains why a positive R² does not unlock an active recommendation. IC, a measure of how well predictions rank outcomes, is 0.0762 when equally averaging benchmark results; its pooled counterpart is 0.1019 and is not statistically convincing after clustering observations by date. The reported LOW confidence is consistent with these limits.

The refactor itself preserved September's behavior: ten output files are byte-identical between the last pre-restructure commit and current master using the same database copy. Only the manifest's commit, timestamp and entry-point name differ. Previously committed monthly summaries were moved unchanged; they are historical records, not newly repaired forecasts.

This is **not an all-clear**. Of 35 original findings, 18 are FIXED, 15 PARTIAL and 2 NOT ADDRESSED when assessed against their complete scope. Twenty dividend histories remain stale, including several live benchmarks; recomputing returns from that database cannot recover missing dividends. The full Windows test run has three failures, including a real hole in the test database-access guard. The forbidden combinatorial K-fold diagnostic still runs. Older research still contains invalid validation, target and cache practices. The requested zero-sys.path criterion also fails: 197 existing exceptions remain outside conftest.

The research plan starts at v200 and consolidates old work into eight sessions, rather than rerunning every release. Its first gate is a dividend/target repair in a separate migration or rebuild PR. Historical metrics below are reproducible on the pinned snapshot but provisional as evidence of investment skill. The most recent 24 mature target months have already been examined, so they cannot become an untouched promotion holdout by renaming them. The plan quarantines them for one retrospective synthesis and requires genuinely unused forward evidence for promotion. The owner decision is whether to wait for that evidence or retain research winners as shadow candidates. No live settings, data or production code changed in this verification PR.

## Scope, independence and safety

- Audited master: `aae0be883309bdc068ee9afbb78b0aa9a4b52b8d` (PR #133), fetched before branching to `review/verification-2026-09-26`.
- Original review: [REPO_REVIEW_2026-09-25.md](REPO_REVIEW_2026-09-25.md), baseline `c948d0f`. Scope is CHANGELOG v171–v185 / PRs #120–#133, including 4b and 4c. No WP11 rerun is represented as complete.
- Snapshot: `data/pgr_financials.db`, SHA256 **`7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d`**. Integrity check: `ok`.
- Tests, defect reversals and replays ran in independent local clones outside the repository, with copied DB files. The source DB was inspected via `file:...?mode=ro&immutable=1`. Python socket connections were blocked. No provider, fetcher, EDGAR, FRED, Alpha Vantage or email call was made. Installing Python packages was the only external network activity.
- The source tracked DB and both replay DB copies have the same SHA256 before/after. No source production artifact or ledger was written. Feature caches and replay outputs live only in scratch clones/temporary paths.
- Only the two requested Markdown deliverables are added. The explicit “and nothing else” deliverable takes precedence over the generic CHANGELOG update and earlier one-report-only wording. This report is the change summary; no production behavior changed.
- Two independent read-only subagents supplied the history/location map and historical research inventory. Dynamic evidence below was executed by the primary verifier. Historical repair reports are context, never substitutes for fresh checks.

## Runtime and reproducibility

Windows PowerShell; Python 3.12.14 in an isolated temporary venv, installed from the project's declared dependency ranges. Important versions: pytest 9.0.2, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, skfolio 1.4.0, xgboost 3.4.1, scipy 1.18.1, statsmodels 0.15.0, matplotlib 3.11.2, mapie 1.5.0, hypothesis 6.135.7, PyYAML 6.0.2. Results apply to this environment; unpinned ranges are a reproducibility limitation.

Evidence directory on the verifier's host: `C:\Users\Jeff\AppData\Local\Temp\pgr-verify-20260926-72d5b327`. Logs and scratch scripts were intentionally not committed. Appendix A embeds the DB/feature audit so its command can be reconstructed. Run every command below in a scratch clone at the audited commit with a copied DB; never aim replay commands at the source checkout. Install the scratch clone editable, retain its repository root on PYTHONPATH, and do not prepend its `src/` directory (that would shadow the `research` namespace).

Command aliases used in the verification table:

| Alias | Executed command / evidence |
|---|---|
| E1 | `python sweep.py` in the scratch current clone; embedded verbatim in Appendix A, immutable connection and redirected cache. `sweep.log` contains the outputs. |
| E2 | `python -m pytest -o addopts="--tb=short" -q tests/integration/pipeline/test_entrypoint_imports.py tests/integration/data/test_db_price_integrity.py tests/integration/data/test_pgr_edgar_integrity.py tests/integration/pipeline/test_validation_gating.py tests/integration/pipeline/test_fred_pipeline.py tests/unit/processing/test_price_features.py` → **113 passed in 31.39s**, exit 0. |
| E3 | `python -m pytest -o addopts="--tb=short" -q` → full-suite result and failed node IDs below. Passing nodes are fresh executions, not historical claims. |
| E4 | `python scripts/replay_monthly_decisions.py --committed-dates --out <temporary>/current-replay.csv` → eight successful dry runs, exit 0, unchanged DB SHA256; script supplies `--skip-fred` to each subprocess. |
| E5 | `python scripts/replay_monthly_decisions.py --as-of 2026-09-21 --out <temporary>/golden.csv` at `b609af6` and `aae0be8`, same copied DB; byte/JSON comparison described below. Pre-restructure script runs the old entry point. |
| E6 | `python scripts/checks/check_doc_links.py`; `python research/tools/registry.py`; `python scripts/checks/check_sys_path_edits.py`; import smoke nodes in E2. |
| E7 | `rg -n 'cv=None' research/studies --glob '*.py'`; `rg -n 'CombinatorialPurgedCV|RidgeCV|shuffle=True|StandardScaler' src cli scripts research --glob '*.py'`; followed by reading each match in its training context. |
| E8 | `rg -l 'sys\.path\.(insert|append)|sys\.path\s*=' --glob '*.py' .`; `rg --files results --glob '*.py'`; workflow/source inspection commands in the history appendix. |
| E9 | Named red/green scratch tests in the reversal table below, with restored current code/data. |
| E10 | `python <temporary>/manual-targets.py` (Appendix C), independent SQLite/calendar/math product calculation: selected benchmark/relative DRIP gaps <7e-16. |
| E11 | `python <temporary>/remaining-toys.py` (Appendix D): BLP outcome-invariance, unusable monthly FFD and actual peer-workflow SQL failure reproduced. |

## Verification table

Statuses assess the **whole original finding**. PARTIAL includes repaired production behavior with outstanding research or secondary symptoms. NOT ADDRESSED means the core defect remains; incidental mitigation is identified. No original finding is labeled REGRESSED without evidence that a previously repaired behavior returned. New Windows and research issues are listed separately.

FE = `src/processing/feature_engineering.py`; DB = `src/database/db_client.py`; D/ = `src/pgr_vds/decision/`; E/ = `src/pgr_vds/ingestion/edgar_monthly/`. Line numbers refer to audited master, not this documentation commit. Current locations were traced using `git log --follow`; extraction additions were also matched to original function bodies because git cannot follow one file into multiple children.

| ID | Original cited location(s) | Current location / function | Status proposal / current static evidence | Fix commit(s) / PR | Evidence command | Fresh result / remaining symptom |
|---|---|---|---|---|---|---|
| F01 | FE:166–175,253–286,1370–1378; `src/ingestion/multi_ticker_loader.py`:43 | FE:320 `build_feature_matrix`, adjusted close:370, calendar momentum:384, 4/13-week volatility:392, 52-week high:406; DB builder:1208 with splits; `src/processing/price_adjustment.py`:37/125/146/173/184/203; loader weekly API retained at same file | PARTIAL: production price-window/split/frequency defects fixed; `daily_prices` is still its legacy name but semantics documented and frequency asserted. Re-selection of v18/v20 feature sets on repaired data remains WP11. Fresh sweep confirms latest mom12 equals true calendar return. | 71854ba #124; fd432f5 #121 | E1; E2 price-feature tests | mom_12m = true 12-month return = -0.1150327855581641; 13-week volatility test passes. Feature re-selection not rerun. |
| F02 | `src/models/wfo_engine.py`:521–530,660–687; `config/model.py`:97; `src/reporting/decision_rendering.py`:50,65 | WFO:473 `cpcv_path_thresholds`, 559 `_recombined_path_members` (columns via `.T` at573), 586 `run_cpcv`, import635/construction654; config:78,125–127 and CPCV embargo; decision rendering:85 gates,77 availability,164 completion gate | PARTIAL: impossible path threshold and wrong iteration fixed; 7 paths scale to ceil(19/28*7)=5 and embargo=2. Verdict demoted to diagnostic. **AGENTS absolute no-KFold rule remains violated:** current monthly run still executes CombinatorialPurgedCV (D/signal_generation:159) and completeness is still a recommendation gate. WFO doc explicitly admits later-training folds. | c8e6f66 #127 | E2 CPCV tests; E4; E7 | Seven recombined paths, current 4/7 MARGINAL, GOOD requires 5; no impossible threshold. Forbidden K-fold execution remains. |
| F03 | `scripts/weekly_fetch.py`:55–73; `scripts/apply_split_history.py`:48–65; `src/processing/multi_total_return.py`:111–127 | Canonical `config/splits.py`:34 VOO 0.5 on 2013-10-24; weekly:66 seed/187 rebuild; apply-split script reads canonical list; multi return:84 builder/122–128 BME window; `src/processing/price_integrity.py`:41 jump guard | FIXED in source: one registry, VOO split, guarded target rebuild. Actual first split-adjusted date is Oct24 rather than weekly-bar Oct25 from original review. Target rebuild committed43e82d5; row diff is step2_target_rebuild_rows.csv. | fd432f5 +43e82d5 #121 | E1 targets; E2 DB target/jump tests; E10 | VOO 2013-10-24 split=0.5; no unexplained jumps; Apr/Sep2013 6M targets match independent E10 split/DRIP to <7e-16. |
| F04 | `src/reporting/backtest_report.py`:63–65 | Same file:35 `compute_oos_r_squared`,85/88 `prevailing_mean_forecast`; `src/models/prequential.py`:91; `src/models/forecast_diagnostics.py`:93/180; D/health:165 aggregate | FIXED in source: realized-only prevailing mean, including training history, replaces current-target expanding mean. Source can also accept explicitly aligned benchmark forecasts. | c8e6f66 #127 | E2 naive/oracle tests; E9 step5; E4 | Changing the current target cannot alter its naive forecast; September R² +2.8643%. Red 1 failed → green 1 passed. |
| F05 | Same as F03 | `config/splits.py`:43 VGT8.0 on 2026-04-21; same canonical seed, jump guard and BME target functions | FIXED in source and committed target rebuild. The issuer/date corrected to Apr21 (original review observed Apr24 weekly bar); original suspect action no longer only inferred from a jump. | fd432f5 +43e82d5 #121 | E1 targets; E2 DB target tests; E10 | VGT 2026-04-21 split=8; Oct2025/Feb2026 production-rebuild gaps0.0; independent E10 gaps<7e-16. |
| F06 | `src/ingestion/fred_loader.py`:131,190–195; FE:102–122,1219–1222 | Fred loader:172 `to_monthly_observations` (BME last at183),206 `fetch_all_fred_macro` default lagsFalse,276 raw upsert; FE:109 `_apply_fred_lags` period arithmetic; DB:1540 canonical upsert | FIXED in source: raw current-vintage storage and one publication lag by calendar period; duplicate month labels canonicalized rather than row-shifted. Fresh sweep reports duplicates0. v134 rerun still pending (F25/WP11). | f72e0bd #122 | E1 FRED SQL; E2 single-lag fixture; E9 step3a | Duplicate (series, calendar-month) groups=0; VIXCLS 2020-04-30=34.15; double-lag reversal fails then passes. |
| F07 | weekly:105; old monthly_decision:309; WFO:409–416; DB:304–387 | weekly:97 FRED step using production union; D/refresh:fetch_fred_step; Fred loader:191 `production_fred_series`; DB:518 `check_data_freshness`; D/signal_generation:102 live-feature warning,212 `find_nan_live_features`; rendering freshness lines435 | FIXED in source: PGR series refreshed in both jobs, per-series/per-ticker freshness, explicit live-feature NaN warnings appear in reports/manifest; they do not themselves fail the recommendation gate. Fresh sweep confirms no NaN on current live row. Original imputation remains training-median imputation, but no longer silent about live missingness. | b66bc01 #120; f72e0bd #122 | E1 freshness/live inputs; E2 FRED tests | All 11 production FRED series meet required months; no missing Ridge12/GBT13 live features; rate gap 0.00854612335. |
| F08 | weekly:162–175; `src/ingestion/multi_dividend_loader.py`:208–224; `.github/workflows/initial_fetch_dividends.yml`:19 | weekly:134 selector,242 dividend refresh,312 weekly PGR dividends; multi-dividend loader same file retry/sleep handling; DB:630 dividend freshness; weekly workflow:9 Wednesday cron; initial dividends workflow now dispatch-only:20 | PARTIAL: feed retry/pacing/budget/schedule/freshness code fixed. Current fresh sweep still finds20 stale tickers (19ETF+ALL); target dividend contamination therefore not fully remediated. Step2 explicitly lacked AV key for backfill; PGR is now refreshed but includes future exdateOct1. | fd432f5 #121;3a05d0c #128 | E1 check_dividend_freshness; E3 feed tests | 20 STALE tickers remain (19 ETFs + ALL); PGR refreshed through scheduled 2026-10-01. Feed-code tests pass, backfill incomplete. |
| F09 | `src/ingestion/edgar_client.py`:197–220,365–371; FE:1180–1189 | Edgar client:150 earliest duration facts,208 `_flow_concept_with_filed`,324 `trailing_roe`,361 fetch,399 `fundamentals_from_companyfacts`; FE:1320 quarterly ROE filing placement; DB:1028 loader | FIXED in source: Q4=FY−9M, earliest filed, trailing NI /5-quarter-equity average, filing_date; NULL pe/pb columns removed by migration007. Fresh sweep73 quarters, all NI reconciles. | b71fdc8 #123;3a05d0c #128 timing | E1 quarterly reconciliation; E3 edgar_client tests | 73 quarters compared, zero NI reconciliation violations under the $1M tolerance; ROE range -0.009396 to 0.386113; Q4 EPS median 0.645, not full-year EPS. |
| F10 | old EDGAR script:816–817; FE:700–706 | E/parse:657 `investment_book_yield` stores yield_raw; FE:826 joins percent-valued yield; migration004 correction | FIXED in source: division by100 removed; repair normalized mixed-unit history. | b66bc01 #120; b71fdc8 #123; moved9fcf367 #133 | E1 yield range; E9 step1 | Stored book yield 1.6–5.6 percent; live 4.3; /100 reversal: 1 failed → 1 passed. |
| F11 | old EDGAR:1753–1765,1813–1823; FE:325–353 | E/derive:19 `_prior_year_key`,42 `compute_derived_fields`; `src/processing/pgr_edgar_derived.py`:64 calendar prior-year,40/45 PIF component definition,111/128 period YoY; FE:447 growth/471 total; E/parse retains printed totals in raw provenance | FIXED in source: leap-year keys, stable PIF components excluding property, period YoY and shared Gainshare. Fresh sweep PIF checks0. x-series raw target problems are separately F25. | b71fdc8 #123; moved9fcf367 #133 | E1 PIF values; E2 integrity tests | Feb2021 growth 0.102362243; Feb2025 0.182309604; component/PIF jump violations=0. |
| F12 | old EDGAR monthly HTML value parser | E/parse:212 `_parse_number`,236 `_row_numbers`,272 `_get_first_numeric` handles parentheses/lone-sign/Unicode; parse_html_exhibit current module | FIXED in source; step3b record says160 sign cells repaired and negative NI restored for2017-08/2018-10. Root identity sweep is additional data evidence. | b71fdc8 #123; moved9fcf367 #133 | E1 known values; E2 known-filed test | Aug2017 NI=-16.8, EPS=-0.03, pretax=-49.5; sign repair survives identity checks. |
| F13 | `config/model.py`:46; evaluation:451–495; consensus_shadow:45; old monthly:532–571,609–721; decision_rendering:49; forecast_diagnostics:103–105; old v38:103/conformal:217,341 | config:45–54 realized-only rule; evaluation:450 reconstruction; prequential:148 weights/215 alpha/261 panel; D/signal_generation:240 calibration,359 conformal; D/health:165; decision rendering:85; forecast diagnostics:180 date-clustered; consensus_shadow quality weights still exist; moved `research/studies/v38_shrinkage/v38_shrinkage.py` | PARTIAL overall: live health uses prequential weights/alpha/ECE/trailing coverage, equal-weight IC and PT directional-skill gate, date-clustered significance. Selection/research rebaseline and quality-consensus promotion evidence remain pending WP11; registry v38 notes49. Drift restart lacks corrected historical records (F31). | c8e6f66 #127;3a05d0c #128 live policy; movementaca5b92 #133 | E2 prequential/cluster/skill fixtures; E4 | Future-fold perturbation tests pass; Sep equal-weight IC .076198, pooled .101940 (p .125422). Old research selection not certified. |
| F14 | old monthly:731–779,3039,3180,3402–3683,3576; weekly dry-run writes | D/health:274 snapshot/355 retrain guard; D/pipeline:338 main,358 ro DB,507 dry flag,570 dry folder,621 skip log,662 retrain dry flag; D/artifacts:45 dry output/288 ledgers/375 manifest; weekly:279 readonly | FIXED in source for DB/committed production artifacts and ledgers. Dry output is results/dry_run; health audit writes skipped, manifest dry_run flag and classification. FE still writes ignored parquet cache (F30), and dry run intentionally emits scratch artifacts. | b66bc01 #120; movementaca5b92 #133 | E4; E5; E3 dry-run tests | Copied DB SHA256 unchanged after every replay; ten production-equivalent outputs remain scratch. Ignored parquet still unversioned (F30). |
| F15 | FE:516–528,745–763,1191–1205,1227–1366; detailed research x1/x9/x12/x17 | FE:644–647 BVPS growth;877–882 buyback share estimate;1265 share-basis restatement;1336–1341 PB;1374 adjusted peer close/1390 spread loops/1416 peers; price_adjustment:249 latest-share basis; research x-series targets same `src/research/x1_targets.py` | PARTIAL: cited production per-share/spread defects fixed; research targets and committed x12 discontinuity interpretations remain unchanged and un-rebased. | 71854ba #124; x files unchanged except study moves02e3fd4 #131 | E1 2006 share-basis series; E2 continuity tests | P/B 3.30493→2.47979 May2006–Jun2007, no 4x split cliff; BVPS YoY .1649–.2180. x-target research still raw. |
| F16 | old EDGAR:2430,2442; detailed derive1813–1838 | E/load:203 `load_from_csv` without row pct_change prefill; E/derive:42/113 recompute; pgr_edgar_derived:111 period-growth | PARTIAL: confirmed gap-based13-month YoY error fixed and data repair completed; **2024 fiscal-month change hypothesis remains uninvestigated**, explicitly step3b report243. No change to monthly flow seasonal choice. | b71fdc8 #123; moved9fcf367 #133 | E1 known YoY; E2 integrity tests | May2016 NPW YoY .105539395, Apr2020 .025723473: calendar comparisons restored. 2024 fiscal-month hypothesis remains untested. |
| F17 | old EDGAR:224–240 (item filter184–189); FE:314–317 | E/fetch:188 `_filings_block` flat/nested,203 item code includes9.01-only,287 full pagination; E/parse selection; FE:439 rolling12 CR; current complete265-month grid | FIXED in source/data according fresh sweep: both2015-05 and2019-04 included, no missing monthly coverage. | b71fdc8 #123; moved9fcf367 #133 | E1 coverage; E2 coverage/known values | 265 monthly rows, missing-month list []; May2015 NPW1581.4, Apr2019 NI487.8 present. |
| F18 | old EDGAR monthly field matching | E/parse:804 anchored equity labels,1523 fallbackBVPS*shares;1636 validator; `src/processing/pgr_edgar_validation.py`; repaired rows step3b cell diff | FIXED cited parser/data symptom: anchored equity/debt and balance plausibility; zero equity identity violations in fresh sweep. Original optional recommendation to winsorize all ratio features was not established as needed after repair. | b71fdc8 #123; moved9fcf367 #133 | E1 equity/debt and identity; E2 integrity tests | Dec2004 equity5155.4/debt1284.3; zero common-equity identity violations after preferred-equity adjustment. |
| F19 | capital_gains:58,155–157,442–467,524,615,629–639; MC:125–145; old monthly:1184–1301 | Capital gains same file:62 eligibility/91 first LTCG date/100 vested/119 wash conflicts/184 optimize/572 gain fraction/583 breakeven (return at619)/637 scenarios (argmax at799); MC:161 weekly vol; D/tax_lots:40 tax text/144 provisional and absolute-return assumption | FIXED cited tax/MC math in source: signed −g(S−L)/(1−L), anniversary+1, per-share lot order, wash windows, unvested filtering, expected-proceeds scenario ranking, adjusted52-week volatility. **Adjacent residual:** `src/portfolio/rebalancer.py` still uses365-day STCG zone (step6 report258); not part of original cited capital-gains locations. |71854ba #124 MC;3a05d0c #128 tax; movedaca5b92 #133 | E9 step6; E2 MC-vol test; E3 tax tests | Fully appreciated lot breakeven=-21.25%; old formula fails then passes. Adjusted weekly vol fixture passes. Adjacent 365-day rebalancer remains. |
| F20 | decision_rendering:14–50; old monthly:406–410 | Rendering:16 `sell_pct_from_consensus`,77 CPCV availability,85 gates; D/signal_generation:159 CPCV exception; `src/models/live_policy_backtest.py`; D/health:399 policy summary | FIXED cited mapping/fail-open symptom: bullish never sells>50%; nonfiniteIC weak; missing/UNKNOWNCPCV fails closed. Exact mapping backtested via realized-only panel fixture; step6 record new+3.81% vsalways50+3.64% (186dates), old+3.53%. This is recorded evidence, not a new run here. CPCV presence itself remainsF02 policy violation. |c8e6f66 #127;3a05d0c #128 | E2 missing/UNKNOWN CPCV gate; E3 policy fixtures; E4 | Fail-closed tests pass; bullish mapping <=50%; all eight replay modes defer. No new economic policy-edge certification here. |
| F21 | multi_benchmark_wfo:310–366; old monthly report1830 | Multi benchmark:270 `get_ensemble_signals` removes Bayesian std from output; D/signal_generation:240 calibration,334–336 writes calibrated prob into both probability columns and tiers; calibration:417 directional tier; recommendation_report:602 calibrated display; redeploy_portfolio:180 preferred calibrated probability | FIXED live reported constant-probability/tier symptom: temporary0.5/LOW placeholders are overwritten by live calibrate_signals; retired Bayesian posterior text gone. Disagreement between forecast sign and calibrated event probability can still exist honestly; tier reads probability in signal direction. |c8e6f66 #127;3a05d0c #128; movementaca5b92 #133 | E2 confidence-tier/direction tests; E4 | Probability no longer constant; Sept calibrated outperform .658071 but UNDERPERFORM signal gives LOW directional support. |
| F22 | multi_total_return:115–127; FE:1454–1473; DB.upsert_prices; detailed backtest_engine229–240 | Multi return:122–128 forward_window_end/BME; FE:1514 truncates target byactual window end; DB:707 upsert and788 dedupe ISOweek; backtest engine same file:229 nearestfirst future event target | PARTIAL complete scope: confirmed BME/duplicate/as-of defects fixed; **suspected vest-event/backtest one-month target offset remains**: backtest_engine was not changed by review fixes and still chooses first target on/afterevent. Need separate quantification before upgrading suspicion. |fd432f5 #121 (43e82d5 rebuild); backtest unchanged | E2 DB target checks; E3 as-of tests | All stored relative-return targets match the production BME rebuild; weekly duplicates=0; actual end-date cutoff tests pass. Vest-event offset remains suspected/unquantified. |
| F23 | config/features:237; FE:142–162 | FE:204 `edgar_availability_dates`,245 `place_edgar_rows_by_filing_date`,1268 monthly/1303EPS/1320quarterROE; config/features fallback lag remains | PARTIAL broader scope: production placement fixed (availability≥filing_date, fixedlag onlymissing dates). Research v142 fixed-lag tuning/old setting evidence not rerun; step6 report259 explicitly research fixedlag remains. |3a05d0c #128 | E3 filing-date tests; E4; step5 same-DB attribution | Production uses observed availability; Sept before/after filing-placement R² .050775316→.028642582. Old fixed-lag research not rerun. |
| F24 | path_b_classifier:286; classification_shadow:765–803; overlay:108–112; v160_ta_features | PathB:235 impute→scaler→logistic,259 requiredX_current/307 current row; shadow:832 passes current_features; models/classification_monitoring:33/83/85 recompute maturity; overlay:79 direction-aware; v160:235 split map/249 weekly-adjustedTA; D/artifacts:288 updates classifier outcomes but TAappend352 | PARTIAL: cited current-row/scaler/classifier-maturity/veto/TA-feature bugs fixed. **TA ledger maturity remains creation-time only** (step6 report256 and artifacts code has no TA outcome attachment). |71854ba #124;3a05d0c #128; movedaca5b92 #133 | E2 TA tests; E3 PathB/monitoring/veto tests; E4 | Current-row pipeline and maturity fixtures pass; TA split/week fixtures pass. TA ledger outcomes still not reattached on later runs. |
| F25 | pb_vs_pe:217–359; x1_targets; old results/research/v39_ridge_alpha:63 and v40–59; v37_utils:118 | Same pb_vs_pe:217 bootstrap/248 one-way stats/304 OOS; same x1_targets:24/58 raw row shifts and137 Q1-only specialdividends; v37_utils:112/118 filters startdates only; moved v39 study same line 63 and16 other RidgeCV calls | NOT ADDRESSED: research inference/conclusions/holdout crossing/x-target math untouched by behavior fixes. Seventeen cv=None calls across 16 moved v39–59 scripts persist. December specials remain outside the Jan–Mar Q1 window. History moves preserve behavior; WP11 reruns pending. |No fixing commit; study movement 02e3fd4 #131; pb_vs_pe last a4236f1, x1 last 8a3ce95 | E7; inspect x1_targets and v37_utils | 17 cv=None RidgeCV calls across16 old studies; raw row-shift x-targets and origin-only holdout filter remain. Original symptom does not pass. |
| F26 | workflows/*.yml; api:34–41; model:163–171; detailed initial_fetch:87, peer_bootstrap:108, post_initial:108, monthly:3027–3033 | Shared db-writer at weekly:25, peer:17, initial prices:32, initial dividends:35, post-initial:62, monthly 8K:22, monthly decision:43; monthly workflow_run:18 / generated guards:111,123,142,172; API:44 strict UA; schedule:22 safe as-of /56 mode validation. Residual peer_bootstrap:119 invalid price_date; initial_fetch:83 force parameter /435 flag. | PARTIAL: core serialization, dispatch-only bootstraps, UA, offline smoke, generated/email gating, mode and as-of fixed. Invalid peer MIN(price_date), ignored initial --force, no CI permissions block, tag-only Actions pins and query-string key logging risk remain. Broad git-add-results removed by e48b402. Concurrency pending cancellation behavior is documented. Historical duplicate emails require logs to confirm. | 3a05d0c #128; e48b402 #129 narrow paths; CLI moves aca5b92/9fcf367 #133 | E8 workflow inspection; E3 workflow tests; E11 schema query | Seven writers share db-writer; output/email gates tested. peer_bootstrap line119 still queries nonexistent price_date; other residuals in location table. |
| F27 | fred_macro_monthly; daily_prices (CB); src/research/v19.py | Fred loader:172/183 and DB:1540 canonical months; loader:139 explicitly current vintage; v19 same file raw public fetch; multi_total_return:202 proxy_fill=0; peer_fetch retains identity handling; artifacts/ops/fetch_status.md | PARTIAL: FRED duplicates and mixed aggregation fixed (monthly last). Current-vintage FRED remains; CB entity splice and TRV dividend overlap not repaired; FZROX orphan proxy / proxy_fill=0 remains; fetch-status semantics largely unchanged. Fresh sweep reports duplicates=0. | f72e0bd #122 FRED portion; artifact move e48b402 #129 | E1 FRED/CB SQL; E2 FRED tests | FRED duplicate count0; CB1402 rows all source=av/proxy_fill0 across1999–2026; entity splice, vintage/proxy risks untouched. |
| F28 | tests/; detailed total-return/as-of/fracdiff/valuation/property/EDGAR/tax fixtures and FE:1059 | tests/unit/processing/test_total_return.py, test_asof_target_truncation.py, test_fracdiff.py, test_valuation_multiples.py, property*.py; tests/unit/models/test_property_wfo_temporal.py; tests/unit/ingestion/test_edgar_client.py; tests/unit/tax/test_capital_gains.py, test_three_scenario_tax.py; tests/integration/pipeline/test_mutation_kills.py; tests/conftest.py:115 guard /144 path redirects; tests/repo_guard.py | PARTIAL broad adequacy: defect-class fixtures, properties, as-of, annual facts, leap-tax and vacuous assertions hardened; step9 reports 0/18 surviving mutations. Guard does not cover collection-time or C-level writes. Old test_total_return still has the flat-price split fixture (history unchanged except move); new matching-price-drop coverage exists elsewhere. Stored-artifact research tests remain, now separated/marked. Fresh primary-verifier test results appear in E3/E9. | 7032649, 5bbb058, 199c65d, 2d675f3 #130; earlier fix-specific tests PR120–128; 95e0c83/96151c2 #132 | E3; E9 step9-math | Production-backed temporal fixtures pass and minimum-row reversal fails both horizons. Three Windows suite failures include a guard escape; not all-clear. |
| F29 | .gitignore:15–21; DB header; pytest.ini; requirements; docs/dictionary/CSV/CLAUDE/config/PEP8 | .gitignore:18 contents negation and WAL patterns; DB:91 finalize; pytest.ini/requirements removed → pyproject:21 Python3.11 /23 pandas>=3,<4 /71 addopts /99 lint; dictionary:8 coverage265 /11 share units; active operator CLI docs; CLAUDE.md points to AGENTS; ROADMAP:301 still Python3.10 | PARTIAL: ignore, WAL, pytest summary, dictionary, CSV, docs and uppercase CLAUDE addressed. Strict PEP8 is not enforced: lint selects only E9,F63,F7,F82. Runtime dependencies remain ranges rather than a full lock; unused config still exists. Archived plans intentionally retain historical commands/daily semantics. Generated report footer still names old monthly script in D/rendering. | b66bc01 #120 hygiene; b71fdc8 #123 CSV/dictionary; 96151c2 #132 config/docs; d616fca/9fcf367 #133 | E3 hygiene tests; E8 pyproject/docs inspection | Summary now visible; packaging/documentation tests mostly pass. Ruff only E9,F63,F7,F82; dependency ranges and Python3.10 ROADMAP contradiction persist. |
| F30 | results/research/v46_classification.py; v66_utils:15; FE:1059; v129 feature map; sixteen x scripts | v46 code → src/research/binary_classification.py:43 compute_binary_metrics; v66_utils:20 imports real module; FE:1182 to_parquet remains; v129_feature_map:15 loads configured study output; shadow_followon:19 study paths; x12 study:50/57 reads unversioned parquet | PARTIAL: production import, global warning filters, stdout and sys.path import effects removed; results has 0 Python files. Unversioned parquet/cache provenance remains; production still reads v128/v141–149 outputs under research/studies/*/outputs. Moving the output path did not remove that dependency. | e48b402 #129; 02e3fd4 #131 study-output routing | E2 import smoke; E8; inspect cache consumers | 0 .py under results; imports pass. FE still to_parquet without dataset/code provenance; live code reads study output files. |
| F31 | FE:1480–1669; blp:194; drift_monitor:45; detailed BL and monthly:3494 importance | FE:1542 weights /1568 min-d /1654 apply_fracdiff retains full-sample/default threshold1e-5; BLP:193 likelihood ignores y; drift_monitor:26/46 averages aggregate-IC snapshots; D/portfolio:45 raw ETF monthly returns /91 BL; D/health:448 importance fallback | NOT ADDRESSED core utilities: fracdiff defaults remain unusable/full-sample, BLP fit ignores outcomes, drift still averages full-history IC snapshots. c8e6f66 only adds metric-version filtering at57, preventing definition mixing without calculating recent IC. BL view/horizon behavior and feature-importance fallback were not repaired; phase5 copied the bodies. | No core fix; drift version filter c8e6f66 #127; function moves aca5b92 #133 | E11 toy symptom check; E8 fracdiff/blp/drift/portfolio inspection | BLP(y) equalsBLP(1-y); FFD gives1 usable output on60/323 rows. Full-sample d and averaged full-history IC snapshots remain. Metric-version filter is mitigation, not rolling IC repair. |
| F32 | whole repo; review section5 structure | pyproject; artifacts paths in config/paths.py; study registry/folders; docs/history; tests/unit,integration,research; D/eleven modules and thin CLI; E/fetch,parse,derive,load and thin CLI; src.pgr_vds import guard | PARTIAL against full target layout, with phases0–5 completed: code removed from results, artifact paths, docs/tests/monoliths migrated. 197 allowlisted sys.path files plus conftest remain. Older src.models/database/processing/etc and config remain separate packages; other production CLIs remain in scripts; parse.py has1,869 lines. Step7 explicitly omitted the repository-tree manifest. | e48b402 #129; 02e3fd4 #131; 95e0c83/96151c2 #132; aca5b92/9fcf367 #133 | E5; E6; E8 counts | Golden outputs match; registry113/0 problems; links340/0 broken; import smoke passes. 197 sys.path exceptions and missing research_lib fail requested full layout. |
| F33 | DB:709–780; monthly_8k_fetch workflow | DB:1118 whole-filing upsert /earliest wins /insert_missing; DB:1208 record raw; E/load:203/392 CSV insert_missing; E/derive:113 recompute; migration006 append-only raw+parse tables/triggers/views; monthly 8K CLI:57 | FIXED cited provenance, COALESCE mixing and CSV reseed symptom: append-only parse history, first/current views, same-filing updates preserve provenance, earlier filing replaces whole row, later filing is logged without merging. Legacy untraceable cells retained only in history/diff. Combine with fresh root provenance/data checks. | b71fdc8 #123; extraction 9fcf367 #133 | E2 provenance-integrity; E3 whole-filing/append-only fixtures; E1 | Append-only raw/parse history and same-filing/earliest-filing fixtures pass; repaired monthly table passes identities. No new remote provenance confirmation claimed. |
| F34 | old EDGAR:937–947,1087–1092 | E/parse:236 decimal-only footnote handling /777 companywide selection /921 ratio extraction /1636 validator; repaired2025-09 CR100.4, ER34.7 | FIXED parser/repair symptom; root identity sweep0 and step3b repair record. Original bad row satisfied CR=LR+ER, so filed-specific regression is essential (tests/integration/data/test_pgr_edgar_integrity.py). | b71fdc8 #123; extraction 9fcf367 #133 | E1 filed-specific row; E2 known-filed test | Sep2025 CR100.4=LR65.7+ER34.7. Filed-specific regression passes (identity alone would not detect old bad row). |
| F35 | old EDGAR:988,1728 versus DB:831 | E/parse:823/1610 emits roe_net_income_trailing_12m; DB:1103 _normalise_edgar_monthly_record /1110 maps alias; E/load:310 CSV map | FIXED live-parser→DB key mismatch; step3b repair filled2026-02–08. | b66bc01 #120; b71fdc8 #123; extraction 9fcf367 #133 | E1 ROE values; E3 alias/parser tests | Feb–Aug2026 ROE 34.7,35.0,35.1,36.0,34.7,34.2,33.4; canonical live key populated. |

## Fix lineage


| Fix step | Implementation / important follow-up commits | Merged PR / merge commit |
|---|---|---|
| 1, v171 | b66bc01, f09ded4 | #120 / dc0e24a |
| 2, v172 | fd432f5, 43e82d5 target rebuild | #121 / 34835d5 |
| 3a, v173 | f72e0bd, 711bfb3 | #122 / 15644cd |
| 3b, v174 | b71fdc8, 6be1175 | #123 / 89dc3e8 |
| 4, v175 | 71854ba | #124 / 8c3b2bc |
| 4b, v176 | 0df782b, 2aaa14c, a666d03 | #125 / 4b31e8b |
| 4c, v177 | 15db5cc | #126 / c089124 |
| 5, v178 | c8e6f66 | #127 / 69889ad |
| 6, v179 | 3a05d0c, e4df3d4, 58469df | #128 / b609af6 |
| 7, v180, phases 0–2 | e48b402 | #129 / 7b768a0 |
| 9, v181/v182 | 7032649, 5bbb058, 199c65d, 2d675f3 | #130 / f76cee2 |
| 10, v183, phase 3 | 02e3fd4 | #131 / 282a6b3 |
| 11, v184, phase 4 | 95e0c83 file moves, 96151c2 wiring | #132 / e45c047 |
| 12, v185, phase 5 | aca5b92 decision split, 9fcf367 EDGAR split, 6d9179c CLI-main fix, 47e2823 requests-stub fix | #133 / aae0be8 |

The original review report was merged by #119 / 9887288. There is no step-8/WP11 implementation in the current history; the registry explicitly flags its reruns pending.


## Tests and counterfactual defect checks

Exact requested full-suite command: `python -m pytest -o addopts="--tb=short" -q`.

Pytest's own summary line, copied from `full-suite.log`:

```text
3 failed, 2491 passed, 1 skipped, 109 warnings in 305.29s (0:05:05)
```

**Exit code: 1.** The single unconditional skip is `tests/unit/models/test_classification_shadow.py::test_detail_df_has_benchmark_specific_columns`, reason `integration test -- run manually with DB connection`. Warnings include numerical conditioning and convergence, not hidden test failures. Two preliminary attempts failed because the scratch src directory shadowed the research namespace and because an existing pytest temp directory was inaccessible; neither is substituted for the completed run above. The final run used the clone root and a fresh PYTEST_DEBUG_TEMPROOT.

Failed nodes:

1. `tests/integration/repo/test_restructure_phase1.py::test_email_reads_report_and_charts_from_artifacts`: expected slash-only regex, actual exception `artifacts\monthly_decisions\2026-09\recommendation.md`. Read-path portability issue; email was never sent.
2. `tests/integration/repo/test_test_suite_hygiene.py::test_classify_access[sqlite3.connect-args11-False-True]`: `classify_access('sqlite3.connect', ('file:///C:/.../data/pgr_financials.db?mode=ro',), False)` returns None. The URI path becomes `/C:/...` rather than the Windows drive path, so an unmarked committed-DB open escapes classification.
3. `tests/integration/repo/test_test_suite_hygiene.py::test_guard_fails_exactly_the_probes_that_touch_the_repo`: the read-only URI probe passes unexpectedly, and the wrapper's slash-only output parsing also misreads Windows node IDs. This is not merely cosmetic because of issue 2.

The named mathematical and data checks in E2 passed separately: **113 passed in 31.39s**, exit 0. Every mathematical repair selected below has an executable regression test; for at least one repair per implemented session I reinstated its defect in an external clone and then restored current code/data. These are targeted counterfactuals, not wholesale checkouts of incompatible older package layouts. All red exits are 1 and all restored green exits are 0. Step4b's mathematical reversal is rejected during frame construction by the share-basis consistency validator (a setup error); the additional historical script reversal lacks the new API. Neither is mislabeled as an assertion failure.

| Session | Named test (each command: `python -m pytest -o addopts="--tb=short" -q <node>`) | Reintroduced defect | Red pytest output | Restored green output |
|---|---|---|---|---|
| step1 | `tests/unit/ingestion/test_edgar_monthly_units_and_keys.py::test_parsing_book_yield_percent_stores_percent` | Parser stores percent /100 again. | 1 failed in 0.62s (exit 1) | 1 passed in 0.48s (exit 0) |
| step2 | `tests/integration/data/test_db_price_integrity.py::test_no_unexplained_weekly_price_jumps` | Delete VOO split from copied DB only. | 1 failed in 0.44s (exit 1) | 1 passed in 0.35s (exit 0) |
| step3a | `tests/integration/pipeline/test_fred_pipeline.py::test_fetch_store_build_gives_feature_equal_raw_minus_configured_lag` | Loader default apply_publication_lags=True again. | 1 failed in 0.56s (exit 1) | 1 passed in 0.47s (exit 0) |
| step3b | `tests/integration/data/test_pgr_edgar_integrity.py::test_known_filed_values` | Replace copied DB with git blob15644cd (before monthly/quarterly repair). | 1 failed in 0.43s (exit 1) | 1 passed in 0.34s (exit 0) |
| step4 | `tests/unit/processing/test_price_features.py::test_mom_12m_equals_true_calendar_return_across_split` | Replace calendar momentum with raw close.shift(months*21). | 1 failed in 0.45s (exit 1) | 1 passed in 0.40s (exit 0) |
| step4b | `tests/unit/scripts/test_repurchase_timeseries_charts.py::test_market_cap_is_split_invariant` | Restore chart script8c3b2bc; old API absent (compatibility proof only). | 1 error in 1.74s (exit 1) | 1 passed in 0.72s (exit 0) |
| step4c | `tests/integration/data/test_pgr_edgar_integrity.py::test_known_filed_values` | Replace copied DB with4b31e8b (before 10-Q repurchase repair). | 1 failed in 0.42s (exit 1) | 1 passed in 0.34s (exit 0) |
| step5 | `tests/integration/pipeline/test_validation_gating.py::test_compute_oos_r_squared_does_not_score_against_the_current_target` | Restore expanding mean including current y as naive. | 1 failed in 1.44s (exit 1) | 1 passed in 1.13s (exit 0) |
| step6 | `tests/unit/tax/test_tax_hand_computed.py::test_fully_appreciated_lot_breakeven_is_a_21_25_pct_fall` | Restore positive unsigned breakeven formula. | 1 failed in 0.42s (exit 1) | 1 passed in 0.36s (exit 0) |
| step7 | `tests/integration/repo/test_restructure_phase0.py::test_pyproject_declares_installable_package` | Remove pyproject (absent atb609af6). | 1 failed, 1 warning in 0.42s (exit 1) | 1 passed in 0.35s (exit 0) |
| step9 | `tests/integration/repo/test_test_suite_hygiene.py::test_ci_runs_artifact_tests_in_a_separate_job` | Restore CI workflow7b768a0 (before split artifact job). | 1 failed in 0.42s (exit 1) | 1 passed in 0.33s (exit 0) |
| step11 | `tests/integration/repo/test_restructure_phase4.py::test_claude_md_points_to_agents_md` | Restore old claude.md contents282a6b3 into CLAUDE.md. | 1 failed in 0.46s (exit 1) | 1 passed in 0.37s (exit 0) |
| step12 | `tests/integration/repo/test_restructure_phase5.py::test_monthly_decision_monolith_is_split` | Restore monthly monolith e45c047. | 1 failed in 0.43s (exit 1) | 1 passed in 0.35s (exit 0) |
| step10 | `tests/integration/repo/test_restructure_phase3.py::test_results_contains_no_python_files` | Restore and stage tracked results/research/v39_ridge_alpha.py fromf76cee2. | 1 failed in 0.43s (exit 1) | 1 passed in 0.37s (exit 0) |
| step4b-math | `tests/unit/scripts/test_repurchase_timeseries_charts.py::test_market_cap_is_split_invariant` | Remove shares × to_latest restatement in capital_return_data.py. | 1 error in 0.73s (exit 1) | 1 passed in 0.66s (exit 0) |
| step9-math | `tests/unit/models/test_wfo_min_rows.py::test_minimum_is_train_plus_gap_plus_two_test_windows` | Minimum rows=train+gap+one test window instead of two. | 2 failed in 1.26s (exit 1) | 2 passed in 1.11s (exit 0) |

These checks cover all fourteen implemented sessions, including restructuring sessions whose regression is structural rather than a mathematical change. They do not establish that every sub-fix survives an exhaustive mutation campaign. The historical step9 claim of 0/18 surviving mutations was not rerun as that entire campaign; the fresh reversals above are the actual evidence here.

### Read-only database hash record

| Checkpoint | SHA256 |
|---|---|
| Source before verification | `7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d` |
| Current scratch clone before/after full suite | Same |
| Current scratch after eight monthly replays | Same |
| Pre-restructure scratch after September replay | Same |
| Source after all verification and document authoring | Same (checked again before commit) |

## Data-integrity sweep

| Check | Fresh result | Interpretation / limit |
|---|---|---|
| SQLite integrity_check | ok | File structure, not economic accuracy. |
| Split/jump and weekly duplicate guards | 0 unexplained jumps;0 duplicate bars | VOO reverse split0.5 and VGT split8 present in canonical history. |
| Duplicate FRED calendar months | 0 | Calendar group, not merely exact-date uniqueness. |
| Monthly coverage | 265 rows;missing[] | Full PGR monthly range2004-08 through2026-08. |
| Revenue−expenses=pretax | 0 violations | Stored monthly fields and repository tolerances. |
| CR=LR+ER | 0 violations | Plus filed-specific Sep2025 test; identity alone insufficient. |
| Common equity≈BVPS×common shares | 0 violations | Subtract493.9M preferred equity during2018-03–2024-01 where applicable. |
| Monthly NI sum vs quarterly XBRL | 73 quarters;0 violations | Reconciles cached/stored XBRL, no fresh EDGAR source request. |
| PIF consistency | 0 jump violations | Stable auto-lines components; leap-Feb YoY independently inspected. |
| Current live features | Ridge12/GBT13;0 NaN | Decision2026-09-21 anchors latest fully completed feature month2026-08-31. |
| Latest momentum | -11.5032786% equals true calendar12M price return | Independently resampled split-adjusted monthly endpoints; this is a price feature, not DRIP target. |
| Current quarterly ROE | -0.9396% to38.6113% | No former4x annual/Q4 scaling; live monthly ROE stored in percent. |
| Stored 6M/12M relative targets vs rebuild | E2 relative-return column comparison passes | Manual split/fractional-DRIP return algorithm; missing dividends remain a source-data limit. |

### Freshness per ticker and series

All nine production price tickers PGR,VOO,VXUS,VWO,VMBS,BND,GLD,DBC,VDE end2026-09-25 (one day old at verification). ALL/TRV/CB/HIG peer prices end2026-09-18. The live freshness checker reports OK; that does **not** certify the separate dividend history.

| Dividend ticker(s) | Latest ex-date | Status on snapshot |
|---|---|---|
| PGR | 2026-10-01 (scheduled future event) | OK; future cash event excluded from past return windows. |
| VTI,VOO,VWO,VIG | 2025-12-22 | STALE |
| VGT,VHT,VFH,VIS,VDE,VPU,VNQ | 2026-03-24 | STALE |
| KIE | 2026-03-23 | STALE |
| VXUS,VEA | 2026-03-20 | STALE |
| SCHD | 2026-03-25 | STALE |
| BND,BNDX,VCIT,VMBS | 2026-03-02 | STALE |
| ALL | 2026-03-02 | STALE |
| TRV | 2026-09-10 | OK |
| CB | 2026-09-11 | OK |
| HIG | 2026-09-01 | OK |
| DBC | 2025-12-22 | OK under its annual cadence |
| GLD | no dividend history | NO_HISTORY; not automatically a missing-payment finding |

| FRED / stored macro series | Latest stored month | Scope / freshness |
|---|---|---|
| BAA10Y,BAMLH0A0HYM2,DCOILWTICO,DTWEXBGS,MORTGAGE30US,NFCI,T10Y2Y,T10YIE,THREEFYTP10,VIXCLS | 2026-09 | Stored September labels can be month-to-date, not final completed observations. Publication/feature lags still apply. |
| GS2,GS5,GS10,CUSR0000SAM2,CUSR0000SETA02,PCU5241265241261,PPIACO,WPU45110101 | 2026-08 | Live-required series meet their lagged required months. |
| MRTSSM447USN,TRFVOLUSM227NFWA | 2026-07 | Non-live monthly release-lagged inputs; inspect vintage/availability before new research. |
| CUSR0000SETE | 2017-12 | Research-only series is stale; no claim all24 series are fresh. |
| SP500_EARNINGS_YIELD_MULTPL,SP500_PE_RATIO_MULTPL,SP500_PRICE_TO_BOOK_MULTPL | 2026-04 | Research-only valuation series stale; source is Multpl, not a FRED release. |

All eleven live FRED series meet the checker's expected required months as of September26. Latest stored labels alone do not establish historical publication vintages. No refresh was run.

## Restructure and regression checks

- Last pre-restructure commit `b609af6` (step6) vs current `aae0be8`, identical copied DB and same Python environment, September21 dry-run with `--skip-fred`: **ten byte-identical files**: benchmark_quality.csv,classification_shadow.csv,consensus_shadow.csv,dashboard.html,decision_overlays.csv,diagnostic.md,monthly_summary.json,plots/calibration_curve.png,recommendation.md,signals.csv. Both processes exited0. Manifest differences are exactly `git_sha`, `run_timestamp_utc`, `script_name`. No other JSON field differs.
- This independently tests the combined steps7/10/11/12 endpoint, with step9 also between those commits. It is not proof for every CLI, input or platform; import/entrypoint tests cover additional seams. Earlier step5 behavior changes are analyzed separately below.
- `rg --files results --glob '*.py'`: no matches. Registry checker: `[registry] 113 studies, 0 problems`. Every current study folder is registered; legacy folders are intentionally outside that contract.
- Link checker before new docs: `[doc-links] 340 files, 0 broken links`; after both deliverables: `[doc-links] 342 files, 0 broken links`. Import smoke and entrypoint-import tests pass in E2.
- sys.path checker: `[sys-path] 0 new edits, 0 stale allowlist entries`. **The user's stronger zero-exceptions criterion fails:** 198 files edit sys.path, comprising197 existing allowlisted files plus tests/conftest.py. The checker prevents additions; it does not remove old hacks.
- New target/repurchase chart number changes have explicit rebuild/repair records (step2 target CSV, step3b cell diffs, step4b/4c reports). All eight committed monthly-summary JSON files compare exactly with their old-layout blobs atc948d0f; recommendations/health are unchanged historical records. I found no silent numerical regeneration in those summaries.
- No new shuffle=True or explicit KFold construction found in source/CLI/studies. Full-sample StandardScaler fits were not found by static fit-context inspection; this is bounded static evidence, not a certification of every estimator. Existing CombinatorialPurgedCV and17 old cv=None RidgeCV calls remain violations.
- v143 full-history correlation pruning and v138 proxy warmup use future outcomes/features; these are additional research side effects, not production scaling discoveries. They must not be imported into the new harness unchanged.
- CI artifact tests are separated from unit tests and all seven DB writers share db-writer. The peer bootstrap workflow still queries `MIN(price_date)` although the table has `date`; it can fail after work is done. Initial --force is still ignored, CI permissions are implicit and Actions use version tags rather than commit pins. No remote workflow run or email replay was initiated.
- Active CLI documentation follows the moves, but ROADMAP still says Python3.10 while packaging requires3.11. The generated footer deliberately retains the old script name for golden compatibility. Governance's step5 baseline remains useful history but differs from the current all-fix replay; it needs a clearly labeled current baseline in a follow-up.
- Current `src/pgr_vds/research_lib/` does not exist. Reusable historical helpers remain in `src/research/`. The new plan explicitly creates the required future research namespace rather than pretending step12 already moved it.

## Decision impact — February through latest committed month

Replay dates are the dates in each committed manifest, not synthetic month ends: February28,March31,April22,May22,June22,July22,August20,September21. As of verification September26, September remains the latest committed decision. DEFER abbreviates DEFER-TO-TAX-DEFAULT; MONITOR abbreviates MONITORING-ONLY. Sell percentages are recommendations, not transactions executed here.

The **current** IC column below is the equally weighted benchmark IC used by the gate. The pooled IC is a different panel statistic. Hit rate is the realised-only health-panel value, compared with the honest past-learned constant rule. Committed historical IC/hit columns in the second table are the old reported aggregates, so differences are not pure investment-performance improvements.

| As-of | Current mode | Sell | Consensus | OOS R² | Gate IC | Pooled IC | Hit | Base hit | Direction-skill p |
|---|---|---|---|---|---|---|---|---|---|
| 2026-02-28 | DEFER | 50.00% | UNDERPERFORM | 3.03% | 0.1007 | 0.1263 | 64.12% | 69.98% | 0.3641 |
| 2026-03-31 | DEFER | 50.00% | NEUTRAL | 5.65% | 0.1058 | 0.1320 | 63.80% | 69.61% | 0.3825 |
| 2026-04-22 | DEFER | 50.00% | NEUTRAL | 5.65% | 0.1058 | 0.1320 | 63.80% | 69.61% | 0.3825 |
| 2026-05-22 | DEFER | 50.00% | UNDERPERFORM | 3.05% | 0.0879 | 0.1076 | 63.08% | 69.42% | 0.4604 |
| 2026-06-22 | DEFER | 50.00% | UNDERPERFORM | 3.52% | 0.0730 | 0.0873 | 61.55% | 68.81% | 0.5757 |
| 2026-07-22 | DEFER | 50.00% | NEUTRAL | 5.13% | 0.0696 | 0.0899 | 61.74% | 68.31% | 0.5024 |
| 2026-08-20 | DEFER | 50.00% | NEUTRAL | 1.21% | 0.0487 | 0.0730 | 60.46% | 68.22% | 0.6154 |
| 2026-09-21 | DEFER | 50.00% | UNDERPERFORM | 2.86% | 0.0762 | 0.1019 | 62.83% | 68.14% | 0.3914 |

Every current row is LOW confidence, has directional-skill FAIL and CPCV4/7 MARGINAL (completion passes). March and April have identical current/step5 development metrics because their additional labels had not matured. ECE ranges14.37–19.29%; interval coverage40.63–48.96% against nominal80%. These are weaknesses, not evidence that the repairs made uncertainty reliable.

### What was committed at the time

| Decision month | Historical mode | Sell | Historical consensus | Old OOS R² | Old reported pooled IC | Old reported hit |
|---|---|---|---|---|---|---|
| 2026-02 | DEFER | 50.00% | UNDERPERFORM | -6.04% | 0.1990 | 66.89% |
| 2026-03 | DEFER | 50.00% | NEUTRAL | -4.53% | 0.1701 | 66.51% |
| 2026-04 | DEFER | 50.00% | NEUTRAL | -2.63% | 0.1530 | 66.24% |
| 2026-05 | DEFER | 50.00% | NEUTRAL | -3.51% | 0.0979 | 63.95% |
| 2026-06 | DEFER | 50.00% | NEUTRAL | -2.13% | 0.1282 | 63.71% |
| 2026-07 | DEFER | 50.00% | NEUTRAL | -2.78% | 0.1195 | 64.86% |
| 2026-08 | DEFER | 50.00% | UNDERPERFORM | -0.14% | 0.1806 | 65.36% |
| 2026-09 | DEFER | 50.00% | NEUTRAL | -1.13% | 0.1649 | 66.18% |

These JSON values are unchanged from the old-layout blobs atc948d0f. All eight DB health rows remain tagged `pre-2026-09-25`; dry runs correctly do not overwrite them. Comparing them to repaired forecasts is a before/after diagnosis, not a pre-registered holdout test.

### Published step5 re-baseline

| As-of | Step5 mode | Sell | Step5 consensus | OOS R² | Equal-weight IC | Pooled IC | Honest hit |
|---|---|---|---|---|---|---|---|
| 2026-02-28 | DEFER | 50.00% | NEUTRAL | 6.81% | 0.1046 | 0.1612 | 64.29% |
| 2026-03-31 | MONITOR | 50.00% | UNDERPERFORM | 7.61% | 0.1141 | 0.1642 | 65.40% |
| 2026-04-22 | MONITOR | 50.00% | UNDERPERFORM | 7.61% | 0.1141 | 0.1642 | 65.40% |
| 2026-05-22 | DEFER | 50.00% | UNDERPERFORM | 6.39% | 0.0923 | 0.1300 | 63.92% |
| 2026-06-22 | DEFER | 50.00% | UNDERPERFORM | 3.76% | 0.0907 | 0.1446 | 62.87% |
| 2026-07-22 | DEFER | 50.00% | NEUTRAL | 3.97% | 0.0762 | 0.1366 | 63.38% |
| 2026-08-20 | DEFER | 50.00% | UNDERPERFORM | 4.96% | 0.0876 | 0.1459 | 62.66% |
| 2026-09-21 | DEFER | 50.00% | NEUTRAL | 5.08% | 0.0706 | 0.1296 | 62.83% |

Source: [step5 CSV](2026-09-25_step5_rebaseline_rows.csv). For causal attribution I also replayed all eight dates at step5 merge `69889ad` on the **same current DB copy and runtime**, exit0, unchanged SHA256. Core WFO/prequential/evaluation/backtest metric code is unchanged between that merge andb609af6; feature_engineering gains filing-date placement. The current-vs-pre-restructure golden comparison is exact. Thus the substantial subsequent metric change is reproduced before restructuring, following the step6 timing repair; probability, confidence and consensus also reflect step6 policy/shadow corrections. Do not attribute it to file moves.

| As-of | Step5 code on same DB: R² / IC / hit | Current minus same-DB step5: R² points | Explanation of differences (one line per month) |
|---|---|---|---|
| 2026-02-28 | 6.81% / 0.1046 / 64.29% | -3.777 pp | Old→step5 removes leaky evaluation and repairs inputs; filing-date timing lowers R² and changes NEUTRAL→UNDERPERFORM; 50% default stays. |
| 2026-03-31 | 7.61% / 0.1141 / 65.40% | -1.962 pp | Old→step5 becomes MONITOR despite 50% sell; timing weakens directional significance, UNDERPERFORM→NEUTRAL and MONITOR→DEFER. |
| 2026-04-22 | 7.61% / 0.1141 / 65.40% | -1.962 pp | Same maturity support as March; corrected labels/metrics alter the old comparison, then timing changes MONITOR→DEFER and consensus→NEUTRAL. |
| 2026-05-22 | 6.39% / 0.0923 / 63.92% | -3.340 pp | Old NEUTRAL→UNDERPERFORM after data/metric repairs; filing-date timing lowers R² further, but directional skill still forces the same default. |
| 2026-06-22 | 3.76% / 0.0907 / 62.87% | -0.242 pp | Old NEUTRAL→UNDERPERFORM; filing-date timing modestly lowers R²/IC; no actionable directional edge, default unchanged. |
| 2026-07-22 | 3.97% / 0.0762 / 63.38% | +1.158 pp | Consensus stays NEUTRAL; filing-date timing improves R² but reduces IC/hit; revised inference still does not pass the skill gate. |
| 2026-08-20 | 4.96% / 0.0877 / 62.66% | -3.754 pp | Old/step5 UNDERPERFORM→current NEUTRAL; timing lowers R² below2% and skill fails; default unchanged. |
| 2026-09-21 | 5.08% / 0.0707 / 62.83% | -2.213 pp | Old/step5 NEUTRAL→current UNDERPERFORM; timing lowers R² and changes calibrated directional support; skill fails, default unchanged. |

The same-code/same-current-DB step5 run matches the published rebaseline closely, but not perfectly: August R² .049614215 vs published .0496031 and September .050775316 vs .0507609. Those small historical snapshot/runtime differences were not causally isolated; they are not blamed on restructuring. The main same-DB code effect above is independently reproduced. September current calibration is .658071 outperform probability versus a negative forecast; LOW reflects weak support in that forecast direction, not constant confidence.

## New issues and unresolved side effects

Severity denotes practical risk, not a claim that every item was introduced by the repairs.

| ID / severity | Evidence | Suggested next action |
|---|---|---|
| V01 HIGH — Windows test DB guard URI escape | Full-suite node args11 returnsNone for file:///C:/...?...mode=ro; tests/repo_guard.py path normalization uses URL path with drive-leading slash. | Normalize file URIs/drive letters correctly before resolve/classification; add Windows URI red/green probes. Keep scratch copies mandatory meanwhile. |
| V02 MEDIUM — Windows path-dependent tests | Full-suite email regex and nested probe-node parser expect slash paths; actual paths have backslashes. | Match normalized Path values / parse structured pytest outcomes; preserve real guard-block assertions. |
| V03 HIGH — Dividend repair incomplete | E1 lists20 stale tickers, including live VOO,VWO,VXUS,VMBS,BND,VDE. Targets recompute but omit missing events. Existing freshness code does not make the data fresh. | Separately authorize feed backfill plus explicit rebuild/migration and reviewed row diff; block clean v200 until required histories pass. F08 remains PARTIAL. |
| V04 HIGH — No virgin historical promotion holdout | Prior studies and13V have already inspected latest mature24-month periods. | Quarantine once for retrospective synthesis; preregister untouched forward origins. Owner decides to wait or retain shadow, never relabel history as unused. |
| V05 MEDIUM — v143 feature selection sees future rows | v143 runner calls prune_feature_overrides on baseline full feature_df; src/research/v139_utils.py correlates whole supplied frame. | Compute pruning only inside each inner training fold; future-feature perturbation regression test in new harness. |
| V06 MEDIUM — v138/149/150 proxy warmup sees future outcomes | src/research/v138_utils.py _build_proxy_frame uses residual_sq.mean() for initial residual MSE and realized.var(ddof=0) for prior variance over all rows. | Past-only declared prior and maturity-aware warmup; test future-outcome perturbations cannot change earlier scores. |
| V07 MEDIUM — Firth adoption and follow-on performance overstated | Backlog claims v159 integration; current tree lacks the implementation/closeout. shadow_followon forwards baseline probabilities with candidate metadata. | Correct current-use claims in separate docs PR; require distinct stored candidate predictions and implementation evidence before tracking as live. |
| V08 MEDIUM — Current governance baseline stale | Governance cites step5 SeptR²5.08%, pooledIC.130/p.030 and5/7GOOD; current same-DB all-fix replay2.86%,.10194/p.12542,4/7MARGINAL. | Retain labeled step5 history but add current all-fix baseline and inherited dividend/data limitations; do not silently edit old artifacts. |
| V09 MEDIUM — Workflow tail failure persists | .github/workflows/peer_bootstrap.yml:119 SELECT MIN(price_date) FROMdaily_prices; schema column isdate. | Fix query and add schema-backed offline workflow smoke; audit ignored initial --force separately. |
| V10 LOW — Requested layout/check stricter than implementation | 197 sys.path exceptions; missing src/pgr_vds/research_lib; lint does not enforce strictPEP8; ROADMAP3.10 vs package3.11. | Create new research namespace in v200; staged removal of old exceptions/docs cleanup separately. Do not claim zero hacks based on no-new-edits checker. |

Remaining original research defects (F25/F30/F31), forbidden production CPCV (F02), current-vintage macro/CB/proxy issues (F27), TA outcome maturation (F24), suspected vest-event offset (F22) and fiscal-month hypothesis (F16) are explicit limits in the finding table. No fix for them is smuggled into this two-document PR.

## Verification boundaries

No remote provider fact revalidation, historical vintage reconstruction, real email, remote workflow execution or full rerun of every archived research study was performed. Stored XBRL comparisons are independent table reconciliation, not a fresh SEC download. Target formulas passing against the current DB do not certify absent dividends. Static review does not prove every estimator path free of leakage. Red/green coverage is one or more selected regressions per implemented step, not exhaustive reversal of every sub-fix. The audit found genuine progress and genuine remaining blockers; it does not approve a model promotion.

## Appendix A — reconstruct the DB/feature evidence command

Save the following as `sweep.py` **outside the source repository**, then run `python <temporary>/sweep.py` with cwd set to a scratch clone at the audited commit and a copied DB. Its feature cache and CSV are redirected to that clone's external parent. This is the actual audit script used (formatting preserved); no fetcher is invoked.

```python
from __future__ import annotations
import sys,sqlite3,hashlib,json,pathlib,datetime
import numpy as np
import pandas as pd
import config
from src.database import db_client
from src.processing import feature_engineering as fe,pgr_edgar_validation as v
from src.processing.price_integrity import find_unexplained_price_jumps,find_duplicate_week_bars
from src.processing.price_adjustment import split_adjusted_close,calendar_momentum
from src.processing.multi_total_return import build_etf_monthly_returns
root=pathlib.Path.cwd()
out=root.parent
p=root/'data/pgr_financials.db'
conn=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True)
conn.row_factory=sqlite3.Row
fe._PROCESSED_PATH=str(out/'feature_matrix.parquet')
before=hashlib.sha256(p.read_bytes()).hexdigest()
print('DB',before,'integrity',conn.execute('pragma integrity_check').fetchone()[0])
print('SPLITS',[(r['ticker'],r['split_date'],r['split_ratio']) for r in conn.execute("select * from split_history where ticker in ('VOO','VGT')")])
print('PRICE_JUMPS',find_unexplained_price_jumps(conn).to_dict('records'))
print('WEEK_DUPES',find_duplicate_week_bars(conn).to_dict('records'))
print('FRED_DUPES', [tuple(r) for r in conn.execute("select series_id,substr(month_end,1,7),count(*) from fred_macro_monthly group by 1,2 having count(*)>1")])
print('VIX_APR2020', [tuple(r) for r in conn.execute("select * from fred_macro_monthly where series_id='VIXCLS' and month_end='2020-04-30'")])
monthly=db_client.get_pgr_edgar_monthly(conn)
quarterly=db_client.get_pgr_fundamentals(conn)
print('EDGAR_ROWS',len(monthly),'QUARTERS',len(quarterly),'MISSING',v.missing_months(monthly))
for name,f in [('income',v.income_identity_violations),('CR',v.combined_ratio_violations),('equity',v.equity_violations),('PIF_jump',v.pif_jump_violations)]:
 print('IDENTITY',name,len(f(monthly)),f(monthly).to_dict('records'))
print('NI_RECONCILE',v.quarters_compared(monthly,quarterly),v.quarterly_net_income_violations(monthly,quarterly).to_dict('records'))
for date,fields in [('2017-08-31',['net_income','eps_diluted','income_before_income_taxes']),('2025-09-30',['combined_ratio','loss_lae_ratio','expense_ratio']),('2015-05-31',['net_premiums_written','combined_ratio']),('2019-04-30',['net_income','book_value_per_share']),('2016-05-31',['npw_growth_yoy']),('2020-04-30',['npw_growth_yoy']),('2021-02-28',['pif_growth_yoy']),('2025-02-28',['pif_growth_yoy']),('2004-12-31',['shareholders_equity','debt'])]:
 print('KNOWN',date,monthly.loc[date,fields].to_dict())
print('BOOK_YIELD_RANGE',float(monthly.investment_book_yield.min()),float(monthly.investment_book_yield.max()))
print('ROE_NULL_2026',monthly.loc['2026-02':'2026-08','roe_net_income_ttm'].to_dict())
print('QUARTER_ROE',quarterly.roe.min(),quarterly.roe.max(),'q4_eps_median',quarterly[quarterly.index.month==12].eps.median())
print('FRED_FRESHNESS_ALL', [tuple(r) for r in conn.execute('select series_id,min(month_end),max(month_end),count(*) from fred_macro_monthly group by series_id')])
fresh=db_client.check_data_freshness(conn,datetime.date(2026,9,26))
print('LIVE_FRESHNESS',json.dumps(fresh,default=str))
print('DIVIDEND_FRESHNESS',json.dumps(db_client.check_dividend_freshness(conn),default=str))
features=fe.build_feature_matrix_from_db(conn,force_refresh=True)
features.to_csv(out/'verified_features.csv')
row=features.loc[features.index<=pd.Timestamp('2026-09-21')].iloc[-1]
print('LIVE_ROW',str(row.name.date()))
for model in ('ridge','gbt'):
 cols=config.MODEL_FEATURE_OVERRIDES[model]
 print('LIVE_FEATURES',model,{col:float(row[col]) for col in cols},'NaN',[col for col in cols if pd.isna(row[col])])
raw=db_client.get_prices(conn,'PGR').close
adjusted=split_adjusted_close(raw,db_client.get_splits(conn,'PGR'))
monthly_close=adjusted.resample('BME').last()
true_mom=float(monthly_close.loc[row.name]/monthly_close.shift(12).loc[row.name]-1)
print('MOM12_TRUE',true_mom,'FEATURE',row['mom_12m'],'EQUAL',abs(true_mom-row['mom_12m'])<1e-12)
for ticker,dates in [('VOO',['2013-04-30','2013-09-30']),('VGT',['2025-10-31','2026-02-27'])]:
 recompute=build_etf_monthly_returns(conn,ticker,6)
 for date in dates:
  stored=conn.execute('select benchmark_return,relative_return from monthly_relative_returns where benchmark=? and target_horizon=6 and date=?',(ticker,date)).fetchone()
  print('TARGET',ticker,date,'recomputed',recompute.loc[date],'stored',tuple(stored),'gap',recompute.loc[date]-stored[0])
print('SPLIT_PB_BVPS',features.loc['2006-05':'2007-06',['pb_ratio','book_value_per_share_growth_yoy']].to_string())
print('CB_SOURCES',[tuple(r) for r in conn.execute("select source,proxy_fill,min(date),max(date),count(*) from daily_prices where ticker='CB' group by source,proxy_fill")])
print('MODEL_LOG', [dict(r) for r in conn.execute('select month_end,metrics_version,aggregate_oos_r2,aggregate_nw_ic,aggregate_hit_rate from model_performance_log')])
conn.close()
after=hashlib.sha256(p.read_bytes()).hexdigest()
print('DB_AFTER',after,'UNCHANGED',before==after)

```

## Appendix B — history and output comparison commands

For each location listed above, run `git log --follow --name-status -- <current_path>` and read the current function plus its original source body. For extraction additions inspect both old monolith history and the split commit; --follow alone cannot track one-to-many extraction. Representative fresh output:

```text
9fcf367 R062 scripts/edgar_8k_fetcher.py src/pgr_vds/ingestion/edgar_monthly/parse.py
aca5b92 A src/pgr_vds/decision/pipeline.py (original functions from scripts/monthly_decision.py)
02e3fd4 R098 results/research/v39_ridge_alpha.py research/studies/v39_ridge_alpha/v39_ridge_alpha.py
95e0c83 moved cited tests into unit/integration/research folders
```

All workflows, docs and moved tests cited by the original report were included in follow/source inspection. Other useful commands actually used:

```powershell
git diff --stat 69889ad b609af6 -- src/models/wfo_engine.py src/models/prequential.py src/models/evaluation.py src/reporting/backtest_report.py src/processing/feature_engineering.py
rg -n 'price_date' .github/workflows/peer_bootstrap.yml
rg -n 'cv=None' src cli scripts research --glob '*.py'
rg -n '\.fit_transform\(|scaler\.fit\(|scale\.fit\(|shuffle\(' src cli scripts research --glob '*.py'
rg -l 'sys\.path\.(insert|append)|sys\.path\s*=' --glob '*.py' .
Get-FileHash -Algorithm SHA256 -LiteralPath data/pgr_financials.db
```

The comparison script iterated the union of both replay output trees, compared bytes, and for differing JSON listed top-level changed fields. It also compared each committed monthly-summary JSON with `git show c948d0f:results/monthly_decisions/<month>/monthly_summary.json`. Thus a missing file or changed recommendation/health could not be hidden by checking only one metric.

## What changed / what is left

This PR adds this independent verification and [the v200 research prompt plan](../research/RERUN_PLAN_v200_codex.md), with complete finding/history maps, fresh tests, counterfactuals, data checks, decision impact and per-study inventory. It changes no production behavior or data. Separate remediation must finish dividends, Windows guard portability, prohibited validation and outstanding research/data/layout issues. v200 starts only after its clean-data prerequisites; no live promotion is endorsed here.

## Appendix C — independent manual split/DRIP check

Executed in the scratch current checkout; only standard-library SQLite/calendar/date/math calls, no production return helpers. The four selected start/end windows include the review's October25 VOO and April24 VGT weekly bars. Product of split ratios and fractional DRIP factors times endpoint price ratio independently validates those stored benchmark and relative targets, while E2 checks full-table consistency with production code.

```python
from pathlib import Path
import sqlite3,calendar,datetime,math
c=sqlite3.connect((Path.cwd()/'data/pgr_financials.db').as_uri()+'?mode=ro&immutable=1',uri=True)
def manual(ticker,anchor):
 t=datetime.date.fromisoformat(anchor);m=t.month+6;yr=t.year+(m-1)//12;m=(m-1)%12+1
 end=datetime.date(yr,m,calendar.monthrange(yr,m)[1])
 while end.weekday()>4:end-=datetime.timedelta(days=1)
 start_bar=c.execute('select date,close from daily_prices where ticker=? and date<=? order by date desc limit1'.replace('limit1','limit 1'),(ticker,anchor)).fetchone()
 end_bar=c.execute('select date,close from daily_prices where ticker=? and date<=? order by date desc limit 1',(ticker,end.isoformat())).fetchone()
 events=[(d,'split',v) for d,v in c.execute('select split_date,split_ratio from split_history where ticker=? and split_date>? and split_date<=?',(ticker,start_bar[0],end_bar[0]))]
 events +=[(d,'div',v) for d,v in c.execute('select ex_date,amount from daily_dividends where ticker=? and ex_date>? and ex_date<=?',(ticker,start_bar[0],end_bar[0]))]
 factors=[]
 for date,kind,value in sorted(events):
  if kind=='split':factors.append(value)
  else:
   price=c.execute('select close from daily_prices where ticker=? and date<=? order by date desc limit 1',(ticker,date)).fetchone()[0]
   factors.append(1+value/price)
 result=math.prod(factors)*end_bar[1]/start_bar[1]-1
 return result,start_bar[0],end_bar[0]
for ticker,anchor in [('VOO','2013-04-30'),('VOO','2013-09-30'),('VGT','2025-10-31'),('VGT','2026-02-27')]:
 ret,start,end=manual(ticker,anchor);pgr,_,_=manual('PGR',anchor)
 stored=c.execute('select benchmark_return,relative_return from monthly_relative_returns where benchmark=? and date=? and target_horizon=6',(ticker,anchor)).fetchone()
 print(ticker,anchor,'bars',start,end,'manualbenchmark',ret,'stored',stored,'benchmarkgap',ret-stored[0],'relativegap',pgr-ret-stored[1])
c.close()


```

Actual output:

```text
VOO 2013-04-30 bars 2013-04-26 2013-10-25 manualbenchmark 0.12384860039652779 stored (0.12384860039652801, -0.07382875125004151) benchmarkgap -2.220446049250313e-16 relativegap 4.440892098500626e-16
VOO 2013-09-30 bars 2013-09-27 2014-03-28 manualbenchmark 0.10813111425697564 stored (0.10813111425697564, -0.1949929193861496) benchmarkgap 0.0 relativegap 0.0
VGT 2025-10-31 bars 2025-10-31 2026-04-24 manualbenchmark 0.05268029089126691 stored (0.05268029089126647, -0.01443039900522436) benchmarkgap 4.440892098500626e-16 relativegap -6.661338147750939e-16
VGT 2026-02-27 bars 2026-02-27 2026-08-28 manualbenchmark 0.32320522639905613 stored (0.3232052263990559, -0.29893217654462334) benchmarkgap 2.220446049250313e-16 relativegap 0.0

```

## Appendix D — remaining utility/workflow symptoms

```python
from pathlib import Path
import sqlite3,numpy as np,pandas as pd
from src.models.blp import BLPModel
from src.processing.feature_engineering import apply_fracdiff,_fracdiff_weights
rng=np.random.default_rng(20260926)
p=rng.uniform(.15,.85,(100,4));y=(rng.random(100)>.5).astype(float)
a=BLPModel(4).fit(p,y).params_;b=BLPModel(4).fit(p,1-y).params_
print('BLP y versus1-y identical',a==b,'parameters',a)
for d in [.1,.2,.3,.4,.5]:print('FFD lags',d,len(_fracdiff_weights(d,10000)))
for n in [60,323]:
 s=pd.Series(np.cumsum(np.random.default_rng(0).normal(0,.02,n))+5,index=pd.date_range('1990-01-31',periods=n,freq='ME'))
 out,d=apply_fracdiff(s);print('FFD monthly rows',n,'d',d,'usable outputs',int(out.notna().sum()))
c=sqlite3.connect((Path.cwd()/'data/pgr_financials.db').as_uri()+'?mode=ro&immutable=1',uri=True)
try:c.execute('SELECT MIN(price_date) FROM daily_prices')
except sqlite3.OperationalError as e:print('PEER query error',str(e))
c.close()

```

Actual substantive output (an ADF-library future-warning omitted):

```text
BLP y versus1-y identical True parameters BLPParams(a=11.124711583691132, b=11.256249277272834, weights=[0.31307696697855514, 0.16216663867114017, 0.25941430880639177, 0.2653420855439129], log_likelihood=-85.16853268167705, converged=True)
FFD lags 0.1 4076
FFD lags 0.2 3382
FFD lags 0.3 2275
FFD lags 0.4 1458
FFD lags 0.5 927
FFD monthly rows 60 d 0.5 usable outputs 1
FFD monthly rows 323 d 0.5 usable outputs 1
PEER query error no such column: price_date
```
