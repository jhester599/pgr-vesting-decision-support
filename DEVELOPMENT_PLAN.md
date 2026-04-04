# PGR Vesting Decision Support — Development Plan
*Last updated: April 2026 | System version: v6.1*

> Historical planning note: this document captures the pre-v6.5 backlog that fed
> the later v6.5/v7.x work. For current completion status and the latest
> enhancement roadmap, see `docs/history/SESSION_PROGRESS.md`,
> `docs/history/claude-v7-plan.md`, and `docs/plans/codex-v8-plan.md`. The
> production ensemble now also uses model-specific
> feature sets from the completed v7.0/v8.x ablation work.

## Executive Summary

The PGR Vesting Decision Support system is a fully automated monthly pipeline that ingests multi-source financial data, engineers insurance-specific features, and produces a calibrated sell-percentage recommendation for PGR RSU vesting decisions. Each month the system fetches OHLCV prices and dividends for 22 tickers via Alpha Vantage, 12 FRED macro series, SEC EDGAR XBRL quarterly fundamentals, and PGR monthly 8-K operating results. It computes monthly relative returns of PGR versus ETF benchmarks, runs a 4-model ensemble (ElasticNet, Ridge, BayesianRidge, GBT) with CPCV walk-forward validation, applies Platt/isotonic calibration and conformal prediction intervals, and sizes the recommendation via Black-Litterman + Kelly. Outputs include signals.csv, recommendation.md, decision_log.md, and an emailed report delivered on the first business day of each month.

The primary data gap is that `pgr_edgar_monthly` contains only 22 months of live-fetched data and captures just 10 of the 65 available fields from PGR's monthly 8-K filings. The full `pgr_edgar_cache.csv` (256 rows, 2004–2026, 65 columns) sits on disk unused, representing over 20 years of high-quality insurance operating metrics that could dramatically improve model signal. Additionally, segment-level data (agency vs. direct channel mix, property segment, commercial lines) and investment portfolio metrics are entirely absent from the feature set despite being well-documented leading indicators of PGR earnings revisions.

The top opportunities ranked by decision-support value are: (1) loading the historical cache CSV into the DB to give the model 20+ years of training data instead of 22 months; (2) expanding the pgr_edgar_monthly schema to capture segment-level NPW/PIF and investment portfolio fields, then deriving channel-mix and underwriting-income features that directly signal margin inflections; (3) fixing the 3 pre-existing CI test failures to restore confidence in the test suite before adding new features; (4) adding ROE, P/B ratio, and buyback yield as valuation anchors; and (5) implementing property-segment CAT exposure tracking as a short-term earnings volatility signal.

## Current System State

| Data Source | Fields Currently Captured | Update Frequency | DB Table | Status |
|---|---|---|---|---|
| Alpha Vantage | OHLCV + dividends, 22 tickers | Weekly (Sun 18:00 UTC) | daily_prices, daily_dividends | ✅ Active |
| FRED Macro | 12 series (CPI, rates, employment, insurance CPI) | Monthly via scheduled fetch | fred_macro_monthly | ✅ Active |
| SEC EDGAR XBRL | Quarterly fundamentals (revenue, net income, EPS, operating cash flow) | Weekly | pgr_fundamentals_quarterly | ✅ Active |
| PGR Monthly 8-K | combined_ratio, pif_total, net_premiums_written, net_premiums_earned, net_income, eps_diluted, loss_ratio, expense_ratio, pif_growth_yoy, gainshare_estimate | Monthly (20th + 25th fallback) | pgr_edgar_monthly | ✅ Active (22 months) |
| PGR 8-K Cache CSV | 65 columns, 2004–2026, 256 rows | Static | Not yet in DB | ⚠️ Partially loaded |

## Priority 1 — High Value, Next 1–7 Days

### P1.1 — Load pgr_edgar_cache.csv into DB (Historical Backfill)
**Addresses:** The cache CSV has 256 rows (2004–2026) and 65 columns, but pgr_edgar_monthly only has ~22 rows from the recent EDGAR fetch. All the historical data is sitting unused on disk.
**Approach:** Write a `--load-from-csv` path in edgar_8k_fetcher.py (or a standalone script) that reads the full CSV, maps its 65 columns to the expanded schema (see Section 5), and upserts all rows. Run it once to seed the DB.
**Complexity:** S
**Dependencies:** P1.2 (schema expansion should happen first, or simultaneously)

### P1.2 — Expand pgr_edgar_monthly Schema (Phase 1: High-Signal Fields)
**Addresses:** The current schema captures only 10 fields out of 65 available. The missing segment-level NPW/PIF data and investment portfolio metrics are directly relevant to the decision model.
**Approach:** Add the following columns to pgr_edgar_monthly (ALTER TABLE or migration): `npw_agency`, `npw_direct`, `npw_commercial`, `npw_property`, `npe_agency`, `npe_direct`, `npe_commercial`, `npe_property`, `pif_agency_auto`, `pif_direct_auto`, `pif_commercial_lines`, `pif_total_personal_lines`, `investment_income`, `total_revenues`, `total_expenses`, `income_before_income_taxes`, `eps_basic`, `book_value_per_share`, `roe_net_income_trailing_12m`, `shares_repurchased`, `total_assets`, `shareholders_equity`. Update schema.sql, db_client.py upsert, and edgar_8k_fetcher.py parser to populate.
**Complexity:** M
**Dependencies:** None

### P1.3 — Fix Pre-existing Test Failures in test_multi_ticker_loader.py
**Addresses:** 3 tests are failing on master, which undermines CI confidence and masks new regressions.
**Approach:** Inspect the 3 failing tests. They likely use a stale AV API mock that doesn't match the current multi-ticker loader interface. Update the mocks or test fixtures to match the current function signatures.
**Complexity:** S
**Dependencies:** None

### P1.4 — Add Channel-Mix Features to Monthly Decision Model
**Addresses:** Agency vs. direct auto channel mix (npw_agency / npw_direct ratio) is a documented ROADMAP candidate feature and a strong leading indicator of margin compression or growth inflection.
**Approach:** After P1.2, add to feature_engineering.py: `channel_mix_agency_pct = npw_agency / (npw_agency + npw_direct)`, `npw_growth_yoy` for each segment. These feed into the ElasticNet/GBT models.
**Complexity:** S
**Dependencies:** P1.2, P1.1

### P1.5 — Merge All Open PRs / Resolve Branch Debt
**Addresses:** Multiple PRs are open (PRs #11, #12, #13 and potentially others). Branch divergence creates merge risk.
**Approach:** Review all open PRs, resolve any conflicts, merge in dependency order: EDGAR XBRL → 8-K parser → retry+backfill.
**Complexity:** S
**Dependencies:** None

## Priority 2 — Near-term, Next 2–4 Weeks

### P2.1 — Expand 8-K Schema Phase 2: Investment Portfolio & Capital Allocation Metrics
**Addresses:** PGR's investment portfolio is a major earnings contributor. `investment_income`, `fte_return_total_portfolio`, `investment_book_yield`, `net_unrealized_gains_fixed`, `fixed_income_duration`, and `weighted_avg_credit_quality` are all in the cache and directly impact earnings quality.
**Approach:** Add investment portfolio columns to pgr_edgar_monthly. Derive `investment_income_growth_yoy`. Consider adding `net_unrealized_gains_fixed` as a regime indicator (rising unrealized gains → rising rate environment signal).
**Complexity:** M
**Dependencies:** P1.2

### P2.2 — Add Underwriting Income as a First-Class Feature
**Addresses:** `underwriting_income = net_premiums_earned × (1 - combined_ratio/100)` is a more direct signal of core profitability than combined_ratio alone. It's noted in ROADMAP but not implemented.
**Approach:** Derive `underwriting_income` and `underwriting_income_growth_yoy` in feature_engineering.py. Add 3-month and 12-month trailing averages.
**Complexity:** S
**Dependencies:** P1.1

### P2.3 — Unearned Premium Reserve Growth as Leading Indicator
**Addresses:** `unearned_premiums` growth is a leading indicator of earned premium growth in future quarters. The cache has this field but it's not being used.
**Approach:** Add `unearned_premium_growth_yoy` and `unearned_premium_to_npw_ratio` features. This captures the "pipeline" of future revenue.
**Complexity:** S
**Dependencies:** P1.2

### P2.4 — ROE and Book Value Features
**Addresses:** `roe_net_income_trailing_12m` and `book_value_per_share` are standard insurance valuation anchors. P/B ratio (price / book_value_per_share) is noted in ROADMAP v6.x as in-progress.
**Approach:** Add `pb_ratio` (daily_price / book_value_per_share) and `roe_trend` (current ROE vs 12-month average) to feature set. Both are high-quality valuation signals for vesting decisions.
**Complexity:** S
**Dependencies:** P1.2

### P2.5 — Share Repurchase Signal
**Addresses:** `shares_repurchased` is a management confidence signal. Accelerating buybacks relative to the float often precede price appreciation.
**Approach:** Derive `buyback_yield = (shares_repurchased × avg_cost_per_share) / market_cap` and `buyback_acceleration` (current vs. trailing 4-quarter average). Add as features.
**Complexity:** S
**Dependencies:** P1.2

### P2.6 — Extend Monthly 8-K EDGAR Fetch to Capture All Phase 1+2 Fields
**Addresses:** New schema fields need to be populated going forward from EDGAR filings, not just the CSV backfill.
**Approach:** Update the HTML parser in edgar_8k_fetcher.py to extract the additional fields (segment NPW table, investment portfolio table, balance sheet summary). These appear in separate tables within the same 8-K exhibit.
**Complexity:** M
**Dependencies:** P1.2

### P2.7 — Add Calibration Diagnostic to Monthly Report
**Addresses:** The system uses Platt scaling and isotonic regression for probability calibration, but the report doesn't currently show how well-calibrated the ensemble is.
**Approach:** Add a calibration curve plot and Expected Calibration Error (ECE) metric to the diagnostic section of the monthly report. This helps validate model confidence over time.
**Complexity:** M
**Dependencies:** None

### P2.8 — Automated Email Delivery Testing
**Addresses:** The email delivery in v6.1 is untested in CI — a broken SMTP config would silently fail.
**Approach:** Add a test that mocks the SMTP connection and verifies the email is constructed correctly (subject, body, attachments). Add a `--dry-run` flag to monthly_decision.py that prints the email content without sending.
**Complexity:** S
**Dependencies:** None

## Priority 3 — Strategic / Longer-term

### P3.1 — Personal Lines Loss Ratio by Channel (Agency vs. Direct) as Regime Feature
**Addresses:** Agency and direct channels have different risk profiles. When direct combined ratio diverges from agency, it signals underwriting discipline changes that precede earnings revisions. The CSV has both NPW and NPE by channel, enabling channel-level combined ratio estimation.
**Approach:** Derive `est_combined_ratio_agency = losses_lae × (npe_agency/npe_total) / npe_agency` as a rough channel-level loss ratio proxy. Add as a feature with appropriate lag.
**Complexity:** M
**Dependencies:** P2.6

### P3.2 — Property Segment Tracking (Weather/CAT Exposure Signal)
**Addresses:** PGR's property segment (`npw_property`, `npe_property`, `pif_property`) is volatile due to weather/CAT events. A spike in property combined ratio is a short-term earnings headwind. The cache has these fields.
**Approach:** Add property-specific features: `npw_property_growth_yoy`, `property_mix_pct = npw_property / npw_total`. Cross with NOAA CAT event data or use FRED property insurance CPI as a proxy.
**Complexity:** M
**Dependencies:** P2.6

### P3.3 — Commercial Lines Monitoring
**Addresses:** PGR's commercial auto segment is ~10% of premium but highly cyclical. `pif_commercial_lines` and `npw_commercial` trends signal market share gains/losses in a segment with different margin dynamics.
**Approach:** Add `commercial_mix_pct` and `commercial_npw_growth_yoy`. Use as auxiliary features in the GBT model specifically.
**Complexity:** S
**Dependencies:** P2.6

### P3.4 — Extend Historical Context: Pre-2023 EDGAR Fetch
**Addresses:** pgr_edgar_monthly currently has only 22 months from the live EDGAR fetcher. The cache CSV has 256 rows back to 2004. Once schema is expanded, older rows can be re-parsed from archived 8-K filings for the new columns.
**Approach:** Run full historical backfill via `--load-from-csv` after P1.2 schema expansion. For fields not in the CSV, attempt EDGAR HTML re-parse for filings back to ~2010.
**Complexity:** M
**Dependencies:** P1.1, P1.2, P2.6

### P3.5 — Conformal Interval Tracking Dashboard
**Addresses:** The system generates conformal prediction intervals (80% nominal coverage) but there's no visualization of interval width over time or coverage validation.
**Approach:** Add a rolling coverage plot to the monthly diagnostic report: for each past month, did the actual return fall within the predicted interval? Track empirical coverage rate. Alert if coverage drops below 70%.
**Complexity:** M
**Dependencies:** None

### P3.6 — BLP (Bayesian Long-term Predictor) Out-of-Sample Validation
**Addresses:** ROADMAP v6.0 includes a BLP framework that needs ~12 months of live OOS predictions before it can be properly validated.
**Approach:** Ensure decision_log.md is capturing all predictions with timestamps. After 12 months of live data (target: Q1 2027), run formal BLP validation: Diebold-Mariano test vs. ensemble baseline, conditional coverage tests.
**Complexity:** L
**Dependencies:** 12 months of live OOS data

### P3.7 — Peer Comparison Expansion (ALL, TRV, CB, HIG)
**Addresses:** v6.0 added 4 peer tickers for relative comparison features. The peer fundamentals (quarterly EDGAR XBRL) could be enriched with the same treatment as PGR — combined ratio proxies from their own 8-K filings where available.
**Approach:** Implement EDGAR 8-K fetchers for ALL, TRV, CB, HIG (each has their own filing format). Store peer monthly metrics. Add peer combined ratio spread as a feature.
**Complexity:** L
**Dependencies:** P2.6 (validate PGR fetcher robustness first)

## Monthly 8-K Data Expansion

### Fields Currently Captured (pgr_edgar_monthly, 10 fields + 2 derived)

| Field | Source |
|---|---|
| combined_ratio | Direct parse |
| pif_total | Direct parse |
| net_premiums_written | Direct parse |
| net_premiums_earned | Direct parse |
| net_income | Direct parse |
| eps_diluted | Direct parse |
| loss_ratio (loss_lae_ratio) | Direct parse |
| expense_ratio | Direct parse |
| pif_growth_yoy | Derived |
| gainshare_estimate | Derived (0–2 scale) |

### Fields in pgr_edgar_cache.csv NOT Currently Fetched/Stored

| Field | Decision Support Value | Segment | Priority |
|---|---|---|---|
| npw_agency | Channel mix signal; agency channel margin trends | Personal Lines Auto | **P1** |
| npw_direct | Channel mix signal; direct channel growth rate | Personal Lines Auto | **P1** |
| npw_commercial | Commercial cycle exposure | Commercial Auto | **P1** |
| npw_property | CAT/weather earnings volatility | Property | **P1** |
| npe_agency | Earned premium recognition by channel | Personal Lines Auto | **P1** |
| npe_direct | Earned premium recognition by channel | Personal Lines Auto | **P1** |
| npe_commercial | Commercial earned premium | Commercial Auto | **P2** |
| npe_property | Property earned premium | Property | **P2** |
| pif_agency_auto | Channel share stability | Personal Lines Auto | **P1** |
| pif_direct_auto | Direct channel growth (leading indicator) | Personal Lines Auto | **P1** |
| pif_commercial_lines | Commercial market share | Commercial Auto | **P2** |
| pif_total_personal_lines | Personal lines total size | Personal Lines | **P2** |
| investment_income | Investment contribution to earnings (rate environment proxy) | Companywide | **P1** |
| total_revenues | Top-line growth signal | Companywide | **P1** |
| book_value_per_share | P/B valuation anchor | Companywide | **P1** |
| roe_net_income_trailing_12m | Capital efficiency | Companywide | **P1** |
| shares_repurchased | Buyback signal / management confidence | Companywide | **P2** |
| avg_cost_per_share | Buyback cost basis | Companywide | **P2** |
| total_assets | Balance sheet size | Companywide | **P3** |
| shareholders_equity | Book value base | Companywide | **P2** |
| unearned_premiums | Leading earned premium indicator | Companywide | **P2** |
| loss_lae_reserves | Reserve adequacy | Companywide | **P3** |
| debt | Leverage | Companywide | **P3** |
| debt_to_total_capital | Capital structure | Companywide | **P3** |
| investment_income | Portfolio return contribution | Companywide | **P1** |
| fte_return_total_portfolio | Investment performance | Companywide | **P2** |
| investment_book_yield | Duration/rate exposure | Companywide | **P2** |
| net_unrealized_gains_fixed | Interest rate regime indicator | Companywide | **P2** |
| fixed_income_duration | Rate sensitivity | Companywide | **P2** |
| weighted_avg_credit_quality | Credit risk | Companywide | **P3** |
| eps_basic | EPS baseline (trailing 12m sum for P/E) | Companywide | **P1** |
| total_net_realized_gains | One-time investment income noise | Companywide | **P2** |
| income_before_income_taxes | Pre-tax earnings quality | Companywide | **P2** |
| pif_special_lines | Specialty segment size | Personal Lines | **P3** |

### Recommended Expanded DB Schema (pgr_edgar_monthly — Phase 1 + Phase 2 additions)

```sql
ALTER TABLE pgr_edgar_monthly ADD COLUMN npw_agency REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN npw_direct REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN npw_commercial REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN npw_property REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN npe_agency REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN npe_direct REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN npe_commercial REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN npe_property REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN pif_agency_auto REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN pif_direct_auto REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN pif_commercial_lines REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN pif_total_personal_lines REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN investment_income REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN total_revenues REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN eps_basic REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN book_value_per_share REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN roe_net_income_ttm REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN shareholders_equity REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN unearned_premiums REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN shares_repurchased REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN avg_cost_per_share REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN fte_return_total_portfolio REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN investment_book_yield REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN net_unrealized_gains_fixed REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN fixed_income_duration REAL;
-- Derived fields (computed at load time)
ALTER TABLE pgr_edgar_monthly ADD COLUMN channel_mix_agency_pct REAL; -- npw_agency / (npw_agency + npw_direct)
ALTER TABLE pgr_edgar_monthly ADD COLUMN npw_growth_yoy REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN underwriting_income REAL; -- npe * (1 - combined_ratio/100)
ALTER TABLE pgr_edgar_monthly ADD COLUMN unearned_premium_growth_yoy REAL;
ALTER TABLE pgr_edgar_monthly ADD COLUMN buyback_yield REAL;
```

## Open Issues

### Pre-existing Test Failures (test_multi_ticker_loader.py — 3 failures)
The multi-ticker loader was refactored but the test mocks weren't updated. The failures are likely one of:
- Mock returns wrong shape (expects list of dicts, gets DataFrame or vice versa)
- Mock doesn't handle the batch-request API format AV now uses
- Ticker list in test doesn't match the updated config.py ETF universe

**Fix approach:** Run `python -m pytest tests/test_multi_ticker_loader.py -v` to see exact error messages. Update the mock return values to match the current `fetch_multi_ticker_prices()` return type. Estimated effort: 30–60 minutes.

### Schema Migration Strategy
Currently there is no migration tooling — schema.sql is the target state but existing DBs need ALTER TABLE statements. As the schema expands (P1.2, P2.1), consider adding a simple `migrations/` folder with numbered SQL files and a migration runner in `db_client.py` that tracks applied migrations in a `schema_version` table.

### EDGAR 8-K Parser Fragility
The HTML parser relies on label-matching (`"Companywide"`, `"Total"`, etc.). PGR has changed table formatting ~3 times since 2004. The parser handles known variants but new format changes will silently produce NULLs. Mitigation: add a completeness check after each monthly fetch — if `combined_ratio` or `pif_total` is NULL for a new filing, raise an alert in the GitHub Actions log.

### Publication Lag Handling
ROADMAP v4.1 documents a 2-month EDGAR filing lag for quarterly data. The 8-K monthly data has a ~2-week lag (filed ~mid-month for the prior month). Ensure feature_engineering.py correctly aligns these lags so the model never uses future data. Current status: documented in ROADMAP but worth an explicit unit test.

### AV Rate Limits (Alpha Vantage)
The root cause of the March 23 GitHub Actions failure was AV rate limiting at 14/22 tickers. The fix (permissions + schedule spread) is in place, but there's no retry-with-backoff logic within the fetcher itself. If AV throttles mid-run, the job fails. Add exponential backoff (3 retries, 60s delay) to the AV fetch loop.
