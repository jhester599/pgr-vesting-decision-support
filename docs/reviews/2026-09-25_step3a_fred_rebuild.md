# Review 2026-09-25, step 3a — FRED pipeline rebuild (WP3)

Findings F06, F07 and the FRED part of F27 in
[`REPO_REVIEW_2026-09-25.md`](REPO_REVIEW_2026-09-25.md).

`scripts/rebuild_fred_macro.py` was run on a copy of the committed DB
(sha256 `ab5651bc…9260` before, `c952af36…9b93` after). The rebuilt copy is
now the committed `data/pgr_financials.db`. 20 FRED series were fetched on
2026-09-25 from the public `fredgraph.csv` endpoint, because the session had
no `FRED_API_KEY`. That endpoint serves the same current-vintage values as the
API. There was one request per series, 0.5 s apart, and responses were cached.

## What the rebuild did

1. Applied migration `005_fred_one_row_per_month`:
   - It deleted the 576 duplicate month rows (64 months × 9 macro series,
     2008-05 → 2026-02).
   - It relabelled every row to its month's last business day and added a
     unique (series, month) index.
2. Replaced every row of the 13 production series (`FRED_SERIES_MACRO` +
   `FRED_SERIES_PGR`, from 1990-01) and the 7 v19 FRED series (from 2008-01)
   with raw, unlagged observations: one row per month, holding the month's
   last observation, with no forward fill.
3. Left `CUSR0000SETE` (BLS) and the three Multpl series alone apart from
   relabelling. Their values were already stored raw.

Only `fred_macro_monthly` and `schema_migrations` changed. Every other table is
identical, and `PRAGMA integrity_check` is `ok`. Rows went from 8,079 to 7,574;
no (series, month) pair repeats, and every label is a business month-end.

## How the old table was lagged

Measured by matching stored values to the refetched raw months,
stored(M) = raw(M − k):

| Series | Stored lag k | Lag in features before | Lag in features after (configured) |
|---|---|---|---|
| VIXCLS, T10Y2Y, GS2/5/10, T10YIE, BAA10Y, BAMLH0A0HYM2, CUSR0000SETA02, CUSR0000SAM2 | 1 (100 % of months) | 2 | 1 |
| NFCI | 2 (older vintage, 29 % exact) | 3–4 | 2 |
| TRFVOLUSM227NFWA | 2 (99 %) | 4 | 2 |
| PCU5241265241261 | 0 (99 %; stored raw by an earlier path) | 1 | 1 |
| v19 series (DTWEXBGS, …) | 0 | 1 | 1 |

These lags were measured on the built feature matrices, using feature rows from
2010 on:

| Feature | Before: best k | After: best k |
|---|---|---|
| `vix` | 2 (100 %) | 1 (100 %) |
| `nfci` | 4 (37 %) | 2 (100 %) |
| `yield_slope` | 2 (100 %) | 1 (100 %) |
| `credit_spread_hy` | 2 (97 %) | 1 (100 %) |

## BAMLH0A0HYM2: truncated FRED history

FRED now serves only the last three years of ICE BofA series, so the fetch
started at 2023-09. Replacing the series outright would have deleted 1996–2023
from `credit_spread_hy`, a live Ridge and GBT feature. Instead, the script:

- measured the stored lag on the 36-month overlap: k = 1, 100 % match;
- kept the 321 earlier stored months (1996-12 → 2023-08), shifted back by one
  month.

The weekly fetch only upserts, so these months stay. Don't delete them: FRED
can't restore them.

## Per-series diff

`months_changed` counts months that exist in both tables with a different
value. Most changes are the removed one-month lag. `months_removed` covers the
following:
- the old lagged 2026-09 GS rows (monthly averages for September are not
  published yet);
- one CUSR0000SAM2 month: BLS never published October 2025 (the shutdown).
  The old loader had forward-filled it; the feature builder now fills the gap
  after lagging.

| series_id | rows_before | duplicate_months_before | rows_after | first_month | last_month_before | last_month_after | months_added | months_removed | months_changed | max_abs_change | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BAA10Y | 504 | 64 | 441 | 1990-01 | 2026-09 | 2026-09 | 1 | 0 | 425 | 1.55 | fredgraph.csv |
| BAMLH0A0HYM2 | 421 | 64 | 358 | 1996-12 | 2026-09 | 2026-09 | 1 | 0 | 356 | 5.21 | fredgraph.csv; kept 321 earlier months 1996-12..2023-08 from the stored table, un-lagged by 1 (overlap 36 months, 100% match) |
| CUSR0000SAM2 | 434 | 0 | 439 | 1990-01 | 2026-03 | 2026-08 | 6 | 1 | 432 | 5.162 | fredgraph.csv |
| CUSR0000SETA02 | 434 | 0 | 440 | 1990-01 | 2026-03 | 2026-08 | 6 | 0 | 425 | 15.92 | fredgraph.csv |
| CUSR0000SETE | 120 | 0 | 120 | 2008-01 | 2017-12 | 2017-12 | 0 | 0 | 0 | 0 | not refetched |
| DCOILWTICO | 219 | 0 | 225 | 2008-01 | 2026-03 | 2026-09 | 6 | 0 | 3 | 1.83 | fredgraph.csv |
| DTWEXBGS | 219 | 0 | 225 | 2008-01 | 2026-03 | 2026-09 | 6 | 0 | 1 | 0.1499 | fredgraph.csv |
| GS10 | 504 | 64 | 440 | 1990-01 | 2026-09 | 2026-08 | 1 | 1 | 428 | 1.11 | fredgraph.csv |
| GS2 | 504 | 64 | 440 | 1990-01 | 2026-09 | 2026-08 | 1 | 1 | 423 | 0.88 | fredgraph.csv |
| GS5 | 504 | 64 | 440 | 1990-01 | 2026-09 | 2026-08 | 1 | 1 | 430 | 0.77 | fredgraph.csv |
| MORTGAGE30US | 220 | 0 | 225 | 2008-01 | 2026-04 | 2026-09 | 5 | 0 | 1 | 0.16 | fredgraph.csv |
| MRTSSM447USN | 217 | 0 | 223 | 2008-01 | 2026-01 | 2026-07 | 6 | 0 | 1 | 117 | fredgraph.csv |
| NFCI | 503 | 64 | 441 | 1990-01 | 2026-09 | 2026-09 | 2 | 0 | 439 | 1.955 | fredgraph.csv |
| PCU5241265241261 | 333 | 0 | 339 | 1998-06 | 2026-02 | 2026-08 | 6 | 0 | 4 | 0.897 | fredgraph.csv |
| PPIACO | 218 | 0 | 224 | 2008-01 | 2026-02 | 2026-08 | 6 | 0 | 4 | 1.687 | fredgraph.csv |
| SP500_EARNINGS_YIELD_MULTPL | 220 | 0 | 220 | 2008-01 | 2026-04 | 2026-04 | 0 | 0 | 0 | 0 | not refetched |
| SP500_PE_RATIO_MULTPL | 220 | 0 | 220 | 2008-01 | 2026-04 | 2026-04 | 0 | 0 | 0 | 0 | not refetched |
| SP500_PRICE_TO_BOOK_MULTPL | 73 | 0 | 73 | 2008-12 | 2026-04 | 2026-04 | 0 | 0 | 0 | 0 | not refetched |
| T10Y2Y | 504 | 64 | 441 | 1990-01 | 2026-09 | 2026-09 | 1 | 0 | 423 | 0.6 | fredgraph.csv |
| T10YIE | 348 | 64 | 285 | 2003-01 | 2026-09 | 2026-09 | 1 | 0 | 278 | 1.03 | fredgraph.csv |
| THREEFYTP10 | 219 | 0 | 225 | 2008-01 | 2026-03 | 2026-09 | 6 | 0 | 1 | 0.0636 | fredgraph.csv |
| TRFVOLUSM227NFWA | 433 | 0 | 439 | 1990-01 | 2026-03 | 2026-07 | 6 | 0 | 433 | 8.271e+04 | fredgraph.csv |
| VIXCLS | 504 | 64 | 441 | 1990-01 | 2026-09 | 2026-09 | 1 | 0 | 440 | 21.27 | fredgraph.csv |
| WPU45110101 | 204 | 0 | 210 | 2009-03 | 2026-02 | 2026-08 | 6 | 0 | 2 | 0.398 | fredgraph.csv |

## Live decision row (2026-08-31)

This compares the master code on the committed DB with this branch on the
rebuilt DB. It is the row the September run used (as-of 2026-09-21). Only
FRED-derived features changed; momentum, volatility and EDGAR features are
identical.

| Feature | Live model | Before | After | Source month before → after |
|---|---|---|---|---|
| `vix` | ridge, gbt | 16.45 | 15.99 | 2026-06 → 2026-07 |
| `nfci` | ridge, gbt | −0.495 | −0.511 | 2026-04 (older vintage) → 2026-06 |
| `yield_slope` | ridge, gbt | 0.30 | 0.47 | 2026-06 → 2026-07 |
| `credit_spread_hy` | ridge, gbt | 2.75 | 2.85 | 2026-06 → 2026-07 |
| `real_rate_10y` | ridge | 2.23 | 2.32 | 2026-06 → 2026-07 |
| `real_yield_change_6m` | ridge | 0.34 | 0.47 | 2026-06 → 2026-07 |
| `yield_curvature` | gbt | −0.16 | −0.16 | 2026-06 → 2026-07 (both −0.16) |
| `rate_adequacy_gap_yoy` | gbt | **NaN** (median-imputed) | 0.0085 | stale (2026-02/03) → 2026-07 |

Research-only FRED features that were NaN in this row are now populated:
`ppi_auto_ins_yoy`, `used_car_cpi_yoy`, `medical_cpi_yoy`,
`severity_index_yoy`, `vmt_yoy`, `usd_*`, `wti_return_3m`,
`mortgage_spread_30y_10y`, `term_premium_10y`,
`legal_services_ppi_relative` and `gasoline_retail_sales_delta`. Across the
whole matrix, 6,081 of 24,396 feature cells changed.
- Most changes are in FRED-derived columns.
- 40 cells in 2000–2007 are in the price-based synthetic columns
  (`pgr_vs_kie_6m`, `pgr_vs_peers_6m`, `pgr_vs_vfh_6m`). There, the old FRED
  index carried only weekend calendar-month-end labels, so the left join
  missed the month's value and an older value was forward-filled.

v134 (the lag sweep) was **not** re-run; that is step 8a. Its committed "lag 0"
was really lag 1.

## Freshness check on 2026-09-25

The new per-series check on the pre-rebuild DB reports **WARNING**:
`PCU5241265241261` is 5 months behind, and `CUSR0000SETA02` and
`CUSR0000SAM2` are 4 months behind. The live feature named in each warning is
`rate_adequacy_gap_yoy`. The old whole-table check reported this DB as OK.
On the rebuilt DB, all 9 tickers and all 11 live FRED series are OK.
