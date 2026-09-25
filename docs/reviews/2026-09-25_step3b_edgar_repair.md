# Review 2026-09-25, step 3b: EDGAR parser and table repair

WP5 + WP6 of `docs/reviews/REPO_REVIEW_2026-09-25.md`: F09, F11, F12, F16,
F17, F18, F33 and F34 (F10 and F35 were fixed in step 1).

- **Full cell-level diff:** `2026-09-25_step3b_edgar_cell_diff.csv` in this
  directory, with one row per changed cell (`table, key, column, old, new,
  row_status`) for every changed row of `pgr_edgar_monthly` and
  `pgr_fundamentals_quarterly`. It was written by
  `scripts/repair_edgar_history.py --diff-csv`.
- **Rebuild command:** `scripts/repair_edgar_history.py`. It was run on a copy
  of the committed DB, which was finalized and then copied over
  `data/pgr_financials.db`.
- **Unchanged tables:** every other table (prices, dividends, splits, FRED,
  targets, logs) is byte-for-byte unchanged. This was checked with a per-table
  content hash.

## What was fetched

- **User-Agent:** SEC EDGAR was called with User-Agent
  `Jeff Hester jeffrey.r.hester@gmail.com`.
- **Rate and cache:** requests were throttled to at most 4 per second and
  cached in the gitignored `data/raw/edgar_8k_cache`.
- **Candidate filings:** 309 8-Ks filed since 2004-09-01 with item 7.01 or
  2.02, or 9.01 alone with an EX-99 exhibit.
  - They were found through the primary submissions file and both flat
    pagination files (`-submissions-001`: 2009–2022; `-002`: 1994–2009).
  - For each: the filing index and every exhibit.
  - 270 parse as a monthly release, giving 265 months with one release
    each. The five months with two parses (2006-03, 2008-03, 2011-03,
    2012-03, 2014-11) keep the earliest filing, which is the release; the
    other is a quarterly letter or an unrelated 8-K.
- **XBRL:** one companyfacts request, `CIK0000080661.json`.
- **Transient errors:** EDGAR returned HTTP 503 intermittently. All requests
  succeeded on retry, with exponential back-off.

The two months the review found missing are now present:

| Month | Accession | Items | Why it was missing |
|---|---|---|---|
| 2015-05 | 0000080661-15-000034 | 9.01 only | Item filter; pagination file never read |
| 2019-04 | 0000080661-19-000027 | 5.07, 7.01, 9.01 | Pagination file never read (flat format) |

Both match the values in review Appendix A: NPW 1,581.4 and CR 94.3; NI
487.8 and BVPS 20.74.

## Validation on every row (after the repair)

| Check | Violations |
|---|---|
| Total revenues − total expenses = pretax income (±$0.2M) | 0 of 265 |
| Combined ratio = loss/LAE ratio + expense ratio (±0.15) | 0 of 265 |
| Equity − preferred stock ≈ BVPS × shares (±6 %; $493.9M preferred 2018-03 … 2024-01) | 0 of 265 (max deviation 0.07 %) |
| Monthly NI summed per quarter = XBRL quarterly NI (±$1M) | 0 of 73 quarters (max gap $0.5M, including Q4-2020) |
| Missing months since 2004-08 | 0 |
| Month-over-month `pif_total` change > 5 % | 0 |

`tests/test_pgr_edgar_integrity.py` runs these checks on the committed DB
(read-only):

- On the pre-repair DB, with only the new migrations applied: **11 failed,
  2 passed**.
- On the repaired DB: **13 passed**.

The two that passed before are CR = LR + ER and the percent book yield.
F34's bad 2025-09 row was internally consistent (65.7 + 23.0 = 88.7). The
test that catches it compares 2025-09 against the filed 100.4.

## Parser changes and what they changed in the data

`pgr_edgar_monthly`: 1,259 cells in 223 rows.

| Change | Cells | Rows | Detail |
|---|---|---|---|
| Sign restored (F12) | 160 | 78 | See the list below |
| NULL filled from the filing | 718 | 166 | See the list below |
| Equity/debt/liabilities re-anchored (F18) | 33 | 11 | See the list below |
| Value corrected | 4 | 2 | See the list below |
| Two months added (F17) | 140 | 2 | 2015-05 and 2019-04 |
| Accession normalised to dashed form | 28 | 28 | 2024-04 … 2026-08 except 2024-09, which was already dashed |
| `weighted_avg_credit_quality` | 9 | 9 | The literal string `'nan'` → NULL (2004-08 … 2005-03, 8 rows); 2007-08 `'nan'` → `AA` |
| Derived fields recomputed | 167 | 68 | See the list below |

Sign restored (F12):

- `total_net_realized_gains`: 42 months;
- `fte_return_*`: 18–29 months per column;
- `net_unrealized_gains_fixed`: 14 months (all of 2018, plus 2009-04 and
  2009-06);
- `total_comprehensive_income` / `comprehensive_eps_diluted`: 8 / 7 months;
- `provision_for_income_taxes`: 7 months;
- `roe_net_income_ttm`: 2009-04 … 07;
- net income, EPS and pretax income for 2017-08 (NI −16.8) and 2018-10
  (NI −31.7).

NULL filled from the filing:

- FTE returns and book yield: 116 months, 2006–2023. These lines shared a
  table with the EPS rows.
- 2006-01 … 2007-01 segment NPW/NPE and ratios. The agency channel was
  labelled "Drive – Auto" then.
- 2007-08 balance sheet. Its labels sit behind a spacer cell.
- 2009-04, 2009-05 and 2010-08 PIF. The PIF block is embedded in a larger
  table.
- 2004-10/11 special lines. The label contained a line break.
- 24 early `total_net_realized_gains`. Each reconciles exactly to total
  revenues.
- `roe_net_income_ttm` 2026-02 … 08. These were the F35 NULLs.
- Pre-2006 fixed-income duration, taken from the release commentary.

Equity/debt/liabilities re-anchored (F18): 2004-12, 2005-01/04/05/08 and
2008-09 … 2009-02. Equity was holding ROE (e.g. 30.0) and is now e.g.
5,155.4, matching the FY2004 10-K. Debt was holding debt/capital (e.g. 19.9)
and is now 1,284.3.

Value corrected:

- 2004-08: `pif_special_lines` −2.0 → 2,332; `total_investments` −2.0 →
  14,552.7. Parsed from the plain-text exhibit.
- 2025-09 (F34): `combined_ratio` 88.7 → **100.4**; `expense_ratio` 23.0 →
  **34.7**. Footnote markers in their own cells had shifted the companywide
  column; ratio rows now read only decimal numbers.

Derived fields recomputed:

| Field | Rows changed | Why |
|---|---|---|
| `pif_total` | 32 | One PIF definition |
| `pif_growth_yoy` | 34 | Calendar YoY; the new PIF definition |
| `gainshare_estimate` | 53 | One formula; NULL without both inputs |
| `pif_total_personal_lines` | 24 | Property excluded |
| `channel_mix_agency_pct` | 15 | Newly filled segment NPW |
| `unearned_premium_growth_yoy` | 6 | Calendar YoY |
| `npw_growth_yoy` | 2 | The F16 rows |
| `underwriting_income` | 1 | 2025-09, from the corrected CR |

The two F16 rows now match the review's true values exactly:

- 2016-05: −0.152 → **+0.1055**.
- 2020-04: +0.272 → **+0.0257**.

`pif_growth_yoy` is now defined in every February after a leap year (F11).
For example, 2021-02 is 0.102 and 2025-02 is 0.182.

## Definitions (one each; `src/processing/pgr_edgar_derived.py`)

- **`pif_total`** = agency auto + direct auto + special lines + commercial
  lines.
  - Property is excluded. The printed "companywide total" added property from
    2024-04, and "total personal lines" did so from 2024-12.
  - The printed totals are kept in `pgr_edgar_monthly_raw`.
  - Growth in 2024-12 … 2025-03 is now 0.182–0.184, down from about 0.31.
- **YoY growth** compares the same calendar month one year earlier. It is
  NULL when that month is missing, never a 13-month change. The row-based
  `pct_change(12)` pre-fill in `load_from_csv` was removed.
- **Gainshare** = 0.5 × clip((96 − CR)/10, 0, 2) + 0.5 × clip(PIF
  growth/0.10, 0, 2), and NULL unless both inputs exist.
  - The script fetcher, `pgr_monthly_loader` and
    `src/ingestion/edgar_8k_fetcher` all call the shared module.
  - The script fetcher had fallen back to the CR score alone.

## Provenance (F33)

Migration 005 adds two append-only tables; triggers abort UPDATE and DELETE:

- `pgr_edgar_filing_parses`: one row per (accession, parser version), with
  the exhibit URL, fetched-at and filing date.
- `pgr_edgar_monthly_raw`: one row per value.

It also adds three views:

- `pgr_edgar_monthly_raw_values`: the flat (accession, field, parser_version,
  fetched_at, …) form.
- `pgr_edgar_monthly_raw_current`.
- `pgr_edgar_monthly_first_reported`.

The repair recorded 15,323 values from 265 parses under parser
`8k-html/2026-09-25`. Each value in `pgr_edgar_monthly` equals the
first-reported value of its row's accession; a test checks this.

A filing that lists both items 2.02 and 7.01 is classed as a quarterly
release (item 2.02). This keeps `filing_type = quarterly_earnings` for 2018-03,
2023-09 and 2024-09.

`upsert_pgr_edgar_monthly` no longer mixes filings:

- A re-parse of the same filing updates values but keeps the provenance
  columns.
- An earlier filing replaces the whole row.
- A later filing is logged and not merged.

`load_from_csv` inserts only missing months (`mode="insert_missing"`), so
re-running it on a live DB changes nothing. Derived fields are recomputed
over the whole table after every write.

Legacy values from the old external extractor were not copied into the raw
table. Their origin is unknown, and they remain in git history and in this
diff.

## Quarterly fundamentals (F09, WP6)

`pgr_fundamentals_quarterly`: 213 cells in 74 rows.

- Q4 rows were annual figures and are now discrete quarters (Q4 = FY − 9M).
  This changes EPS in 18 rows, NI in 21 and revenue in 23.
- Values are now the earliest filed.
  - Q1–Q3 2024 NI and revenue move by ≤ $0.6M to the originally filed values.
  - Q1–Q3 2012 revenue moves by ≤ $78.4M.
- The 2007-12-31 row was removed. It was a full year from a 10-K comparative
  and has no nine-month partner.
- ROE = TTM NI / average of the five quarter-end equities.
  - Old values ranged from −0.64 to 1.49 (Q4 = 4 × annual); new values range
    from −0.01 to 0.39.
  - The new values track the trailing ROE printed in the monthly releases
    within 0.27 points on average (FY2018: 24.5 % vs 24.7 %).
  - A two-point average was worse (0.72 points).
- `filing_date` was added (the filing date of the NI fact; the 10-K for Q4).
- The always-NULL `pe_ratio` / `pb_ratio` columns were dropped (migration
  006).

## Effect on model inputs

The monthly table feeds these features (with the filing lag):

- **Live GBT `pif_growth_yoy`:**
  - Latest stored month 2026-08 changes from 0.0687 to 0.0757.
  - 2024-12 … 2025-03 changes from about 0.31 to 0.18.
  - Every month from 2024-04 on changes.
- **Live Ridge `npw_growth_yoy`:** the two F16 rows.
- **`combined_ratio_ttm`:** 2025-09 changes from 88.7 to 100.4, which
  affects the feature rows 2025-11 → 2026-10.
- **`gainshare_est`, `roe_trend`, the realized-gain ratio and the ratio
  features:** these change where their inputs changed.

Live predictions will move. This PR does not re-run `monthly_decision.py`.

## Not done / remaining

- **Revenue components:** in 8 months (2010-07, 2012-03/04, 2013-09,
  2014-09, 2015-07, 2016-06, 2017-11) the revenue components do not add up to
  total revenues. The difference is a gain or loss on extinguishment of debt,
  which has no column. It is not a parse error.
- **2024 fiscal-month change (review F16, SUSPECTED):** not investigated.
- **Filing-date placement (F23 / WP10):** not addressed. Features still use
  the fixed 2-month lag.
