# PGR EDGAR Monthly 8-K Data Dictionary

The v8.9 live-parser expansion now captures a much larger subset of these
fields directly from recent monthly 8-K HTML exhibits, reducing the gap
between the live EDGAR fetch path and this historical CSV baseline.

**Source file:** `data/processed/pgr_edgar_cache.csv`
**Coverage:** August 2004 – January 2026 (256 monthly observations)
**Filing types:** `monthly_results` (170 rows) and `quarterly_earnings` (86 rows)
**Units:** Dollar figures in millions USD unless noted. Share counts in thousands.

Progressive files monthly 8-K supplements ("Monthly Results") and quarterly earnings supplements
with consistent, machine-readable operating data stretching back to 2004 — an unusually rich
primary source for a publicly traded insurer. All fields are sourced directly from SEC EDGAR
XBRL and structured filings; no third-party data vendors.

---

## Identifiers

| Column | Non-null | Description |
|---|---|---|
| `report_period` | 256/256 | Month-end period the filing covers (`YYYY-MM` parsed to date) |
| `filing_date` | 256/256 | Date the 8-K was filed with SEC |
| `filing_type` | 256/256 | `monthly_results` or `quarterly_earnings` |
| `accession_number` | 256/256 | SEC EDGAR accession number — unique filing key |

---

## Income Statement

All figures represent the **single month** reported in each 8-K supplement (not cumulative YTD).
TTM aggregations are derived downstream in `feature_engineering.py` via rolling sums.

| Column | Non-null | Min | Median | Max | Notes |
|---|---|---|---|---|---|
| `net_premiums_written` | 256/256 | 905.7 | 1,705.0 | 9,041.0 | NPW — new business bound in period |
| `net_premiums_earned` | 256/256 | 1,007.4 | 1,670.6 | 7,121.0 | NPE — revenue recognized in period |
| `investment_income` | 256/256 | 30.0 | 50.1 | 333.0 | Net investment income |
| `total_net_realized_gains` | 232/256 | -1,039.8 | 19.6 | 537.3 | Realized investment gains/losses |
| `service_revenues` | 256/256 | 0.9 | 7.2 | 48.0 | Fee-based service income |
| `fees_and_other_revenues` | 155/256 | 20.7 | 45.1 | 104.0 | Reported from ~2013 onward |
| `total_revenues` | 256/256 | 71.3 | 1,757.3 | 7,763.0 | Sum of all revenue lines |
| `losses_lae` | 256/256 | 665.2 | 1,198.7 | 4,853.0 | Losses and loss adjustment expenses |
| `policy_acquisition_costs` | 256/256 | 102.5 | 140.9 | 530.0 | Agent commissions and DAC |
| `other_underwriting_expenses` | 256/256 | 84.4 | 223.2 | 1,031.0 | G&A, technology, other opex |
| `interest_expense` | 256/256 | 6.0 | 11.7 | 24.0 | Debt service |
| `total_expenses` | 256/256 | 888.3 | 1,576.1 | 7,029.0 | Sum of all expense lines |
| `income_before_income_taxes` | 256/256 | -982.0 | 177.4 | 1,549.0 | Pre-tax income |
| `provision_for_income_taxes` | 256/256 | -351.2 | 55.3 | 329.0 | Income tax expense |
| `net_income` | 256/256 | -684.4 | 120.6 | 1,220.0 | GAAP net income |
| `total_comprehensive_income` | 177/256 | -1,485.0 | 158.6 | 1,833.0 | Includes OCI (unrealized gains) |

---

## Per Share Data

Shares in thousands; EPS in dollars.

| Column | Non-null | Min | Median | Max | Notes |
|---|---|---|---|---|---|
| `eps_basic` | 256/256 | -1.17 | 0.19 | 2.08 | Basic EPS for the single month |
| `eps_diluted` | 256/256 | -1.17 | 0.19 | 2.07 | Diluted EPS for the single month |
| `comprehensive_eps_diluted` | 167/256 | -2.54 | 0.28 | 3.12 | Comprehensive EPS including OCI |
| `avg_shares_basic` | 256/256 | 194.2 | 585.5 | 774.8 | Weighted avg basic shares (thousands) |
| `avg_shares_diluted` | 256/256 | 196.7 | 587.7 | 784.6 | Weighted avg diluted shares (thousands) |
| `avg_diluted_equivalent_shares` | 184/256 | 197.5 | 587.7 | 781.0 | Reported directly in some filings |

> **Key insight for feature engineering:** `eps_basic` is a **single-month** figure, not quarterly.
> TTM EPS = `eps_basic.rolling(12).sum()`. This is superior to using EDGAR XBRL quarterly EPS
> (which requires 4-quarter aggregation with filing lags) because the monthly cadence gives 3×
> more data points and tighter lag — the 8-K is typically filed within 6 weeks of month-end.

---

## Underwriting Ratios

Reported directly by Progressive; no need to derive from components.

| Column | Non-null | Min | Median | Max | Notes |
|---|---|---|---|---|---|
| `combined_ratio` | 256/256 | 77.1 | 92.2 | 116.2 | Loss ratio + expense ratio; <100 = underwriting profit |
| `loss_lae_ratio` | 241/256 | 48.0 | 71.7 | 97.2 | Losses+LAE / NPE |
| `expense_ratio` | 241/256 | 15.4 | 20.5 | 38.6 | Underwriting expenses / NPE |

---

## Policies in Force (PIF)

Count of active policies at period-end, in thousands.

| Column | Non-null | Min | Median | Max | Notes |
|---|---|---|---|---|---|
| `pif_agency_auto` | 240/256 | 4,216 | 4,883 | 10,855 | Agency auto channel PIF |
| `pif_direct_auto` | 253/256 | 2,029 | 4,745 | 16,164 | Direct (online/phone) auto PIF |
| `pif_special_lines` | 251/256 | -2 | 4,111 | 7,012 | Motorcycles, RVs, watercraft, etc. |
| `pif_property` | 128/256 | 1,043 | 2,432 | 3,660 | Homeowners / renters; reported from ~2015 |
| `pif_total_personal_lines` | 253/256 | 8,577 | 13,626 | 37,686 | Sum of personal lines segments |
| `pif_commercial_lines` | 253/256 | 413 | 554 | 1,201 | Commercial auto (trucking/fleet) |
| `pif_total` | 252/256 | 9,049 | 14,174 | 38,875 | Total company-wide PIF |

---

## Net Premiums Written / Earned by Segment

In millions USD.

| Column | Non-null | Min | Median | Max | Notes |
|---|---|---|---|---|---|
| `npw_agency` | 241/256 | 483.1 | 843.3 | 2,714.0 | Agency auto NPW |
| `npw_direct` | 241/256 | 263.3 | 754.3 | 3,623.0 | Direct auto NPW |
| `npw_property` | 126/256 | 49.3 | 165.4 | 296.0 | Property NPW; available from ~2015 |
| `npw_commercial` | 239/256 | 86.8 | 204.6 | 2,408.0 | Commercial lines NPW |
| `npe_agency` | 241/256 | 558.1 | 831.3 | 2,559.0 | Agency auto NPE |
| `npe_direct` | 241/256 | 287.0 | 733.1 | 3,425.0 | Direct auto NPE |
| `npe_property` | 128/256 | 63.5 | 150.6 | 274.6 | Property NPE |
| `npe_commercial` | 241/256 | 109.3 | 183.4 | 952.0 | Commercial lines NPE |

---

## Balance Sheet

Snapshot at month-end, in millions USD.

| Column | Non-null | Min | Median | Max | Notes |
|---|---|---|---|---|---|
| `total_investments` | 246/256 | -2.0 | 20,896 | 97,391 | Investment portfolio market value |
| `total_assets` | 253/256 | 17,184 | 29,304 | 124,032 | Total GAAP assets |
| `loss_lae_reserves` | 248/256 | 5,076 | 9,811 | 43,601 | Outstanding claims liability |
| `unearned_premiums` | 253/256 | 4,090 | 6,622 | 26,822 | Liability for premiums not yet earned |
| `debt` | 253/256 | 18.3 | 2,664 | 6,898 | Total outstanding debt |
| `total_liabilities` | 253/256 | 12,365 | 21,511 | 92,716 | Total GAAP liabilities |
| `shareholders_equity` | 253/256 | -2.1 | 7,320 | 37,540 | Book value of equity |
| `book_value_per_share` | 253/256 | 5.86 | 14.10 | 64.04 | Equity / shares outstanding (dollars) |
| `common_shares_outstanding` | 252/256 | 195.6 | 585.8 | 779.2 | Period-end diluted share count (thousands) |

---

## Capital Management

| Column | Non-null | Min | Median | Max | Notes |
|---|---|---|---|---|---|
| `shares_repurchased` | 242/256 | 0.0 | 0.3 | 16.9 | Shares bought back in period (millions of shares, normalized from source filings) |
| `avg_cost_per_share` | 245/256 | 0.00 | 23.39 | 281.35 | Average buyback price (dollars) |
| `debt_to_total_capital` | 253/256 | 14.4% | 24.8% | 35.4% | Debt / (debt + equity) |
| `roe_net_income_trailing_12m` | 253/256 | -2.1% | 18.7% | 38.5% | Pre-computed TTM ROE (net income basis) |
| `roe_comprehensive_trailing_12m` | 168/256 | -16.3% | 19.6% | 50.9% | Pre-computed TTM ROE (comprehensive basis) |

---

## Investment Portfolio

| Column | Non-null | Min | Median | Max | Notes |
|---|---|---|---|---|---|
| `fixed_income_duration` | 248/256 | 1.6 | 2.8 | 3.5 | Interest rate sensitivity (years) |
| `investment_book_yield` | 126/256 | 2.4% | 3.9% | 5.6% | Portfolio book yield; available from ~2015 |
| `fte_return_fixed_income` | 135/256 | -4.2% | 0.5% | 3.0% | Fully taxable equiv return on fixed income |
| `fte_return_common_stocks` | 134/256 | -17.0% | 2.3% | 11.3% | Fully taxable equiv return on equities |
| `fte_return_total_portfolio` | 135/256 | -5.0% | 0.7% | 2.9% | Fully taxable equiv return, total portfolio |
| `net_unrealized_gains_fixed` | 252/256 | -4,226 | 716.7 | 1,997.6 | Unrealized G/L on fixed income (millions) |
| `weighted_avg_credit_quality` | 247/256 | — | AA- | — | Portfolio credit rating (4 distinct values) |

---

## Features Currently Derived in `feature_engineering.py`

| Feature | Source columns | Method |
|---|---|---|
| `combined_ratio_ttm` | `combined_ratio` | `rolling(12).mean()` |
| `pif_growth_yoy` | `pif_total` | `pct_change(12)` |
| `gainshare_est` | `combined_ratio_ttm` | Gainshare lookup table (≤96 CR → multiplier) |
| `cr_acceleration` | `combined_ratio_ttm` | `.diff(3)` (3-period second difference) |
| `pe_ratio` | `eps_basic` + price | `rolling(12).sum()` for TTM EPS, then price / TTM EPS |
| `pb_ratio` | `book_value_per_share` + price | price / BVPS (already monthly) |
| `roe` | `roe_net_income_trailing_12m` | Used directly (pre-computed in filing) |

---

## Candidate Features for Future Development

The following are derivable from existing columns and have theoretical grounding as
insurance sector / PGR-specific predictors. None are currently in the feature matrix.

### Underwriting & Growth

| Candidate Feature | Derivation | Rationale |
|---|---|---|
| `npw_growth_yoy` | `net_premiums_written.pct_change(12)` | Premium volume growth is a leading indicator of PIF expansion and future earned premium. Acceleration above peers signals pricing power. |
| `npw_vs_npe_spread` | `net_premiums_written - net_premiums_earned` | Positive spread = premium is being written faster than earned → growth mode. Negative = runoff or rate adequacy pressure. |
| `expense_ratio_ttm` | `rolling(12).mean()` of `expense_ratio` | Structural cost efficiency trend. Progressive's direct channel scale advantage shows up here. |
| `loss_ratio_ttm` | `rolling(12).mean()` of `loss_lae_ratio` | Separates underwriting deterioration from expense pressure within combined_ratio. |
| `channel_mix_direct_pct` | `pif_direct_auto / pif_total_personal_lines` | Mix shift toward direct channel = higher margin. A rising share is bullish for long-run margins. |
| `commercial_mix_pct` | `pif_commercial_lines / pif_total` | Commercial auto is higher-margin and less cyclical than personal auto. |
| `property_mix_pct` | `pif_property / pif_total` | Property segment has a different loss cycle than auto; mix changes affect CR volatility. |
| `npw_per_pif` | `net_premiums_written / pif_total` | Proxy for average premium per policy — captures rate increases independently of volume. |

### Balance Sheet & Capital

| Candidate Feature | Derivation | Rationale |
|---|---|---|
| `reserve_to_npe_ratio` | `loss_lae_reserves / net_premiums_earned` | Reserve adequacy signal. Rising ratio may precede adverse development; falling may signal reserve releases. |
| `unearned_premium_growth_yoy` | `unearned_premiums.pct_change(12)` | Forward-looking revenue signal — unearned premiums convert to earned revenue over the next 6–12 months. |
| `debt_to_equity` | `debt / shareholders_equity` | Leverage alternative to `debt_to_total_capital`; more sensitive to equity swings from unrealized gains. |
| `investment_leverage` | `total_investments / shareholders_equity` | Insurance-specific leverage metric; higher = more interest rate and credit risk. |
| `buyback_yield_monthly` | `(shares_repurchased * avg_cost_per_share) / (shares_outstanding * price)` | Monthly capital return signal. Progressive uses variable dividends + buybacks; this captures non-dividend return. |
| `equity_per_share_growth_yoy` | `book_value_per_share.pct_change(12)` | Intrinsic value compounding rate — alternative ROE expression on a per-share basis. |

### Investment Portfolio

| Candidate Feature | Derivation | Rationale |
|---|---|---|
| `unrealized_gain_pct_equity` | `net_unrealized_gains_fixed / shareholders_equity` | OCI sensitivity to rate moves. Large negative values signal book value at risk in rising-rate environments. |
| `duration_vs_rate_regime` | `fixed_income_duration` × (change in 10Y yield) | Estimated mark-to-market impact of rate shifts on the investment portfolio. |
| `portfolio_yield_spread` | `investment_book_yield` - 10Y Treasury | Excess yield above risk-free; compresses as competition for IG credit increases. |
| `realized_gain_to_ni_ratio` | `total_net_realized_gains / net_income` | Quality-of-earnings flag. High ratio means reported income is driven by portfolio sales, not underwriting. |

### Earnings Quality & Valuation

| Candidate Feature | Derivation | Rationale |
|---|---|---|
| `underwriting_income` | `net_premiums_earned - losses_lae - policy_acquisition_costs - other_underwriting_expenses` | Core insurance profit before investment income and taxes. More stable signal than net income. |
| `combined_ratio_ex_cats` | Not directly in data — could be approximated via deviation from trend | Catastrophe-adjusted CR isolates secular underwriting trends from weather noise. |
| `price_to_npw` | `market_cap / (net_premiums_written * 12)` | Insurance-specific valuation: premium multiple. Commonly used by sector analysts alongside P/B. |
| `price_to_npe_ttm` | `price / (net_premiums_earned.rolling(12).sum())` | Earnings-power valuation on premium revenue rather than net income. |
| `eps_revision_momentum` | `eps_basic - eps_basic.shift(12)` (level diff) | Earnings revision signal; accelerating EPS growth predicts positive analyst revision cycles. |

### Regime / Macro Interaction

| Candidate Feature | Derivation | Rationale |
|---|---|---|
| `cr_vs_industry_spread` | `combined_ratio` - industry peer average CR | Relative underwriting quality vs. peers. Requires external peer data (Travelers, Allstate, etc.). |
| `investment_income_growth_yoy` | `investment_income.pct_change(12)` | Investment income is interest-rate sensitive; YoY growth captures reinvestment rate tailwind/headwind. |
| `tax_rate_effective` | `provision_for_income_taxes / income_before_income_taxes` | Effective tax rate variation signals use of tax-advantaged investments (munis). |

---

## Data Quality Notes

- **Units shifted over time:** Share counts in early filings (pre-2012) appear to be in thousands; later filings may report differently. Validate `eps_basic * avg_shares_basic ≈ net_income` before using share counts directly.
- **`pif_special_lines` has a -2 outlier** (2004-08) — likely a data artifact from early filing format; treat as missing.
- **`pif_property`, `npw_property`, `npe_property`** are only available from ~2015 when Progressive entered the homeowners market via the ASI acquisition.
- **`fees_and_other_revenues`** begins around 2013 when Progressive started separately reporting this line.
- **Investment return fields** (`fte_return_*`, `investment_book_yield`) are only available from approximately 2015.
- **`total_investments` has a -2.0 outlier** in early data** — treat as missing.
- All EDGAR monthly data should be used with a **2-month filing lag** (`config.EDGAR_FILING_LAG_MONTHS`) to prevent look-ahead bias in the feature matrix.
