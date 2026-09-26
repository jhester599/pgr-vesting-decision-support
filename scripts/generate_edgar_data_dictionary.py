#!/usr/bin/env python3
"""Regenerate docs/PGR_EDGAR_CACHE_DATA_DICTIONARY.md from the DB.

Coverage and min/median/max are computed from ``pgr_edgar_monthly`` (the
table ``data/processed/pgr_edgar_cache.csv`` is exported from), so the
dictionary cannot drift from the data.  Column descriptions live in
``SECTIONS`` below.  The DB is opened read-only.

Usage::

    python scripts/generate_edgar_data_dictionary.py [--db PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.database import db_client  # noqa: E402

DEFAULT_OUTPUT = os.path.join("docs", "PGR_EDGAR_CACHE_DATA_DICTIONARY.md")

# (section title, intro, [(csv column, format, note)]).  Format is a
# Python format spec for min/median/max, or "text" for string columns.
SECTIONS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    (
        "Income Statement",
        "All figures are the **single month** reported in each release (not "
        "year-to-date), in millions USD. Negative values are losses; releases "
        "print them in parentheses.",
        [
            ("net_premiums_written", ",.1f", "NPW: premiums written in the month"),
            ("net_premiums_earned", ",.1f", "NPE: premiums earned in the month"),
            ("investment_income", ",.1f", "Net investment income"),
            ("total_net_realized_gains", ",.1f", "Total net realized gains (losses) on securities"),
            ("service_revenues", ",.1f", "Fee-based service income"),
            ("fees_and_other_revenues", ",.1f", "Reported as a separate line from ~2013"),
            ("total_revenues", ",.1f", "Total revenues (includes any gain or loss on extinguishment of debt, which has no column)"),
            ("losses_lae", ",.1f", "Losses and loss adjustment expenses"),
            ("policy_acquisition_costs", ",.1f", "Agent commissions and other acquisition costs"),
            ("other_underwriting_expenses", ",.1f", "Other underwriting expenses"),
            ("interest_expense", ",.1f", "Interest on debt"),
            ("total_expenses", ",.1f", "Total expenses (includes policyholder credit expense where reported, e.g. 2025-09)"),
            ("income_before_income_taxes", ",.1f", "Pretax income; equals total revenues − total expenses in every row"),
            ("provision_for_income_taxes", ",.1f", "Income tax expense (benefit)"),
            ("net_income", ",.1f", "GAAP net income; monthly sums reconcile to XBRL quarterly net income"),
            ("total_comprehensive_income", ",.1f", "Net income plus OCI"),
        ],
    ),
    (
        "Per Share Data",
        "EPS in dollars; share counts in **millions**.",
        [
            ("eps_basic", ".2f", "Basic EPS for the month"),
            ("eps_diluted", ".2f", "Diluted EPS for the month"),
            ("comprehensive_eps_diluted", ".2f", "Comprehensive income per diluted share"),
            ("avg_shares_basic", ",.1f", "Weighted-average basic shares (millions)"),
            ("avg_shares_diluted", ",.1f", "Weighted-average diluted shares (millions)"),
            ("avg_diluted_equivalent_shares", ",.1f", "Average diluted equivalent shares from the summary table (millions)"),
        ],
    ),
    (
        "Underwriting Ratios",
        "Companywide GAAP ratios as printed (percent). The combined ratio equals "
        "the loss/LAE ratio plus the expense ratio in every row.",
        [
            ("combined_ratio", ".1f", "Loss/LAE ratio + expense ratio; < 100 = underwriting profit"),
            ("loss_lae_ratio", ".1f", "Losses and LAE / NPE"),
            ("expense_ratio", ".1f", "Underwriting expenses / NPE (includes policyholder credits where reported)"),
        ],
    ),
    (
        "Policies in Force (PIF)",
        "Policies at month-end, in thousands. `pif_total` and "
        "`pif_total_personal_lines` are computed from their components with one "
        "definition for the whole history (property excluded; see below).",
        [
            ("pif_agency_auto", ",.0f", "Agency auto (\"Drive\" brand in 2006)"),
            ("pif_direct_auto", ",.0f", "Direct auto"),
            ("pif_special_lines", ",.0f", "Motorcycles, RVs, watercraft, etc."),
            ("pif_property", ",.0f", "Homeowners / renters; from 2015 (ARX acquisition). Not in pif_total"),
            ("pif_total_personal_lines", ",.0f", "Agency auto + direct auto + special lines"),
            ("pif_commercial_lines", ",.0f", "Commercial auto"),
            ("pif_total", ",.0f", "Personal lines + commercial lines (property excluded)"),
        ],
    ),
    (
        "Net Premiums Written / Earned by Segment",
        "Millions USD.",
        [
            ("npw_agency", ",.1f", "Agency auto NPW"),
            ("npw_direct", ",.1f", "Direct auto NPW"),
            ("npw_property", ",.1f", "Property NPW; from 2015"),
            ("npw_commercial", ",.1f", "Commercial lines NPW"),
            ("npe_agency", ",.1f", "Agency auto NPE"),
            ("npe_direct", ",.1f", "Direct auto NPE"),
            ("npe_property", ",.1f", "Property NPE; from 2015"),
            ("npe_commercial", ",.1f", "Commercial lines NPE"),
        ],
    ),
    (
        "Balance Sheet",
        "Month-end, millions USD (book value per share in dollars; shares in millions).",
        [
            ("total_investments", ",.1f", "Investment portfolio"),
            ("total_assets", ",.1f", "Total GAAP assets"),
            ("loss_lae_reserves", ",.1f", "Loss and LAE reserves"),
            ("unearned_premiums", ",.1f", "Unearned premiums"),
            ("debt", ",.1f", "Debt outstanding (the dollar line, never the debt-to-capital ratio)"),
            ("total_liabilities", ",.1f", "Total liabilities"),
            ("shareholders_equity", ",.1f", "Total shareholders' equity (includes $493.9M preferred stock 2018-03 … 2024-01); BVPS × shares when the release has no equity line"),
            ("book_value_per_share", ".2f", "Book value per common share"),
            ("common_shares_outstanding", ",.1f", "Common shares outstanding at month-end (millions)"),
        ],
    ),
    (
        "Capital Management",
        "",
        [
            ("shares_repurchased", ",.2f", "Shares repurchased in the month (millions)"),
            ("avg_cost_per_share", ",.2f", "Average repurchase price (dollars)"),
            ("debt_to_total_capital", ".1f", "Debt / (debt + equity), percent"),
            ("roe_net_income_trailing_12m", ".1f", "Trailing-12-month ROE on net income, percent (DB column `roe_net_income_ttm`)"),
            ("roe_comprehensive_trailing_12m", ".1f", "Trailing-12-month ROE on comprehensive income, percent"),
        ],
    ),
    (
        "Investment Portfolio",
        "Returns and yields in percent.",
        [
            ("fixed_income_duration", ".1f", "Fixed-income duration (years); pre-2006 from the release commentary"),
            ("investment_book_yield", ".2f", "Pretax annualized investment income book yield, percent"),
            ("fte_return_fixed_income", ".1f", "Fully taxable equivalent total return, fixed income (month)"),
            ("fte_return_common_stocks", ".1f", "Fully taxable equivalent total return, common stocks (month)"),
            ("fte_return_total_portfolio", ".1f", "Fully taxable equivalent total return, total portfolio (month)"),
            ("net_unrealized_gains_fixed", ",.1f", "Net unrealized pretax gains (losses), millions"),
            ("weighted_avg_credit_quality", "text", "Weighted-average credit quality of fixed income"),
        ],
    ),
]

IDENTIFIERS: list[tuple[str, str]] = [
    ("report_period", "Month the release covers (`YYYY-MM`)"),
    ("filing_date", "Date the 8-K was filed"),
    ("filing_type", "`monthly_results` (item 7.01 or 9.01) or `quarterly_earnings` (item 2.02)"),
    ("accession_number", "EDGAR accession number of the release, dashed; every value in the row comes from this filing"),
]

INTRO = """\
# PGR EDGAR Monthly 8-K Data Dictionary

> Generated by `scripts/generate_edgar_data_dictionary.py` from
> `pgr_edgar_monthly` in `data/pgr_financials.db`; do not edit by hand.

**Source file:** `data/processed/pgr_edgar_cache.csv` (exported from the DB by
`scripts/repair_edgar_history.py --export-csv`)
**Coverage:** {first} – {last} ({n} monthly observations, no missing months)
**Filing types:** {types}
**Parser:** `scripts/edgar_8k_fetcher.py`, version `{parser}`
**Units:** millions USD unless noted; shares in millions; PIF in thousands of policies.

Every row is one monthly release (8-K Exhibit 99), re-fetched from EDGAR and
parsed by the repository's parser (review 2026-09-25, step 3b). Each value is
also recorded, append-only, in `pgr_edgar_monthly_raw` with the filing, the
exhibit URL, the fetch time and the parser version; the
`pgr_edgar_monthly_first_reported` view gives the first-reported value of
every field.

Row-level checks (`tests/test_pgr_edgar_integrity.py`) hold on every row:
total revenues − total expenses = pretax income; combined ratio = loss/LAE
ratio + expense ratio; shareholders' equity (net of preferred stock) is
within 6 % of book value per share × shares outstanding; monthly net income
summed per quarter matches XBRL quarterly net income within $1M.

---
"""

DERIVED = """\
## Derived Columns (DB only)

Recomputed over the whole table after every write
(`edgar_8k_fetcher.recompute_derived_fields`; definitions in
`src/processing/pgr_edgar_derived.py`). They are not in the CSV except
`pif_total` and `pif_total_personal_lines`.

| Column | Definition |
|---|---|
| `pif_total` | agency auto + direct auto + special lines + commercial lines; NULL unless all four are present |
| `pif_total_personal_lines` | agency auto + direct auto + special lines |
| `pif_growth_yoy` | `pif_total` vs the same calendar month a year earlier; NULL when that month is missing |
| `npw_growth_yoy` | same, for `net_premiums_written` |
| `unearned_premium_growth_yoy` | same, for `unearned_premiums` |
| `gainshare_estimate` | 0.5 × clip((96 − CR) / 10, 0, 2) + 0.5 × clip(PIF growth / 0.10, 0, 2); NULL unless both inputs exist |
| `channel_mix_agency_pct` | `npw_agency / (npw_agency + npw_direct)` |
| `underwriting_income` | `net_premiums_earned × (1 − combined_ratio / 100)` |

**Why property is excluded from `pif_total`:** PGR's printed "companywide
total" added property PIF from April 2024 (and "total personal lines" from
December 2024), which made reported PIF growth jump about 13 points with no
change in the business (review F11). Property PIF is stored in
`pif_property`. The printed totals are kept in `pgr_edgar_monthly_raw`.

---
"""

FEATURES = """\
## Features Currently Derived in `feature_engineering.py`

| Feature | Source columns | Method |
|---|---|---|
| `combined_ratio_ttm` | `combined_ratio` | `rolling(12).mean()` |
| `pif_growth_yoy` | `pif_growth_yoy` (DB) | Stored calendar-month YoY of `pif_total`, forward-filled to month-ends |
| `gainshare_est` | `gainshare_estimate` (DB) | Stored Gainshare estimate (see Derived Columns) |
| `cr_acceleration` | `combined_ratio_ttm` | `.diff(3)` (3-period second difference) |
| `pe_ratio` | `eps_basic` + price + splits | TTM EPS = 12 consecutive calendar months restated to one share basis (`src/processing/valuation_multiples.py`), placed on the first month-end on or after the filing date, then split-consistent price / TTM EPS |
| `pb_ratio` | `book_value_per_share` + price | price / BVPS (already monthly) |
| `roe` | `pgr_fundamentals_quarterly.roe` | XBRL TTM net income / average equity, placed on the first month-end on or after the filing date and forward-filled |

---

## Candidate Features for Future Development

The following are derivable from existing columns and have theoretical grounding as
insurance sector / PGR-specific predictors. None are currently in the feature matrix.

### Underwriting & Growth

| Candidate Feature | Derivation | Rationale |
|---|---|---|
| `npw_growth_yoy` | calendar-month YoY of `net_premiums_written` (stored in the DB) | Premium volume growth is a leading indicator of PIF expansion and future earned premium. Acceleration above peers signals pricing power. |
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
| `unearned_premium_growth_yoy` | calendar-month YoY of `unearned_premiums` (stored in the DB) | Forward-looking revenue signal — unearned premiums convert to earned revenue over the next 6–12 months. |
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

"""

NOTES = """\
## Data Quality Notes

- **Negatives:** releases print losses in parentheses, sometimes with the
  closing parenthesis in its own cell. These are parsed as negative (review
  F12); earlier extractions stored them as positive.
- **Missing lines:** `fees_and_other_revenues` is reported from ~2013;
  property segment columns from 2015; `comprehensive_eps_diluted`,
  `roe_comprehensive_trailing_12m` and `avg_diluted_equivalent_shares` only in
  some release formats. The FTE return and book-yield lines are read wherever
  the release prints them.
- **Equity without an equity line:** some 2004–2005 releases print only book
  value per share; `shareholders_equity` is then book value per share × shares
  outstanding (marked `method = 'derived'` in `pgr_edgar_monthly_raw`).
- **2006-05 buybacks (split month):** the 8-K prints 2.3M shares, which adds
  pre-split and post-split shares, and an average cost of "NM". Both columns
  come from the Q2 2006 10-Q (0000950152-06-006431, Part II Item 2) on the
  post-split basis: 331,496 × 4 + 1,932,200 = 3.258184M shares at $27.09
  ($88.26M). They are recorded in `pgr_edgar_monthly_raw` under parser
  `10q-issuer-purchases/2026-09-25` by `scripts/repair_split_month_buybacks.py`.
- **Revenue identity:** in a few months (e.g. 2010-07, 2013-09, 2014-09)
  total revenues include a gain or loss on extinguishment of debt, which has
  no column, so the revenue components do not add to `total_revenues`.
- **Timing:** a row enters the features on the first business month-end on or
  after its `filing_date` (`feature_engineering.edgar_availability_dates`);
  `config.EDGAR_FILING_LAG_MONTHS` is only the fallback for a missing date.
- **Share basis:** per-share columns are on the basis of their own month
  (PGR split 4-for-1 on 2006-05-19); restate with `share_basis_factor`
  before comparing across the split.
"""


def _fmt(value: float, spec: str) -> str:
    return format(value, spec)


def _stats_table(df: pd.DataFrame, rows: list[tuple[str, str, str]]) -> str:
    n = len(df)
    lines = [
        "| Column | Non-null | Min | Median | Max | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for col, spec, note in rows:
        series = df[col]
        count = int(series.notna().sum())
        if spec == "text":
            values = series.dropna()
            mode = values.mode().iloc[0] if not values.empty else "—"
            lines.append(
                f"| `{col}` | {count}/{n} | — | {mode} | — | {note} "
                f"({values.nunique()} distinct values) |"
            )
            continue
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            lines.append(f"| `{col}` | 0/{n} | — | — | — | {note} |")
            continue
        lines.append(
            f"| `{col}` | {count}/{n} | {_fmt(numeric.min(), spec)} | "
            f"{_fmt(numeric.median(), spec)} | {_fmt(numeric.max(), spec)} | {note} |"
        )
    return "\n".join(lines)


def render(conn) -> str:
    """Return the dictionary markdown for the DB behind ``conn``."""
    from scripts.edgar_8k_fetcher import PARSER_VERSION

    df = pd.read_sql_query("SELECT * FROM pgr_edgar_monthly ORDER BY month_end", conn)
    df["report_period"] = df["month_end"].str.slice(0, 7)
    df["roe_net_income_trailing_12m"] = df["roe_net_income_ttm"]
    types = df["filing_type"].value_counts()
    parts = [
        INTRO.format(
            first=pd.Timestamp(df["month_end"].iloc[0]).strftime("%B %Y"),
            last=pd.Timestamp(df["month_end"].iloc[-1]).strftime("%B %Y"),
            n=len(df),
            types=" and ".join(f"`{k}` ({v} rows)" for k, v in types.items()),
            parser=PARSER_VERSION,
        )
    ]
    n = len(df)
    ident = ["## Identifiers", "", "| Column | Non-null | Description |", "|---|---|---|"]
    ident += [
        f"| `{col}` | {int(df[col].notna().sum())}/{n} | {desc} |" for col, desc in IDENTIFIERS
    ]
    parts.append("\n".join(ident) + "\n\n---\n")
    for title, intro, rows in SECTIONS:
        block = [f"## {title}", ""]
        if intro:
            block += [intro, ""]
        block.append(_stats_table(df, rows))
        parts.append("\n".join(block) + "\n\n---\n")
    parts.append(DERIVED)
    parts.append(FEATURES)
    parts.append(NOTES)
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=config.DB_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    conn = db_client.get_connection(args.db, read_only=True)
    try:
        text = render(conn)
    finally:
        conn.close()
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
