"""Row-level validation of the PGR monthly EDGAR table (review 2026-09-25, WP5).

Each check returns a DataFrame of violating rows (empty when the table is
clean).  They are run by ``tests/test_pgr_edgar_integrity.py`` against the
committed DB and by ``scripts/repair_edgar_history.py`` after a rebuild.

Checks
------
* ``income_identity_violations``    total revenues − total expenses = pretax income
* ``combined_ratio_violations``      combined ratio = loss/LAE ratio + expense ratio
* ``equity_violations``              equity ≈ book value per share × common shares
  (net of the Series B preferred stock outstanding 2018-03 … 2024-01)
* ``quarterly_net_income_violations`` monthly net income summed per calendar
  quarter = XBRL quarterly net income (``pgr_fundamentals_quarterly``)
* ``missing_months``                 every calendar month since 2004-08 present
* ``pif_jump_violations``            month-over-month ``pif_total`` change ≤ 5 %
"""

from __future__ import annotations

import pandas as pd

FIRST_MONTH = "2004-08"

# Serial Preferred Shares, Series B: issued March 2018 ($500M liquidation
# preference, $493.9M carrying value), outstanding through the January 2024
# release.  Included in total shareholders' equity but not in book value per
# common share.
PREFERRED_STOCK_CARRYING_VALUE: float = 493.9
PREFERRED_STOCK_FIRST_MONTH = "2018-03"
PREFERRED_STOCK_LAST_MONTH = "2024-01"

INCOME_IDENTITY_TOLERANCE: float = 0.2      # $M; three independently rounded lines
COMBINED_RATIO_TOLERANCE: float = 0.15      # points; LR and ER are each rounded
EQUITY_TOLERANCE: float = 0.06              # relative
QUARTERLY_NI_TOLERANCE: float = 1.0         # $M
PIF_JUMP_TOLERANCE: float = 0.05            # relative month-over-month

# Reviewed month-over-month PIF moves above the tolerance: {month: reason}.
PIF_JUMP_ANNOTATIONS: dict[str, str] = {}


def _periods(index: pd.Index) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.to_datetime(index), freq="M")


def income_identity_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where total revenues − total expenses ≠ income before income taxes."""
    diff = df["total_revenues"] - df["total_expenses"] - df["income_before_income_taxes"]
    out = df.loc[diff.abs() > INCOME_IDENTITY_TOLERANCE,
                 ["total_revenues", "total_expenses", "income_before_income_taxes"]]
    return out.assign(difference=diff[out.index])


def combined_ratio_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the combined ratio ≠ loss/LAE ratio + expense ratio."""
    diff = df["combined_ratio"] - (df["loss_lae_ratio"] + df["expense_ratio"])
    out = df.loc[diff.abs() > COMBINED_RATIO_TOLERANCE,
                 ["combined_ratio", "loss_lae_ratio", "expense_ratio"]]
    return out.assign(difference=diff[out.index])


def preferred_stock(index: pd.Index) -> pd.Series:
    """Preferred stock carrying value ($M) included in equity for each month."""
    periods = _periods(index)
    outstanding = (periods >= pd.Period(PREFERRED_STOCK_FIRST_MONTH, "M")) & (
        periods <= pd.Period(PREFERRED_STOCK_LAST_MONTH, "M")
    )
    return pd.Series(
        [PREFERRED_STOCK_CARRYING_VALUE if flag else 0.0 for flag in outstanding],
        index=index,
    )


def equity_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where common equity differs from BVPS × shares by more than 6 %.

    Common equity is ``shareholders_equity`` less the preferred stock
    outstanding that month.  A row missing any of the three inputs is a
    violation only if it has ``shareholders_equity`` (a stored equity that
    cannot be checked).
    """
    implied = df["book_value_per_share"] * df["common_shares_outstanding"]
    common = df["shareholders_equity"] - preferred_stock(df.index)
    ratio = common / implied - 1.0
    bad = ratio.abs() > EQUITY_TOLERANCE
    unchecked = df["shareholders_equity"].notna() & implied.isna()
    out = df.loc[bad | unchecked,
                 ["shareholders_equity", "book_value_per_share", "common_shares_outstanding"]]
    return out.assign(relative_difference=ratio[out.index])


def quarterly_net_income_violations(
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> pd.DataFrame:
    """Quarters whose summed monthly net income differs from XBRL by > $1M.

    Args:
        monthly:   ``pgr_edgar_monthly`` indexed by month-end, with ``net_income`` ($M).
        quarterly: ``pgr_fundamentals_quarterly`` indexed by quarter-end, with
                   ``net_income`` in USD (discrete quarters).

    Only quarters with all three months and an XBRL value are compared.
    """
    ni = monthly["net_income"].copy()
    ni.index = _periods(ni.index)
    by_quarter = ni.groupby(ni.index.asfreq("Q")).agg(["sum", "count"])
    xbrl = quarterly["net_income"].dropna() / 1e6
    xbrl.index = pd.PeriodIndex(pd.to_datetime(xbrl.index), freq="Q")
    joined = by_quarter.join(xbrl.rename("xbrl_net_income"), how="inner")
    joined = joined[joined["count"] == 3]
    joined["difference"] = joined["sum"] - joined["xbrl_net_income"]
    return joined[joined["difference"].abs() > QUARTERLY_NI_TOLERANCE]


def quarters_compared(monthly: pd.DataFrame, quarterly: pd.DataFrame) -> int:
    """Number of quarters ``quarterly_net_income_violations`` can compare."""
    ni = monthly["net_income"].copy()
    ni.index = _periods(ni.index)
    counts = ni.groupby(ni.index.asfreq("Q")).count()
    xbrl = quarterly["net_income"].dropna()
    xbrl_q = set(pd.PeriodIndex(pd.to_datetime(xbrl.index), freq="Q"))
    return int(sum(1 for q, n in counts.items() if n == 3 and q in xbrl_q))


def missing_months(df: pd.DataFrame, through: str | None = None) -> list[str]:
    """Calendar months from 2004-08 to the last row (or ``through``) with no row."""
    have = set(_periods(df.index))
    last = pd.Period(through, "M") if through else max(have)
    return [
        str(p) for p in pd.period_range(FIRST_MONTH, last, freq="M") if p not in have
    ]


def pif_jump_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Months whose ``pif_total`` moved > 5 % from the previous calendar month.

    Reviewed moves listed in ``PIF_JUMP_ANNOTATIONS`` are exempt.
    """
    pif = df["pif_total"].copy()
    pif.index = _periods(pif.index)
    pif = pif.reindex(pd.period_range(pif.index.min(), pif.index.max(), freq="M"))
    change = pif / pif.shift(1) - 1.0
    bad = change[change.abs() > PIF_JUMP_TOLERANCE]
    bad = bad[[str(p) not in PIF_JUMP_ANNOTATIONS for p in bad.index]]
    return pd.DataFrame({"pif_total": pif[bad.index], "mom_change": bad})
