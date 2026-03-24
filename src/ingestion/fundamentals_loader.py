"""
Load quarterly fundamental data for PGR from SEC EDGAR XBRL.

Replaces FMP as the data source.  Retrieves income-statement and
balance-sheet metrics via the EDGAR companyfacts API (free, no API key
required).  Data is available for all 10-Q and 10-K filings on record,
which covers PGR back to the mid-1990s.

Point-in-time integrity: EDGAR data is keyed by fiscal period-end date,
which is used as the feature observation date.  No future information is
introduced.  A filing lag of ``config.EDGAR_FILING_LAG_MONTHS`` is applied
by feature_engineering.py when constructing the feature matrix.
"""

import os

import pandas as pd

from src.ingestion import edgar_client
import config


_PROCESSED_PATH = os.path.join(
    config.DATA_PROCESSED_DIR, "fundamentals_quarterly.parquet"
)


def load(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return quarterly fundamentals for PGR (ROE, EPS, revenue, net income).

    Args:
        force_refresh: If True, bypass cache and re-fetch from SEC EDGAR.

    Returns:
        DataFrame indexed by period end date (DatetimeIndex, ascending) with
        columns: pe_ratio, pb_ratio, roe, eps, revenue, net_income.
        All values are float64; missing data appears as NaN.

        Note: pe_ratio and pb_ratio are always NaN — they require market
        price data that is not available from XBRL.  They can be computed
        downstream by joining with daily_prices.
    """
    if not force_refresh and os.path.exists(_PROCESSED_PATH):
        return pd.read_parquet(_PROCESSED_PATH)

    records = edgar_client.fetch_pgr_fundamentals_quarterly(
        force_refresh=force_refresh
    )

    if not records:
        empty = pd.DataFrame(
            columns=["pe_ratio", "pb_ratio", "roe", "eps", "revenue", "net_income"]
        )
        empty.index = pd.DatetimeIndex([], name="date")
        return empty

    df = pd.DataFrame(records)
    df["period_end"] = pd.to_datetime(df["period_end"])
    df = df.sort_values("period_end").set_index("period_end")
    df.index.name = "date"

    # Select only the columns expected by feature_engineering.py.
    cols = ["pe_ratio", "pb_ratio", "roe", "eps", "revenue", "net_income"]
    df = df.reindex(columns=cols).astype("float64")

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    df.to_parquet(_PROCESSED_PATH)
    return df
