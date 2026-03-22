"""
Load historical stock split data for PGR.

Source: Hardcoded from SEC filings / config.PGR_KNOWN_SPLITS.
PGR has had exactly 3 stock splits in its public history, all pre-2010.
No API call is required; the data is static and stable.

Known PGR splits:
  - 3-for-1 on 1992-12-09
  - 3-for-1 on 2002-04-23
  - 4-for-1 on 2006-05-19

Split ratios are used in corporate_actions.py to forward-adjust share
counts for accurate position sizing and DRIP calculations.
"""

import os

import pandas as pd

import config


_PROCESSED_PATH = os.path.join(config.DATA_PROCESSED_DIR, "split_history.parquet")


def load(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return PGR historical stock splits.

    Built from config.PGR_KNOWN_SPLITS (sourced from SEC EDGAR filings).
    No API call is made.

    Args:
        force_refresh: If True, rebuild from config rather than reading Parquet.

    Returns:
        DataFrame with columns:
          - date (DatetimeIndex, ascending): split effective date
          - numerator (float64): new shares per old share (e.g. 4.0 in 4-for-1)
          - denominator (float64): old shares (always 1.0)
          - split_ratio (float64): numerator / denominator
    """
    if not force_refresh and os.path.exists(_PROCESSED_PATH):
        return pd.read_parquet(_PROCESSED_PATH)

    records = [
        {
            "date": pd.Timestamp(s["date"]),
            "numerator": float(s["ratio"]),
            "denominator": 1.0,
            "split_ratio": float(s["ratio"]),
        }
        for s in config.PGR_KNOWN_SPLITS
    ]

    df = (
        pd.DataFrame(records)
        .sort_values("date")
        .reset_index(drop=True)
        .set_index("date")
    )

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    df.to_parquet(_PROCESSED_PATH)
    return df
