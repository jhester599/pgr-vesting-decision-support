"""
Load PGR monthly earnings data from the user-provided EDGAR cache CSV.

This module reads the CSV file that the user pre-gathered from SEC EDGAR
monthly earnings releases. Because the file schema is determined upon
receipt, this loader performs flexible schema detection and normalizes
the data to a canonical ``pgr_monthly_fundamentals`` Parquet format.

Expected canonical output columns:
  - combined_ratio (float64): GAAP Combined Ratio (loss ratio + expense ratio)
  - pif_total (float64): Total Policies in Force (all segments, if available)
  - pif_growth_yoy (float64): YoY % change in PIF (computed here if not present)
  - gainshare_estimate (float64): Derived Gainshare estimate (0.0–2.0)

The Gainshare estimate is an **optional** output. If combined_ratio or PIF
data are absent, those columns will be NaN and the WFO engine will drop
them per the threshold in config.WFO_MIN_GAINSHARE_OBS.
"""

import os

import numpy as np
import pandas as pd

import config


_EDGAR_CACHE_PATH = os.path.join(config.DATA_PROCESSED_DIR, "pgr_edgar_cache.csv")
_PROCESSED_PATH = os.path.join(
    config.DATA_PROCESSED_DIR, "pgr_monthly_fundamentals.parquet"
)

# Candidate column name aliases for flexible schema detection.
# Maps canonical name -> list of possible source column names (case-insensitive).
# The user's pre-extracted pgr_monthly_dataset.csv uses 'report_period' for the date
# and 'combined_ratio' / 'pif_total' directly — these are listed first.
_ALIASES: dict[str, list[str]] = {
    "combined_ratio": ["combined_ratio", "combinedratio", "combined ratio", "cr"],
    "loss_ratio": ["loss_ratio", "lossratio", "loss ratio", "lr", "loss_lae_ratio"],
    "expense_ratio": ["expense_ratio", "expenseratio", "expense ratio", "er"],
    "pif_total": ["pif_total", "piftotal", "policies_in_force", "pif", "total_pif"],
    "date": ["report_period", "date", "period", "report_date", "month", "year_month"],
}


def _find_col(df: pd.DataFrame, canonical: str) -> str | None:
    """Return the first matching column name for a canonical field, or None."""
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for alias in _ALIASES.get(canonical, [canonical]):
        if alias.lower() in lower_cols:
            return lower_cols[alias.lower()]
    return None


def load(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return normalized PGR monthly fundamentals from the EDGAR cache CSV.

    If the EDGAR cache CSV does not exist, returns an empty DataFrame with
    the canonical columns so downstream modules degrade gracefully.

    Args:
        force_refresh: If True, re-parse the CSV even if the Parquet exists.

    Returns:
        DataFrame indexed by month-end date (DatetimeIndex, ascending) with
        columns: combined_ratio, pif_total, pif_growth_yoy, gainshare_estimate.
    """
    if not force_refresh and os.path.exists(_PROCESSED_PATH):
        return pd.read_parquet(_PROCESSED_PATH)

    if not os.path.exists(_EDGAR_CACHE_PATH):
        # Graceful degradation: return empty canonical DataFrame.
        empty = pd.DataFrame(
            columns=["combined_ratio", "pif_total", "pif_growth_yoy", "gainshare_estimate"]
        )
        empty.index = pd.DatetimeIndex([], name="date")
        return empty

    raw = pd.read_csv(_EDGAR_CACHE_PATH)
    raw.columns = raw.columns.str.strip()

    # --- Detect and parse date column ---
    date_col = _find_col(raw, "date")
    if date_col is None:
        raise ValueError(
            "pgr_edgar_cache.csv: cannot find a date column. "
            f"Detected columns: {list(raw.columns)}"
        )
    raw["date"] = pd.to_datetime(raw[date_col])
    raw = raw.sort_values("date").set_index("date")

    df = pd.DataFrame(index=raw.index)

    # --- Combined Ratio ---
    cr_col = _find_col(raw, "combined_ratio")
    if cr_col:
        df["combined_ratio"] = pd.to_numeric(raw[cr_col], errors="coerce")
    else:
        # Try to derive from loss_ratio + expense_ratio
        lr_col = _find_col(raw, "loss_ratio")
        er_col = _find_col(raw, "expense_ratio")
        if lr_col and er_col:
            df["combined_ratio"] = (
                pd.to_numeric(raw[lr_col], errors="coerce")
                + pd.to_numeric(raw[er_col], errors="coerce")
            )
        else:
            df["combined_ratio"] = np.nan

    # --- Policies in Force ---
    pif_col = _find_col(raw, "pif_total")
    df["pif_total"] = (
        pd.to_numeric(raw[pif_col], errors="coerce") if pif_col else np.nan
    )

    # --- YoY PIF growth (compute if not already present) ---
    if not df["pif_total"].isna().all():
        df["pif_growth_yoy"] = df["pif_total"].pct_change(periods=12)
    else:
        df["pif_growth_yoy"] = np.nan

    # --- Gainshare estimate (0.0–2.0) ---
    # Based on public proxy filing disclosures; conditional on CR availability.
    cr = df["combined_ratio"]
    pif_growth = df["pif_growth_yoy"]

    if not cr.isna().all():
        cr_score = ((96.0 - cr) / 10.0).clip(lower=0.0, upper=2.0)
    else:
        cr_score = pd.Series(np.nan, index=df.index)

    if not pif_growth.isna().all():
        # 10% YoY PIF growth = max score
        pif_score = (pif_growth / 0.10).clip(lower=0.0, upper=2.0)
    else:
        pif_score = pd.Series(np.nan, index=df.index)

    df["gainshare_estimate"] = (0.5 * cr_score + 0.5 * pif_score)

    df = df.astype("float64")

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    df.to_parquet(_PROCESSED_PATH)
    return df
