"""
SEC EDGAR XBRL client for PGR quarterly fundamental data.

Replaces FMP as the authoritative source of quarterly financial facts.
Uses the companyfacts API endpoint which returns all XBRL facts in a
single JSON document (~5–10 MB), so the entire fetch is one HTTP request.

Rate limits (SEC EDGAR ToS):
  - Maximum 10 requests per second.
  - Requires ``User-Agent`` header with name and contact email.
  - No daily quota; free and public.

Cache policy:
  - companyfacts JSON: 168 hours (7 days), same TTL as FMP fundamentals.
  - Cached to data/raw/edgar_pgr_companyfacts.json.

XBRL concept → DB column mapping (us-gaap taxonomy):
  Revenues                               → revenue
      (fallback: PremiumsEarnedNet)
  NetIncomeLoss                          → net_income
  EarningsPerShareBasic                  → eps
  StockholdersEquity (instant)           → used to compute roe; not stored
      (fallback: StockholdersEquityAttributableToParent)
  ROE = annualised(net_income) / equity  → roe  (derived, not a filed XBRL fact)

  pe_ratio, pb_ratio: NOT available from XBRL alone (require market price
  data combined with fundamental data). Stored as NULL in the XBRL pipeline.
  Computed downstream in feature_engineering.build_feature_matrix_from_db():
    pe_ratio = monthly_price / TTM_EPS  (rolling 4-quarter EPS sum + AV price)
    pb_ratio = monthly_price / BVPS     (monthly 8-K BVPS + AV price; v6.x)

Monthly operating metrics (combined ratio, PIF, gainshare):
  PGR files these in monthly 8-K PDF/HTML supplements. They are NOT
  present in XBRL filings. The pgr_edgar_monthly table is populated
  separately via pgr_monthly_loader.py from the user-provided CSV cache.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests

import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PGR_CIK: str = "CIK0000080661"
COMPANYFACTS_URL: str = (
    f"https://data.sec.gov/api/xbrl/companyfacts/{PGR_CIK}.json"
)

_CACHE_HOURS: int = 168  # 7 days

# Minimum and maximum days in a single calendar quarter.
_QUARTER_DAYS_MIN: int = 60
_QUARTER_DAYS_MAX: int = 120

# Minimum and maximum days in a fiscal year (for 10-K full-year facts).
_ANNUAL_DAYS_MIN: int = 320
_ANNUAL_DAYS_MAX: int = 380


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path() -> str:
    return os.path.join(config.DATA_RAW_DIR, "edgar_pgr_companyfacts.json")


def _is_cache_valid(path: str, max_age_hours: int) -> bool:
    if not os.path.exists(path):
        return False
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds < max_age_hours * 3600


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_companyfacts(force_refresh: bool = False) -> dict:
    """Fetch and cache the full PGR XBRL companyfacts JSON.

    Args:
        force_refresh: If True, skip the cache and re-fetch from SEC EDGAR.

    Returns:
        Parsed JSON dict with structure::

            {
              "cik": 80661,
              "entityName": "PROGRESSIVE CORP",
              "facts": {
                "us-gaap": { "<ConceptName>": { "units": { "<unit>": [...] } } },
                "dei": { ... }
              }
            }

    Raises:
        requests.HTTPError: On non-2xx HTTP responses from EDGAR.
    """
    path = _cache_path()
    if not force_refresh and _is_cache_valid(path, _CACHE_HOURS):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    resp = requests.get(
        COMPANYFACTS_URL,
        headers=config.build_edgar_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)

    return data


# ---------------------------------------------------------------------------
# Concept extraction helpers
# ---------------------------------------------------------------------------

def _pick_unit_records(concept_units: dict, preferred: str) -> list:
    """Return fact records for the preferred unit key, or the first available."""
    if preferred in concept_units:
        return concept_units[preferred]
    return next(iter(concept_units.values())) if concept_units else []


def _extract_flow_concept(
    facts: dict,
    taxonomy: str,
    concept: str,
    unit: str = "USD",
) -> pd.Series:
    """Extract single-quarter values for a flow (income-statement) XBRL concept.

    Flow concepts span a date range (``start`` → ``end``).  This function
    keeps only facts whose reporting period is approximately one calendar
    quarter (60–120 days) from a 10-Q filing, or one full year (320–380
    days) from a 10-K filing.  Per-period duplicates (e.g., amended
    filings) are resolved by keeping the most recently filed value.

    Args:
        facts:    Parsed companyfacts dict from :func:`fetch_companyfacts`.
        taxonomy: XBRL taxonomy namespace, e.g. ``"us-gaap"``.
        concept:  Concept name, e.g. ``"Revenues"``.
        unit:     Expected unit key (``"USD"`` for monetary, ``"USD/shares"``
                  for per-share values).

    Returns:
        Series indexed by ``period_end`` (str ``"YYYY-MM-DD"``), values
        are float.  Name is set to ``concept``.  Empty Series if concept
        not found or no qualifying facts exist.
    """
    try:
        concept_units = facts["facts"][taxonomy][concept]["units"]
    except KeyError:
        return pd.Series(name=concept, dtype=float)

    records = _pick_unit_records(concept_units, unit)
    rows: list[dict] = []

    for rec in records:
        form = rec.get("form", "")
        if form not in ("10-Q", "10-K"):
            continue

        start = rec.get("start")
        end = rec.get("end")
        if not start or not end:
            continue

        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            continue

        days = (end_dt - start_dt).days

        is_single_quarter = _QUARTER_DAYS_MIN <= days <= _QUARTER_DAYS_MAX
        is_full_year = form == "10-K" and _ANNUAL_DAYS_MIN <= days <= _ANNUAL_DAYS_MAX

        if not (is_single_quarter or is_full_year):
            continue

        rows.append(
            {
                "period_end": end,
                "val": rec.get("val"),
                "filed": rec.get("filed", ""),
                "form": form,
                "days": days,
            }
        )

    if not rows:
        return pd.Series(name=concept, dtype=float)

    df = pd.DataFrame(rows)
    # Keep single-quarter (10-Q) over full-year (10-K) for the same period_end
    # when both exist; otherwise keep the most recently filed amendment.
    df["is_quarter"] = df["days"].between(_QUARTER_DAYS_MIN, _QUARTER_DAYS_MAX).astype(int)
    df = df.sort_values(["filed", "is_quarter"]).groupby("period_end").last()

    series = df["val"].astype(float)
    series.name = concept
    return series


def _extract_instant_concept(
    facts: dict,
    taxonomy: str,
    concept: str,
    unit: str = "USD",
) -> pd.Series:
    """Extract point-in-time values for an instant (balance-sheet) XBRL concept.

    Instant concepts have only an ``end`` date (the balance-sheet date).
    They have no ``start`` field.  This function keeps the most recently
    filed value per period end date from 10-Q and 10-K filings.

    Args:
        facts:    Parsed companyfacts dict from :func:`fetch_companyfacts`.
        taxonomy: XBRL taxonomy namespace, e.g. ``"us-gaap"``.
        concept:  Concept name, e.g. ``"StockholdersEquity"``.
        unit:     Expected unit key (typically ``"USD"``).

    Returns:
        Series indexed by ``period_end`` (str ``"YYYY-MM-DD"``), values
        are float.  Name is set to ``concept``.
    """
    try:
        concept_units = facts["facts"][taxonomy][concept]["units"]
    except KeyError:
        return pd.Series(name=concept, dtype=float)

    records = _pick_unit_records(concept_units, unit)
    rows: list[dict] = []

    for rec in records:
        form = rec.get("form", "")
        if form not in ("10-Q", "10-K"):
            continue

        end = rec.get("end")
        if not end:
            continue

        rows.append(
            {
                "period_end": end,
                "val": rec.get("val"),
                "filed": rec.get("filed", ""),
            }
        )

    if not rows:
        return pd.Series(name=concept, dtype=float)

    df = pd.DataFrame(rows)
    df = df.sort_values("filed").groupby("period_end").last()

    series = df["val"].astype(float)
    series.name = concept
    return series


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_pgr_fundamentals_quarterly(
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch PGR quarterly fundamentals from SEC EDGAR XBRL.

    Makes at most one HTTP request (to the companyfacts endpoint); subsequent
    calls within the 7-day cache window are served from disk.

    XBRL concepts fetched (us-gaap taxonomy):
      - ``Revenues`` → ``revenue``  (fallback: ``PremiumsEarnedNet``)
      - ``NetIncomeLoss`` → ``net_income``
      - ``EarningsPerShareBasic`` → ``eps``
      - ``StockholdersEquity`` (instant) → used to derive ``roe``
        (fallback: ``StockholdersEquityAttributableToParent``)

    Derived fields:
      - ``roe`` = annualised quarterly net income / end-of-period equity
        = (net_income × 4) / equity

    Not available from XBRL alone:
      - ``pe_ratio`` — computed in ``feature_engineering.build_feature_matrix_from_db()``
        from EDGAR quarterly EPS + AV monthly prices; stored NULL in this table.
      - ``pb_ratio`` — computed from monthly 8-K BVPS (``pgr_edgar_monthly.book_value_per_share``)
        + AV monthly prices; stored NULL in this table.

    Args:
        force_refresh: If True, bypass the disk cache and re-fetch from EDGAR.

    Returns:
        List of dicts, one per quarter, with keys matching
        ``pgr_fundamentals_quarterly`` columns:
        ``period_end``, ``pe_ratio``, ``pb_ratio``, ``roe``, ``eps``,
        ``revenue``, ``net_income``, ``source``.
        Rows with no data across all financial columns are dropped.

    Raises:
        requests.HTTPError: If the EDGAR fetch fails.
    """
    facts = fetch_companyfacts(force_refresh=force_refresh)

    # --- Revenue ---
    revenue = _extract_flow_concept(facts, "us-gaap", "Revenues")
    if revenue.empty:
        revenue = _extract_flow_concept(facts, "us-gaap", "PremiumsEarnedNet")
    revenue = revenue.rename("revenue")

    # --- Net income ---
    net_income = _extract_flow_concept(facts, "us-gaap", "NetIncomeLoss")
    net_income = net_income.rename("net_income")

    # --- EPS (per-share unit) ---
    eps = _extract_flow_concept(
        facts, "us-gaap", "EarningsPerShareBasic", unit="USD/shares"
    )
    eps = eps.rename("eps")

    # --- Stockholders equity (instant; for ROE computation) ---
    equity = _extract_instant_concept(facts, "us-gaap", "StockholdersEquity")
    if equity.empty:
        equity = _extract_instant_concept(
            facts, "us-gaap", "StockholdersEquityAttributableToParent"
        )
    equity = equity.rename("equity")

    # --- Combine all series on period_end index ---
    frames: list[pd.Series] = []
    for s in (revenue, net_income, eps, equity):
        if not s.empty:
            frames.append(s)

    if not frames:
        return []

    combined: pd.DataFrame = pd.concat(frames, axis=1)
    combined.index.name = "period_end"

    # --- Derive annualised ROE ---
    if "net_income" in combined.columns and "equity" in combined.columns:
        valid_mask = combined["equity"].notna() & (combined["equity"] != 0.0)
        combined["roe"] = float("nan")
        combined.loc[valid_mask, "roe"] = (
            combined.loc[valid_mask, "net_income"] * 4.0
        ) / combined.loc[valid_mask, "equity"]
    else:
        combined["roe"] = float("nan")

    # Drop the equity helper (not a DB column).
    combined = combined.drop(columns=["equity"], errors="ignore")

    # pe_ratio and pb_ratio require market price data unavailable from XBRL.
    combined["pe_ratio"] = float("nan")
    combined["pb_ratio"] = float("nan")
    combined["source"] = "edgar"

    # Drop rows with no meaningful financial data (all NaN except source).
    data_cols = ["revenue", "net_income", "eps"]
    present = [c for c in data_cols if c in combined.columns]
    if present:
        combined = combined.dropna(subset=present, how="all")

    if combined.empty:
        return []

    combined = combined.reset_index()
    combined["period_end"] = combined["period_end"].astype(str)

    return combined.to_dict("records")
