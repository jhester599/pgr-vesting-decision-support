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

XBRL concept → DB column mapping (us-gaap taxonomy), one discrete quarter
per row (Q4 = 10-K full year − 10-Q nine months), earliest-filed values:
  Revenues                               → revenue
      (fallback: PremiumsEarnedNet)
  NetIncomeLoss                          → net_income
  EarningsPerShareBasic                  → eps
  StockholdersEquity (instant)           → used to compute roe; not stored
      (fallback: StockholdersEquityAttributableToParent)
  ROE = TTM net income / average of the five quarter-end equities
                                         → roe  (derived, not a filed XBRL fact)
  filing date of the net-income fact     → filing_date

  P/E and P/B are computed downstream from monthly 8-K EPS/BVPS and prices
  (feature_engineering.build_feature_matrix_from_db); the always-NULL
  pe_ratio / pb_ratio columns were dropped by migration 007.

Monthly operating metrics (combined ratio, PIF, gainshare):
  PGR files these in monthly 8-K HTML supplements. They are NOT present in
  XBRL filings. The pgr_edgar_monthly table is populated separately by
  scripts/edgar_8k_fetcher.py.
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
from src.ingestion.http_utils import build_retry_session


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

# Nine-month year-to-date window (10-Q Q3), used to derive Q4 = FY - 9M.
_NINE_MONTH_DAYS_MIN: int = 250
_NINE_MONTH_DAYS_MAX: int = 290


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

    session = build_retry_session()
    resp = session.get(
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


def _duration_facts(
    facts: dict,
    taxonomy: str,
    concept: str,
    unit: str,
) -> pd.DataFrame:
    """Return the earliest-filed 10-Q/10-K fact for each (start, end) window.

    Columns: ``start``, ``end``, ``days``, ``val``, ``filed``, ``form``.
    Later filings repeat a period as a comparative (sometimes restated); the
    first filing is what was known at the time (F09).
    """
    try:
        concept_units = facts["facts"][taxonomy][concept]["units"]
    except KeyError:
        return pd.DataFrame(columns=["start", "end", "days", "val", "filed", "form"])

    rows: list[dict[str, Any]] = []
    for rec in _pick_unit_records(concept_units, unit):
        form = rec.get("form", "")
        if form not in ("10-Q", "10-K"):
            continue
        start, end = rec.get("start"), rec.get("end")
        if not start or not end or rec.get("val") is None:
            continue
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            continue
        rows.append({
            "start": start,
            "end": end,
            "days": (end_dt - start_dt).days,
            "val": float(rec["val"]),
            "filed": rec.get("filed", ""),
            "form": form,
        })
    if not rows:
        return pd.DataFrame(columns=["start", "end", "days", "val", "filed", "form"])
    df = pd.DataFrame(rows)
    return (
        df.sort_values(["filed", "form"])
        .groupby(["start", "end"], as_index=False)
        .first()
    )


def _extract_flow_concept(
    facts: dict,
    taxonomy: str,
    concept: str,
    unit: str = "USD",
) -> pd.Series:
    """Discrete single-quarter values of a flow concept (see ``_flow_concept_with_filed``)."""
    return _flow_concept_with_filed(facts, taxonomy, concept, unit)[0]


def _flow_concept_with_filed(
    facts: dict,
    taxonomy: str,
    concept: str,
    unit: str = "USD",
) -> tuple[pd.Series, pd.Series]:
    """Extract discrete single-quarter values for a flow (income-statement) concept.

    * Q1–Q3 are the 10-Q facts spanning one quarter (60–120 days).
    * Q4 is not filed as a quarter: it is the 10-K full-year fact
      (320–380 days) less the 10-Q nine-month year-to-date fact (250–290
      days) with the same start date (F09).  A full-year fact without its
      nine-month partner is dropped rather than stored as a quarter.
    * Each window keeps its **earliest-filed** value; a Q4 value is dated by
      the 10-K filing.

    Args:
        facts:    Parsed companyfacts dict from :func:`fetch_companyfacts`.
        taxonomy: XBRL taxonomy namespace, e.g. ``"us-gaap"``.
        concept:  Concept name, e.g. ``"Revenues"``.
        unit:     Expected unit key (``"USD"`` for monetary, ``"USD/shares"``
                  for per-share values).

    Returns:
        ``(values, filed)``: Series indexed by ``period_end`` (str
        ``"YYYY-MM-DD"``); ``values`` is named ``concept`` and ``filed`` holds
        each value's filing date.  Empty if the concept is absent or no
        qualifying facts exist.
    """
    windows = _duration_facts(facts, taxonomy, concept, unit)
    values: dict[str, float] = {}
    filed: dict[str, str] = {}

    quarters = windows[windows["days"].between(_QUARTER_DAYS_MIN, _QUARTER_DAYS_MAX)]
    for rec in quarters.itertuples(index=False):
        values[rec.end] = rec.val
        filed[rec.end] = rec.filed

    nine_months = windows[windows["days"].between(_NINE_MONTH_DAYS_MIN, _NINE_MONTH_DAYS_MAX)]
    nine_by_start = {rec.start: rec for rec in nine_months.itertuples(index=False)}
    annual = windows[
        windows["days"].between(_ANNUAL_DAYS_MIN, _ANNUAL_DAYS_MAX)
        & (windows["form"] == "10-K")
    ]
    for rec in annual.itertuples(index=False):
        if rec.end in values:
            continue
        ytd = nine_by_start.get(rec.start)
        if ytd is None or ytd.end >= rec.end:
            continue
        values[rec.end] = rec.val - ytd.val
        filed[rec.end] = rec.filed

    index = sorted(values)
    series = pd.Series([values[k] for k in index], index=index, name=concept, dtype=float)
    return series, pd.Series([filed[k] for k in index], index=index, name="filed")


def _extract_instant_concept(
    facts: dict,
    taxonomy: str,
    concept: str,
    unit: str = "USD",
) -> pd.Series:
    """Extract point-in-time values for an instant (balance-sheet) XBRL concept.

    Instant concepts have only an ``end`` date (the balance-sheet date).
    This function keeps the **earliest-filed** value per period end date from
    10-Q and 10-K filings (what was first reported).

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
        if not end or rec.get("start"):
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
    df = df.sort_values("filed").groupby("period_end").first()

    series = df["val"].astype(float)
    series.name = concept
    return series


def trailing_roe(net_income: pd.Series, equity: pd.Series) -> pd.Series:
    """Return trailing-12-month ROE: TTM net income / average equity.

    TTM net income is the sum of the four calendar quarters ending at each
    quarter-end; average equity is the mean of the five quarter-end equity
    values spanning that year (t-4 … t).  Over 2009-2026 this tracks the
    trailing ROE printed in PGR's monthly releases within 0.3 points on
    average (FY2018: 2,615.3 / 10,657.9 = 24.5 % vs 24.7 % printed), better
    than the two-point average (0.7 points).  NaN unless all four quarters and
    all five equity points exist.

    Args:
        net_income: Discrete-quarter net income indexed by period-end string.
        equity:     Quarter-end shareholders' equity indexed by date string.

    Returns:
        Series indexed like ``net_income``.
    """
    ni = {pd.Period(k, freq="Q"): v for k, v in net_income.dropna().items()}
    eq = {pd.Period(k, freq="Q"): v for k, v in equity.dropna().items()}
    out: dict[str, float] = {}
    for key in net_income.index:
        q = pd.Period(key, freq="Q")
        window = [ni[q - lag] for lag in range(4) if q - lag in ni]
        points = [eq[q - lag] for lag in range(5) if q - lag in eq]
        if len(window) < 4 or len(points) < 5:
            out[key] = float("nan")
            continue
        avg_equity = sum(points) / 5.0
        out[key] = sum(window) / avg_equity if avg_equity > 0 else float("nan")
    return pd.Series(out, name="roe", dtype=float).reindex(net_income.index)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_pgr_fundamentals_quarterly(
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch PGR quarterly fundamentals from SEC EDGAR XBRL.

    Makes at most one HTTP request (to the companyfacts endpoint); subsequent
    calls within the 7-day cache window are served from disk.

    XBRL concepts fetched (us-gaap taxonomy), each as a discrete quarter with
    Q4 = full year − nine months and earliest-filed values (F09):
      - ``Revenues`` → ``revenue``  (fallback: ``PremiumsEarnedNet``)
      - ``NetIncomeLoss`` → ``net_income``
      - ``EarningsPerShareBasic`` → ``eps``  (Q4 = FY − 9M is approximate for
        a per-share value, as in PGR's own quarterly tables)
      - ``StockholdersEquity`` (instant) → used to derive ``roe``
        (fallback: ``StockholdersEquityAttributableToParent``)

    Derived:
      - ``roe`` = trailing-12-month net income / average of the five
        quarter-end equity values spanning the year (see ``trailing_roe``).
      - ``filing_date`` = filing date of the net-income value (the 10-K for Q4).

    Args:
        force_refresh: If True, bypass the disk cache and re-fetch from EDGAR.

    Returns:
        List of dicts, one per quarter, with keys matching
        ``pgr_fundamentals_quarterly`` columns: ``period_end``, ``roe``,
        ``eps``, ``revenue``, ``net_income``, ``filing_date``, ``source``.
        Rows with no data across all financial columns are dropped.

    Raises:
        requests.HTTPError: If the EDGAR fetch fails.
    """
    facts = fetch_companyfacts(force_refresh=force_refresh)
    return fundamentals_from_companyfacts(facts)


def fundamentals_from_companyfacts(facts: dict) -> list[dict[str, Any]]:
    """Build ``pgr_fundamentals_quarterly`` rows from a companyfacts dict."""
    revenue = _extract_flow_concept(facts, "us-gaap", "Revenues")
    if revenue.empty:
        revenue = _extract_flow_concept(facts, "us-gaap", "PremiumsEarnedNet")
    revenue = revenue.rename("revenue")

    net_income, ni_filed = _flow_concept_with_filed(facts, "us-gaap", "NetIncomeLoss")
    net_income = net_income.rename("net_income")

    eps = _extract_flow_concept(
        facts, "us-gaap", "EarningsPerShareBasic", unit="USD/shares"
    ).rename("eps")

    equity = _extract_instant_concept(facts, "us-gaap", "StockholdersEquity")
    if equity.empty:
        equity = _extract_instant_concept(
            facts, "us-gaap", "StockholdersEquityAttributableToParent"
        )

    frames = [s for s in (revenue, net_income, eps) if not s.empty]
    if not frames:
        return []
    combined: pd.DataFrame = pd.concat(frames, axis=1)
    combined.index.name = "period_end"
    for col in ("revenue", "net_income", "eps"):
        if col not in combined.columns:
            combined[col] = float("nan")

    combined["roe"] = (
        trailing_roe(combined["net_income"], equity)
        if not equity.empty
        else float("nan")
    )
    combined["filing_date"] = ni_filed.reindex(combined.index)
    combined["source"] = "edgar"
    combined = combined.dropna(subset=["revenue", "net_income", "eps"], how="all")
    if combined.empty:
        return []

    combined = combined.sort_index().reset_index()
    combined["period_end"] = combined["period_end"].astype(str)
    records = combined[
        ["period_end", "roe", "eps", "revenue", "net_income", "filing_date", "source"]
    ].to_dict("records")
    for rec in records:
        for key, value in rec.items():
            if isinstance(value, float) and value != value:
                rec[key] = None
    return records
