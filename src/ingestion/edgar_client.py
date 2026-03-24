"""
SEC EDGAR XBRL Company-Concept API client.

Replaces FMP `/v3/income-statement` and `/v3/key-metrics` for PGR quarterly
fundamentals after FMP deprecated all v3 endpoints on 2025-08-31.

API used:  https://data.sec.gov/api/xbrl/companyconcept/{cik}/{taxonomy}/{concept}.json
No API key is required.  Requests must include a descriptive User-Agent per
SEC fair-use policy (https://www.sec.gov/developer).

Data cadence:  quarterly only (10-Q / 10-K).  PGR monthly 8-K earnings
supplements are PDF attachments and are NOT available via the XBRL API.
Use ``pgr_monthly_loader`` for the user-provided monthly CSV cache.

Returned records match the ``pgr_fundamentals_quarterly`` DB schema:
  period_end  (str, YYYY-MM-DD)
  eps         (float | None)
  revenue     (float | None)
  net_income  (float | None)
  pe_ratio    (float | None)  — not in XBRL; always None
  pb_ratio    (float | None)  — not in XBRL; always None
  roe         (float | None)  — not in XBRL; always None
  source      (str)           — "edgar_xbrl"
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://data.sec.gov/api/xbrl/companyconcept"

# CIK for The Progressive Corporation (leading zeros padded to 10 digits).
_PGR_CIK = "0000080661"

# XBRL concepts we fetch from the us-gaap taxonomy.
# Keys are the concept tag names; values are the canonical DB column names.
_CONCEPTS: dict[str, str] = {
    "EarningsPerShareDiluted": "eps",
    "Revenues": "revenue",
    "NetIncomeLoss": "net_income",
}

# SEC requests a User-Agent string identifying the application and contact.
_HEADERS = {
    "User-Agent": "pgr-vesting-decision-support jeff@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# Polite delay between consecutive API calls (seconds).
_INTER_REQUEST_DELAY = 0.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_concept(
    cik: str,
    taxonomy: str,
    concept: str,
    *,
    timeout: int = 20,
) -> list[dict[str, Any]]:
    """Fetch all filings for one XBRL concept and return the ``units`` list.

    Returns the list of quarterly/annual data points as dicts with keys:
    ``accn``, ``cik``, ``entityName``, ``loc``, ``end``, ``val``, ``form``,
    ``filed``, ``frame`` (the last being absent for annual-only filings).
    """
    url = f"{_BASE_URL}/{cik}/us-gaap/{concept}.json"
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    # Units section contains one or more unit keys (e.g., "USD", "USD/shares").
    units: dict[str, list] = payload.get("units", {})
    # For monetary / per-share values there should be exactly one unit key.
    for rows in units.values():
        return rows  # type: ignore[return-value]
    return []


def _filter_quarterly(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only 10-Q and 10-K instant/duration rows with an ``end`` date.

    EDGAR returns both quarterly (10-Q) and annual (10-K) filings, plus any
    amended forms.  We keep form types matching ``10-Q`` or ``10-K``.
    We deduplicate on ``end`` date, keeping the most recently filed row.
    """
    kept: dict[str, dict[str, Any]] = {}
    for row in rows:
        form = row.get("form", "")
        if form not in ("10-Q", "10-K"):
            continue
        end = row.get("end", "")
        if not end:
            continue
        # Prefer later filings for the same period-end (amendments).
        existing = kept.get(end)
        if existing is None or row.get("filed", "") > existing.get("filed", ""):
            kept[end] = row
    return list(kept.values())


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_pgr_quarterly_fundamentals(
    *,
    start_date: str | None = None,
    timeout: int = 20,
) -> list[dict[str, Any]]:
    """Fetch PGR quarterly fundamentals from SEC EDGAR XBRL.

    Retrieves EPS (diluted), revenue, and net income from 10-Q/10-K filings
    for The Progressive Corporation (CIK 0000080661).

    Args:
        start_date:  Optional ISO date string ``"YYYY-MM-DD"``.  Rows with
                     ``period_end < start_date`` are excluded.  Useful for
                     incremental refreshes.
        timeout:     HTTP request timeout in seconds.

    Returns:
        List of dicts compatible with ``db_client.upsert_pgr_fundamentals()``:
        ``{period_end, eps, revenue, net_income, pe_ratio, pb_ratio, roe, source}``.
        ``pe_ratio``, ``pb_ratio``, and ``roe`` are always ``None`` (not
        available in XBRL).
    """
    concept_data: dict[str, dict[str, float]] = {}  # period_end -> {col: value}

    for concept_tag, col_name in _CONCEPTS.items():
        rows = _fetch_concept(_PGR_CIK, "us-gaap", concept_tag, timeout=timeout)
        quarterly = _filter_quarterly(rows)
        for row in quarterly:
            period_end = row["end"]
            if start_date and period_end < start_date:
                continue
            bucket = concept_data.setdefault(period_end, {})
            bucket[col_name] = float(row["val"])
        time.sleep(_INTER_REQUEST_DELAY)

    records: list[dict[str, Any]] = []
    for period_end, values in sorted(concept_data.items()):
        records.append(
            {
                "period_end": period_end,
                "eps":        values.get("eps"),
                "revenue":    values.get("revenue"),
                "net_income": values.get("net_income"),
                "pe_ratio":   None,
                "pb_ratio":   None,
                "roe":        None,
                "source":     "edgar_xbrl",
            }
        )
    return records


def fetch_pgr_latest_quarter(*, timeout: int = 20) -> dict[str, Any] | None:
    """Convenience wrapper: return only the most recent quarterly record.

    Returns ``None`` if no data is available.
    """
    records = fetch_pgr_quarterly_fundamentals(timeout=timeout)
    if not records:
        return None
    return max(records, key=lambda r: r["period_end"])
