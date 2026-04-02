#!/usr/bin/env python3
"""
Fetch PGR monthly 8-K operating metrics from SEC EDGAR and upsert to the DB.

PGR files Regulation FD supplements (item 7.01) each month disclosing:
  - Combined Ratio (GAAP): loss ratio + expense ratio; below 96% = PGR target
  - Policies in Force (PIF): total count across all segments
  - Underlying metrics used to estimate the annual Gainshare multiplier (0.0–2.0)

Schedule (see .github/workflows/monthly_8k_fetch.yml):
  - Primary run:  20th of each month at 14:00 UTC
  - Fallback run: 25th of each month at 14:00 UTC (covers late filers)

Both runs are idempotent: ``db_client.upsert_pgr_edgar_monthly`` uses
``INSERT OR REPLACE``, so running the fetcher twice in the same month is a safe
no-op when no new data is present, and correctly overwrites the existing row
when updated data is available.

EDGAR pagination:
  The primary submissions JSON (CIK0000080661.json) only contains the ~40 most
  recent filings.  Older filings are in paginated files listed in
  ``filings.files``: ``CIK0000080661-submissions-001.json``, etc.
  This script fetches all pagination files until the requested cutoff date is
  exceeded, giving access to the full PGR 8-K history.

HTML parsing coverage:
  PGR's monthly supplement format has been broadly consistent since ~2010.
  Pre-2015 filings sometimes use legacy table layouts; parse exceptions are
  caught per-filing and logged — one bad filing never aborts the full run.

SEC EDGAR rate limits: 10 requests/second max; ``User-Agent`` header required.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

# Resolve project root so this script can be run directly or via GitHub Actions.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PGR_CIK: str = "CIK0000080661"
PGR_CIK_NUMERIC: int = 80661          # numeric CIK for archive URLs

SUBMISSIONS_BASE_URL: str = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES_URL: str = "https://www.sec.gov/Archives/edgar/data/80661"

_EDGAR_HEADERS: dict[str, str] = {
    "User-Agent": "Jeff Hester jeffrey.r.hester@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# PGR's monthly 8-K supplement HTML format is reliably parseable from ~2010
# onward.  Set this as the hard earliest boundary for backfills.
BACKFILL_EARLIEST_DATE: str = "2010-01-01"

# Polite rate limit: well under SEC's 10 req/s ceiling.
_SEC_SLEEP_SECONDS: float = 0.15


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> requests.Response:
    """GET with exponential back-off retry and polite inter-request delay."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=30)
            resp.raise_for_status()
            time.sleep(_SEC_SLEEP_SECONDS)
            return resp
        except requests.HTTPError as exc:
            last_exc = exc
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** (attempt + 1)
                log.warning(
                    "HTTP %d on %s — retrying in %ds", resp.status_code, url, wait
                )
                time.sleep(wait)
            else:
                raise
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(
        f"Failed to GET {url} after {retries} attempts: {last_exc}"
    )


# ---------------------------------------------------------------------------
# EDGAR submissions fetch + pagination
# ---------------------------------------------------------------------------

def _fetch_submissions_page(page_id: str | None = None) -> dict:
    """Fetch one EDGAR submissions page for PGR.

    Args:
        page_id: ``None`` for the primary file, or e.g. ``"001"`` for the
            first paginated overflow file.

    Returns:
        Parsed JSON dict from the EDGAR submissions endpoint.
    """
    if page_id is None:
        url = f"{SUBMISSIONS_BASE_URL}/{PGR_CIK}.json"
    else:
        url = f"{SUBMISSIONS_BASE_URL}/{PGR_CIK}-submissions-{page_id}.json"
    resp = _get(url)
    return resp.json()


def _collect_8k_filings(
    recent: dict,
    cutoff_date: str,
    out: list[dict[str, Any]],
) -> bool:
    """Extract 8-K / item 7.01 filings from one ``filings.recent`` block.

    Modifies ``out`` in-place with matching filings.

    Args:
        recent: Dict with parallel arrays ``form``, ``filingDate``, ``items``,
            ``accessionNumber`` as returned by the EDGAR submissions endpoint.
        cutoff_date: ISO date string (``"YYYY-MM-DD"``).  Filings with
            ``filingDate < cutoff_date`` are excluded.
        out: List to append matched filings to.

    Returns:
        ``True`` if any filing in this block predates ``cutoff_date``, which
        signals the caller to stop fetching older pagination pages.
    """
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    items_list = recent.get("items", [])
    accessions = recent.get("accessionNumber", [])

    passed_cutoff = False
    for form, filing_date, item_str, accession in zip(
        forms, dates, items_list, accessions
    ):
        if filing_date < cutoff_date:
            passed_cutoff = True
            continue
        if form != "8-K":
            continue
        # ``items`` is a comma-separated string like ``"7.01,9.01"``
        parsed_items = [i.strip() for i in str(item_str).split(",")]
        if "7.01" not in parsed_items:
            continue
        # Clean accession number: remove dashes for use in archive paths
        cleaned = accession.replace("-", "")
        out.append(
            {
                "accession_number": cleaned,
                "accession_dashed": accession,
                "filing_date": filing_date,
                "form": form,
                "items": item_str,
            }
        )

    return passed_cutoff


def fetch_all_8k_filings(cutoff_date: str) -> list[dict[str, Any]]:
    """Fetch all PGR 8-K (item 7.01) filings back to ``cutoff_date``.

    Reads the primary submissions JSON then follows all paginated overflow
    files until ``cutoff_date`` is exceeded in the filing history.

    Args:
        cutoff_date: ISO date string; oldest filing date to include.

    Returns:
        List of filing dicts sorted ascending by ``filing_date``.  Each dict
        has keys: ``accession_number`` (no dashes), ``accession_dashed``,
        ``filing_date``, ``form``, ``items``.
    """
    results: list[dict[str, Any]] = []

    log.info("Fetching primary EDGAR submissions for PGR …")
    primary = _fetch_submissions_page()
    recent = primary.get("filings", {}).get("recent", {})
    _collect_8k_filings(recent, cutoff_date, results)

    # Pagination overflow files are listed in primary["filings"]["files"]
    extra_files = primary.get("filings", {}).get("files", [])
    for file_entry in extra_files:
        name = file_entry.get("name", "")
        m = re.search(r"-submissions-(\d+)\.json$", name)
        if not m:
            continue
        page_id = m.group(1)
        log.info("Fetching pagination file %s …", name)
        page_data = _fetch_submissions_page(page_id)
        page_recent = page_data.get("filings", {}).get("recent", {})
        stop = _collect_8k_filings(page_recent, cutoff_date, results)
        if stop:
            log.debug("Oldest filing in %s precedes cutoff — stopping pagination.", name)
            break

    results.sort(key=lambda r: r["filing_date"])
    log.info(
        "Found %d 8-K (item 7.01) filings back to %s.", len(results), cutoff_date
    )
    return results


# ---------------------------------------------------------------------------
# Filing document URL resolution
# ---------------------------------------------------------------------------

def _get_filing_doc_url(
    accession_number: str,
    accession_dashed: str,
) -> str | None:
    """Resolve the primary HTML exhibit URL for an 8-K filing.

    Fetches the filing index page and returns the first suitable ``.htm``
    document, preferring files whose name contains ``"8k"`` or ``"pgr"``.

    Args:
        accession_number: Cleaned accession (no dashes), e.g.
            ``"000008066124000001"``.
        accession_dashed: Original dashed form, e.g.
            ``"0000080661-24-000001"``.

    Returns:
        Full URL to the primary HTML exhibit, or ``None`` if none found.
    """
    index_url = (
        f"{EDGAR_ARCHIVES_URL}/{accession_number}"
        f"/{accession_dashed}-index.htm"
    )
    try:
        resp = _get(index_url)
        html = resp.text
    except Exception as exc:
        log.debug("Cannot fetch index for %s: %s", accession_number, exc)
        return None

    # Extract all .htm hrefs from the filing index
    pattern = re.compile(
        r'href="(/Archives/edgar/data/80661/[^"]+\.htm)"',
        re.IGNORECASE,
    )
    candidates = pattern.findall(html)

    for href in candidates:
        fname = href.split("/")[-1].lower()
        # Skip XBRL viewer and inline XBRL files
        if any(skip in href for skip in ("ix?doc=", "R1.htm", "R2.htm")):
            continue
        if fname.startswith("r") and fname[1:].isdigit():
            continue
        # Prefer PGR/8-K named files
        if "8k" in fname or "8-k" in fname or "pgr" in fname:
            return f"https://www.sec.gov{href}"

    # Fall back to first candidate that isn't a viewer redirect
    for href in candidates:
        if "ix?doc=" not in href:
            return f"https://www.sec.gov{href}"

    return None


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _parse_html_exhibit(
    html: str,
    filing_date: str,
) -> dict[str, Any] | None:
    """Parse a PGR 8-K HTML exhibit for combined_ratio and PIF.

    PGR's monthly supplement contains a table with:
      - Combined Ratio (loss + expense; typically 85–105 for PGR)
      - Policies in Force (total count; typically 10M–30M)

    The ``gainshare_estimate`` and ``pif_growth_yoy`` fields are left as
    ``None`` here; they are computed in ``_compute_derived_fields`` once the
    full sorted time series is available.

    The filing date is used to derive ``month_end``: PGR files its monthly
    supplement in the first 3 weeks of the following month, so the data
    period is the month immediately prior to the filing date.

    Args:
        html: Raw HTML text of the 8-K exhibit.
        filing_date: ISO date string (``"YYYY-MM-DD"``).

    Returns:
        Dict with keys ``month_end``, ``combined_ratio``, ``pif_total``,
        ``pif_growth_yoy``, ``gainshare_estimate``, or ``None`` if no usable
        data is found.
    """
    filed_dt = datetime.strptime(filing_date, "%Y-%m-%d")
    # Period: last day of the month before the filing month
    first_of_filing_month = filed_dt.replace(day=1)
    last_day_prior_month = first_of_filing_month - timedelta(days=1)
    month_end = last_day_prior_month.strftime("%Y-%m-%d")

    combined_ratio: float | None = None
    pif_total: float | None = None

    # --- Combined Ratio extraction ---
    # Try multiple regex patterns, from most specific to most permissive.
    cr_patterns = [
        # Table cell pattern: label cell, then value cell
        r"(?i)combined\s+ratio[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*([\d]+\.[\d]+)",
        # Inline near label
        r"(?i)combined\s+ratio\b[^<]{0,80}?([\d]{2,3}\.[\d]{1,2})",
        # Abbreviated "CR" with value in nearby cell
        r"(?i)\bCR\b[^<]{0,40}</t[dh]>\s*<t[dh][^>]*>\s*([\d]+\.[\d]+)",
        # Looser: anywhere "combined" appears near a plausible number
        r"(?si)combined.{0,300}?(\b\d{2,3}\.\d{1,2}\b)",
    ]
    for pat in cr_patterns:
        m = re.search(pat, html)
        if m:
            try:
                val = float(m.group(1))
                if 60.0 <= val <= 140.0:
                    combined_ratio = val
                    break
            except ValueError:
                pass

    # --- Policies in Force extraction ---
    pif_patterns = [
        # Table cell pattern
        r"(?i)policies\s+in\s+force[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*([\d,]+)",
        r"(?i)total\s+pif\b[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*([\d,]+)",
        # Inline near label
        r"(?i)policies\s+in\s+force\b[^<]{0,80}?([\d,]{6,})",
        r"(?i)\bpif\b[^<]{0,40}</t[dh]>\s*<t[dh][^>]*>\s*([\d,]+)",
        # Looser
        r"(?si)policies\s+in\s+force.{0,300}?(\b[\d,]{6,}\b)",
    ]
    for pat in pif_patterns:
        m = re.search(pat, html)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                # PGR PIF: realistic range 100k–60M (accounts for growth over 15 years)
                if 100_000 <= val <= 60_000_000:
                    pif_total = val
                    break
            except ValueError:
                pass

    if combined_ratio is None and pif_total is None:
        return None

    return {
        "month_end": month_end,
        "combined_ratio": combined_ratio,
        "pif_total": pif_total,
        "pif_growth_yoy": None,
        "gainshare_estimate": None,
    }


# ---------------------------------------------------------------------------
# Derived field computation
# ---------------------------------------------------------------------------

def _compute_derived_fields(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute ``pif_growth_yoy`` and ``gainshare_estimate`` for all records.

    These fields require a full sorted time series so that year-ago values can
    be looked up.  Must be called after all records are collected and sorted
    ascending by ``month_end``.

    The gainshare formula mirrors ``pgr_monthly_loader.py``:
      - ``cr_score``    = clip((96 − CR) / 10,  0, 2)
      - ``pif_score``   = clip(pif_growth / 0.10, 0, 2)
      - ``gainshare``   = 0.5 × cr_score + 0.5 × pif_score

    Args:
        records: List of record dicts, already sorted ascending by ``month_end``.

    Returns:
        The same list with ``pif_growth_yoy`` and ``gainshare_estimate`` filled
        in where computable.
    """
    pif_by_month: dict[str, float] = {
        r["month_end"]: r["pif_total"]
        for r in records
        if r["pif_total"] is not None
    }

    for rec in records:
        # --- YoY PIF growth ---
        if rec["pif_total"] is not None:
            me = datetime.strptime(rec["month_end"], "%Y-%m-%d")
            # Safe prior-year lookup: Feb 29 in a leap year → Feb 28 in a non-leap year
            try:
                prior_dt = me.replace(year=me.year - 1)
            except ValueError:
                prior_dt = me.replace(year=me.year - 1, day=28)
            prior_key = prior_dt.strftime("%Y-%m-%d")
            prior_pif = pif_by_month.get(prior_key)
            if prior_pif and prior_pif != 0.0:
                rec["pif_growth_yoy"] = (rec["pif_total"] - prior_pif) / prior_pif

        # --- Gainshare estimate ---
        cr = rec.get("combined_ratio")
        pif_growth = rec.get("pif_growth_yoy")

        cr_score: float | None = None
        pif_score: float | None = None

        if cr is not None:
            cr_score = min(max((96.0 - cr) / 10.0, 0.0), 2.0)
        if pif_growth is not None:
            pif_score = min(max(pif_growth / 0.10, 0.0), 2.0)

        if cr_score is not None and pif_score is not None:
            rec["gainshare_estimate"] = 0.5 * cr_score + 0.5 * pif_score
        elif cr_score is not None:
            rec["gainshare_estimate"] = cr_score
        elif pif_score is not None:
            rec["gainshare_estimate"] = pif_score

    return records


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def check_staleness(conn: sqlite3.Connection) -> None:
    """Log a warning if the most recent 8-K data is more than 45 days old.

    PGR typically files within 20 days of month-end.  If the newest row is
    older than 45 days it almost certainly means a filing was missed (or the
    workflow's primary-pass run failed and the fallback hasn't fired yet).

    Args:
        conn: Open SQLite connection with ``pgr_edgar_monthly`` populated.
    """
    row = conn.execute(
        "SELECT MAX(month_end) FROM pgr_edgar_monthly"
    ).fetchone()

    if row is None or row[0] is None:
        log.warning("WARNING: pgr_edgar_monthly table is empty — no 8-K data present.")
        return

    most_recent = datetime.strptime(row[0], "%Y-%m-%d").date()
    age_days = (date.today() - most_recent).days

    if age_days > 45:
        log.warning(
            "WARNING: Most recent 8-K data is %d days old — "
            "PGR may not have filed yet.",
            age_days,
        )
    else:
        log.info(
            "Most recent 8-K data: %s (%d days old).", row[0], age_days
        )


# ---------------------------------------------------------------------------
# Main fetch-and-upsert logic
# ---------------------------------------------------------------------------

def fetch_and_upsert(
    conn: sqlite3.Connection,
    backfill_years: int = 2,
    dry_run: bool = False,
) -> int:
    """Fetch PGR 8-K operating metrics and upsert them to the DB.

    Workflow:
      1. Compute cutoff date (today minus backfill_years, floored at
         BACKFILL_EARLIEST_DATE).
      2. Fetch all 8-K (item 7.01) filings from EDGAR submissions (with
         pagination) back to the cutoff.
      3. For each filing, resolve the primary HTML exhibit URL, parse it for
         combined_ratio and PIF, and collect the result.  Parse failures are
         logged and skipped (never abort the full run).
      4. Compute derived fields (pif_growth_yoy, gainshare_estimate) over the
         full sorted time series.
      5. Deduplicate by month_end (last filing for that period wins).
      6. Upsert all rows via db_client.upsert_pgr_edgar_monthly (INSERT OR REPLACE).

    Args:
        conn: Open SQLite connection.
        backfill_years: How many years back to fetch (default: 2).
        dry_run: If True, parse everything but skip the DB write.

    Returns:
        Number of rows upserted (0 for dry runs).
    """
    today = date.today()
    cutoff_raw = date(today.year - backfill_years, today.month, today.day)
    earliest = date.fromisoformat(BACKFILL_EARLIEST_DATE)
    effective_cutoff = max(cutoff_raw, earliest).isoformat()

    log.info(
        "Backfill window: %s → %s  (backfill_years=%d)",
        effective_cutoff,
        today.isoformat(),
        backfill_years,
    )

    filings = fetch_all_8k_filings(cutoff_date=effective_cutoff)
    if not filings:
        log.info("No 8-K (item 7.01) filings found in the requested date range.")
        return 0

    records: list[dict[str, Any]] = []
    parse_errors = 0

    for filing in filings:
        accession = filing["accession_number"]
        accession_dashed = filing["accession_dashed"]
        filing_date = filing["filing_date"]
        log.debug("Processing %s (filed %s) …", accession, filing_date)

        try:
            doc_url = _get_filing_doc_url(accession, accession_dashed)
            if doc_url is None:
                log.debug("No HTML exhibit found for %s — skipping.", accession)
                continue

            resp = _get(doc_url)
            html = resp.text
            parsed = _parse_html_exhibit(html, filing_date)

            if parsed is None:
                log.debug(
                    "No parseable data in %s (filed %s).", accession, filing_date
                )
                continue

            log.info(
                "Parsed %s  month_end=%-12s  CR=%-6s  PIF=%s",
                accession,
                parsed["month_end"],
                f"{parsed['combined_ratio']:.1f}" if parsed["combined_ratio"] else "n/a",
                f"{parsed['pif_total']:,.0f}" if parsed["pif_total"] else "n/a",
            )
            records.append(parsed)

        except Exception as exc:
            parse_errors += 1
            log.warning(
                "SKIP %s (filed %s): %s", accession, filing_date, exc
            )
            continue

    if parse_errors > 0:
        log.warning("%d filing(s) skipped due to parse errors.", parse_errors)

    if not records:
        log.info("No records to upsert.")
        return 0

    # Sort, derive fields, deduplicate
    records.sort(key=lambda r: r["month_end"])
    records = _compute_derived_fields(records)

    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        seen[rec["month_end"]] = rec
    deduped = sorted(seen.values(), key=lambda r: r["month_end"])

    # Coverage report
    n_total = len(deduped)
    cr_present = sum(1 for r in deduped if r.get("combined_ratio") is not None)
    pif_present = sum(1 for r in deduped if r.get("pif_total") is not None)
    gs_present = sum(1 for r in deduped if r.get("gainshare_estimate") is not None)
    log.info(
        "Coverage  combined_ratio=%d/%d  pif_total=%d/%d  gainshare=%d/%d  "
        "date_range=%s→%s",
        cr_present, n_total,
        pif_present, n_total,
        gs_present, n_total,
        deduped[0]["month_end"],
        deduped[-1]["month_end"],
    )

    if dry_run:
        log.info("Dry run — skipping DB write (%d rows would be upserted).", n_total)
        return 0

    n = db_client.upsert_pgr_edgar_monthly(conn, deduped)
    log.info("Upserted %d rows to pgr_edgar_monthly.", n)
    return n


# ---------------------------------------------------------------------------
# CSV seed loader (bootstraps pgr_edgar_monthly from the committed CSV)
# ---------------------------------------------------------------------------

def load_from_csv(
    conn: sqlite3.Connection,
    csv_path: str,
    dry_run: bool = False,
) -> int:
    """Seed ``pgr_edgar_monthly`` from the committed ``pgr_edgar_cache.csv``.

    The CSV (``data/processed/pgr_edgar_cache.csv``) contains 256+ rows of
    monthly PGR data going back to 2004, pre-extracted from SEC EDGAR filings.
    This function converts the CSV's ``report_period`` (``"YYYY-MM"``) to
    ``month_end`` (last calendar day of that month, ``"YYYY-MM-DD"``), maps all
    65 CSV columns into the v6.2 expanded DB schema, computes derived features,
    and upserts all rows.

    No network calls are made.  The regular ``fetch_and_upsert`` EDGAR fetch
    covers recent months not yet in the CSV.

    Args:
        conn: Open SQLite connection.
        csv_path: Path to ``pgr_edgar_cache.csv``.
        dry_run: If True, parse but skip the DB write.

    Returns:
        Number of rows upserted (0 for dry runs).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If ``report_period`` column is missing from the CSV.
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log.info("Loading historical data from %s …", csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "report_period" not in df.columns:
        raise ValueError(
            f"Expected 'report_period' column in {csv_path}; "
            f"found: {list(df.columns[:10])}"
        )

    # Convert "YYYY-MM" → last calendar day of that month ("YYYY-MM-DD")
    df["month_end"] = (
        pd.to_datetime(df["report_period"].astype(str), format="%Y-%m")
        + pd.offsets.MonthEnd(0)
    )
    df["month_end"] = df["month_end"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("month_end").reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Direct CSV column → DB column mappings (v6.2 expanded schema)
    # CSV column name              DB column name
    # -----------------------------------------------------------------------
    DIRECT_MAP: dict[str, str] = {
        "combined_ratio":              "combined_ratio",
        "pif_total":                   "pif_total",
        "net_premiums_written":        "net_premiums_written",
        "net_premiums_earned":         "net_premiums_earned",
        "net_income":                  "net_income",
        "eps_diluted":                 "eps_diluted",
        "eps_basic":                   "eps_basic",
        "loss_lae_ratio":              "loss_lae_ratio",
        "expense_ratio":               "expense_ratio",
        "book_value_per_share":        "book_value_per_share",
        # Segment-level channel metrics
        "npw_agency":                  "npw_agency",
        "npw_direct":                  "npw_direct",
        "npw_commercial":              "npw_commercial",
        "npw_property":                "npw_property",
        "npe_agency":                  "npe_agency",
        "npe_direct":                  "npe_direct",
        "npe_commercial":              "npe_commercial",
        "npe_property":                "npe_property",
        "pif_agency_auto":             "pif_agency_auto",
        "pif_direct_auto":             "pif_direct_auto",
        "pif_commercial_lines":        "pif_commercial_lines",
        "pif_total_personal_lines":    "pif_total_personal_lines",
        # Company-level operating metrics
        "investment_income":           "investment_income",
        "total_revenues":              "total_revenues",
        "total_expenses":              "total_expenses",
        "income_before_income_taxes":  "income_before_income_taxes",
        "roe_net_income_trailing_12m": "roe_net_income_ttm",  # CSV name differs
        "shareholders_equity":         "shareholders_equity",
        "total_assets":                "total_assets",
        "unearned_premiums":           "unearned_premiums",
        "shares_repurchased":          "shares_repurchased",
        "avg_cost_per_share":          "avg_cost_per_share",
        # Investment portfolio metrics
        "fte_return_total_portfolio":  "fte_return_total_portfolio",
        "investment_book_yield":       "investment_book_yield",
        "net_unrealized_gains_fixed":  "net_unrealized_gains_fixed",
        "fixed_income_duration":       "fixed_income_duration",
    }

    for csv_col, db_col in DIRECT_MAP.items():
        if csv_col in df.columns:
            df[db_col] = pd.to_numeric(df[csv_col], errors="coerce")
        else:
            df[db_col] = float("nan")

    # -----------------------------------------------------------------------
    # Derived fields (computed once the full sorted time series is available)
    # -----------------------------------------------------------------------

    # pif_growth_yoy and gainshare_estimate: reuse existing logic via records path
    # npw_growth_yoy: 12-month YoY on net_premiums_written
    df["npw_growth_yoy"] = df["net_premiums_written"].pct_change(periods=12, fill_method=None)

    # channel_mix_agency_pct = npw_agency / (npw_agency + npw_direct)
    npw_pl = df["npw_agency"] + df["npw_direct"]
    df["channel_mix_agency_pct"] = df["npw_agency"].where(npw_pl > 0) / npw_pl.where(npw_pl > 0)

    # underwriting_income = npe * (1 - combined_ratio / 100)
    df["underwriting_income"] = df["net_premiums_earned"] * (
        1.0 - df["combined_ratio"] / 100.0
    )

    # unearned_premium_growth_yoy: 12-month YoY on unearned_premiums
    df["unearned_premium_growth_yoy"] = df["unearned_premiums"].pct_change(periods=12, fill_method=None)

    # buyback_yield requires market_cap (price data) — set NULL for CSV path
    df["buyback_yield"] = float("nan")

    # -----------------------------------------------------------------------
    # Build records list and apply pif_growth_yoy / gainshare via existing helper
    # -----------------------------------------------------------------------
    db_cols = list(DIRECT_MAP.values()) + [
        "month_end",
        "npw_growth_yoy", "channel_mix_agency_pct",
        "underwriting_income", "unearned_premium_growth_yoy", "buyback_yield",
    ]
    # Deduplicate column list (roe_net_income_ttm could appear once)
    seen_cols: set[str] = set()
    unique_cols = []
    for c in db_cols:
        if c not in seen_cols:
            unique_cols.append(c)
            seen_cols.add(c)

    df_out = df[unique_cols].copy()

    def _nan_to_none(val: Any) -> Any:
        """Convert float NaN to None for SQLite NULL storage."""
        try:
            if val != val:  # NaN check
                return None
        except TypeError:
            pass
        return val

    records_raw: list[dict[str, Any]] = [
        {col: _nan_to_none(row[col]) for col in unique_cols}
        for _, row in df_out.iterrows()
    ]

    # Compute pif_growth_yoy and gainshare_estimate using the existing helper
    # (operates on sorted list; only needs pif_total and combined_ratio)
    records_raw = _compute_derived_fields(records_raw)

    log.info(
        "CSV loaded: %d rows  date_range=%s→%s",
        len(records_raw),
        records_raw[0]["month_end"] if records_raw else "n/a",
        records_raw[-1]["month_end"] if records_raw else "n/a",
    )

    # Coverage summary
    def _coverage(field: str) -> str:
        n = sum(1 for r in records_raw if r.get(field) is not None)
        return f"{n}/{len(records_raw)}"

    log.info(
        "Coverage  combined_ratio=%s  npw=%s  npw_agency=%s  "
        "investment_income=%s  book_value=%s  gainshare=%s",
        _coverage("combined_ratio"),
        _coverage("net_premiums_written"),
        _coverage("npw_agency"),
        _coverage("investment_income"),
        _coverage("book_value_per_share"),
        _coverage("gainshare_estimate"),
    )

    if dry_run:
        log.info("Dry run — skipping DB write (%d rows would be upserted).", len(records_raw))
        return 0

    n = db_client.upsert_pgr_edgar_monthly(conn, records_raw)
    log.info("Upserted %d rows from CSV to pgr_edgar_monthly.", n)
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch PGR monthly 8-K operating metrics from SEC EDGAR "
            "and upsert them into the local SQLite database."
        )
    )
    parser.add_argument(
        "--backfill-years",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Number of years back to fetch from EDGAR (default: 2).  "
            "Set to a large value (e.g. 15) for a full historical backfill "
            f"back to {BACKFILL_EARLIEST_DATE}.  "
            "Note: if the committed pgr_edgar_cache.csv already covers this "
            "range, use --load-from-csv instead to avoid unnecessary HTTP calls."
        ),
    )
    parser.add_argument(
        "--load-from-csv",
        metavar="PATH",
        nargs="?",
        const=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "processed", "pgr_edgar_cache.csv",
        ),
        default=None,
        help=(
            "Seed pgr_edgar_monthly from an existing CSV file instead of "
            "fetching from EDGAR.  Defaults to data/processed/pgr_edgar_cache.csv "
            "when the flag is given without a path.  No network calls are made."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse/read data but do not write to the database.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    try:
        if args.load_from_csv is not None:
            n = load_from_csv(conn, args.load_from_csv, dry_run=args.dry_run)
        else:
            n = fetch_and_upsert(
                conn,
                backfill_years=args.backfill_years,
                dry_run=args.dry_run,
            )
        check_staleness(conn)
        log.info("Done. %d rows written.", n)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
