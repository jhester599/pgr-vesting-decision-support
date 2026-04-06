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
import io
import json
import logging
import os
import re
import sqlite3
import sys
import time
from html import unescape
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

# Resolve project root so this script can be run directly or via GitHub Actions.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.logging_config import configure_logging


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PGR_CIK: str = "CIK0000080661"
PGR_CIK_NUMERIC: int = 80661          # numeric CIK for archive URLs

SUBMISSIONS_BASE_URL: str = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES_URL: str = "https://www.sec.gov/Archives/edgar/data/80661"

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
            resp = requests.get(
                url,
                headers=config.build_edgar_headers(),
                timeout=30,
            )
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
        log.debug("Cannot fetch index for %s: %s", accession_number, exc, exc_info=True)
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

def _try_parse_dollar(
    html: str,
    patterns: list[str],
    lo: float,
    hi: float,
    scale: float = 1.0,
) -> float | None:
    """Try each regex in turn; return the first in-range numeric match, or None.

    Args:
        html:     Raw HTML text to search.
        patterns: Ordered list of regex patterns (most-specific first).
                  Each pattern must have exactly one capture group for the value.
        lo:       Inclusive lower bound on the parsed value (after scaling).
        hi:       Inclusive upper bound on the parsed value (after scaling).
        scale:    Multiply the raw parsed number by this factor before range
                  check (e.g. 1e6 to convert millions reported in the HTML to
                  absolute dollars).

    Returns:
        The first valid parsed value, or ``None`` if no pattern matches.
    """
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            try:
                raw = float(m.group(1).replace(",", "").replace("$", ""))
                val = raw * scale
                if lo <= val <= hi:
                    return val
            except (ValueError, AttributeError):
                pass
    return None


def _strip_html_text(html: str) -> str:
    """Return a whitespace-normalized plain-text view of an HTML exhibit."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _try_parse_last_text_match(
    text: str,
    patterns: list[str],
    lo: float,
    hi: float,
    scale: float = 1.0,
) -> float | None:
    """Return the last valid text-mode regex match after scaling/range check."""
    for pat in patterns:
        matches = list(re.finditer(pat, text))
        for match in reversed(matches):
            try:
                raw = float(match.group(1).replace(",", "").replace("$", ""))
            except (ValueError, AttributeError):
                continue
            val = raw * scale
            if lo <= val <= hi:
                return val
    return None


def _dedup_row(row: pd.Series) -> list[str]:
    """Collapse duplicate consecutive table-cell values into unique tokens."""
    tokens: list[str] = []
    prev: str | None = None
    for val in row:
        s = str(val).strip()
        if s in ("nan", "NaN", "None", ""):
            continue
        if s != prev:
            tokens.append(s)
            prev = s
    return tokens


def _try_float(val: str) -> float | None:
    """Parse a number string, handling commas, $, %, and parentheses."""
    cleaned = (
        val.replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .replace("\xa0", " ")
        .strip()
    )
    if cleaned.lower() in ("nan", "none", "n/a", ""):
        return None
    cleaned = cleaned.replace("(", "-").replace(")", "").strip()
    try:
        result = float(cleaned)
        if result != result:
            return None
        return result
    except ValueError:
        return None


def _extract_numerics(tokens: list[str]) -> list[float]:
    """Return all parseable numeric values from a token list."""
    nums: list[float] = []
    for token in tokens:
        value = _try_float(token)
        if value is not None:
            nums.append(value)
    return nums


def _read_exhibit_tables(html: str) -> list[pd.DataFrame]:
    """Parse all tabular structures from an exhibit HTML document."""
    try:
        return pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception:
        log.debug("_read_exhibit_tables: pd.read_html parse failed; returning empty list", exc_info=True)
        return []


def _normalise_pif_value(value: float | None) -> float | None:
    """Normalise PIF counts to the canonical 'thousands of policies' unit."""
    if value is None:
        return None
    if value >= 100_000:
        return value / 1_000.0
    return value


def _parse_segment_metrics(
    nums: list[float],
) -> tuple[float, float, float | None, float | None, float]:
    """Infer segment metric ordering from a six-value company row."""
    agency, direct = nums[0], nums[1]
    middle = list(nums[2:5])
    company_total = nums[-1]

    subtotal_idx = None
    for idx, candidate in enumerate(middle):
        if candidate <= 0:
            continue
        if abs(candidate - (agency + direct)) / candidate <= 0.03:
            subtotal_idx = idx
            break
        others = [value for j, value in enumerate(middle) if j != idx]
        if others and abs(candidate - (agency + direct + min(others))) / candidate <= 0.03:
            subtotal_idx = idx
            break

    segments = [value for idx, value in enumerate(middle) if idx != subtotal_idx]
    commercial = max(segments) if segments else None
    property_value = min(segments) if len(segments) >= 2 else None

    return agency, direct, commercial, property_value, company_total


def _parse_html_exhibit(
    html: str,
    filing_date: str,
) -> dict[str, Any] | None:
    """Parse a PGR 8-K HTML exhibit for operating metrics.

    PGR's monthly Regulation FD supplement (item 7.01) contains tables with:
      - Combined Ratio (loss + expense; typically 85–105 for PGR)
      - Policies in Force (total count; typically 10M–30M)
      - Net Premiums Written by segment (Agency, Direct, Commercial, Property)
      - Net Premiums Earned totals
      - Net Investment Income
      - Book Value per Share
      - EPS (basic)
      - Shares Repurchased and average cost
      - Investment book yield

    The ``gainshare_estimate``, ``pif_growth_yoy``, ``npw_growth_yoy``,
    ``channel_mix_agency_pct``, ``underwriting_income``, and
    ``unearned_premium_growth_yoy`` fields are left as ``None`` here; they are
    computed in ``_compute_derived_fields`` once the full sorted time series is
    available (YoY features) or on a per-row basis for ratio/product features.

    The filing date is used to derive ``month_end``: PGR files its monthly
    supplement in the first 3 weeks of the following month, so the data
    period is the month immediately prior to the filing date.

    Args:
        html: Raw HTML text of the 8-K exhibit.
        filing_date: ISO date string (``"YYYY-MM-DD"``).

    Returns:
        Dict with all parseable field values set, None placeholders for derived
        fields.  Returns ``None`` if neither combined_ratio nor pif_total can
        be extracted (filing is likely not a monthly supplement).
    """
    filed_dt = datetime.strptime(filing_date, "%Y-%m-%d")
    text = _strip_html_text(html)
    # Period: last day of the month before the filing month
    first_of_filing_month = filed_dt.replace(day=1)
    last_day_prior_month = first_of_filing_month - timedelta(days=1)
    month_end = last_day_prior_month.strftime("%Y-%m-%d")

    table_metrics: dict[str, Any] = {
        "avg_diluted_equivalent_shares": None,
        "avg_shares_basic": None,
        "avg_shares_diluted": None,
        "book_value_per_share": None,
        "combined_ratio": None,
        "common_shares_outstanding": None,
        "comprehensive_eps_diluted": None,
        "debt": None,
        "debt_to_total_capital": None,
        "eps_basic": None,
        "eps_diluted": None,
        "expense_ratio": None,
        "fees_and_other_revenues": None,
        "filing_date": filing_date,
        "filing_type": "monthly_results",
        "fixed_income_duration": None,
        "fte_return_common_stocks": None,
        "fte_return_fixed_income": None,
        "fte_return_total_portfolio": None,
        "income_before_income_taxes": None,
        "interest_expense": None,
        "investment_book_yield": None,
        "investment_income": None,
        "loss_lae_ratio": None,
        "loss_lae_reserves": None,
        "losses_lae": None,
        "net_income": None,
        "net_premiums_earned": None,
        "net_premiums_written": None,
        "net_unrealized_gains_fixed": None,
        "npe_agency": None,
        "npe_commercial": None,
        "npe_direct": None,
        "npe_property": None,
        "npw_agency": None,
        "npw_commercial": None,
        "npw_direct": None,
        "npw_property": None,
        "other_underwriting_expenses": None,
        "pif_agency_auto": None,
        "pif_commercial_lines": None,
        "pif_direct_auto": None,
        "pif_property": None,
        "pif_special_lines": None,
        "pif_total": None,
        "pif_total_personal_lines": None,
        "policy_acquisition_costs": None,
        "provision_for_income_taxes": None,
        "roe_comprehensive_trailing_12m": None,
        "roe_net_income_trailing_12m": None,
        "service_revenues": None,
        "shareholders_equity": None,
        "shares_repurchased": None,
        "total_assets": None,
        "total_comprehensive_income": None,
        "total_expenses": None,
        "total_investments": None,
        "total_liabilities": None,
        "total_net_realized_gains": None,
        "total_revenues": None,
        "unearned_premiums": None,
        "avg_cost_per_share": None,
        "weighted_avg_credit_quality": None,
    }

    tables = _read_exhibit_tables(html)
    for table in tables:
        for _, row in table.iterrows():
            tokens = _dedup_row(row)
            if not tokens:
                continue
            label = tokens[0].lower().replace("’", "'")
            nums = _extract_numerics(tokens[1:])

            if "average diluted equivalent common shares" in label and nums:
                table_metrics["avg_diluted_equivalent_shares"] = nums[0]
            elif "average common shares outstanding - basic" in label and nums:
                table_metrics["avg_shares_basic"] = nums[0]
            elif "total average equivalent common shares - diluted" in label and nums:
                table_metrics["avg_shares_diluted"] = nums[0]
            elif label.startswith("net premiums written") and len(nums) >= 6:
                candidate = _parse_segment_metrics(nums[-6:])
                candidate_total = candidate[-1]
                candidate_parts = [value for value in candidate[:-1] if value is not None]
                current_total = table_metrics["net_premiums_written"]
                # Some exhibits include both current-month and YTD segment tables.
                # Prefer the smaller current-month slice over the larger YTD totals.
                if candidate_parts and candidate_total >= max(candidate_parts) and (
                    current_total is None
                    or table_metrics["npw_agency"] is None
                    or candidate_total < current_total
                ):
                    (
                        table_metrics["npw_agency"],
                        table_metrics["npw_direct"],
                        table_metrics["npw_commercial"],
                        table_metrics["npw_property"],
                        table_metrics["net_premiums_written"],
                    ) = candidate
            elif label.startswith("net premiums earned") and len(nums) >= 6:
                candidate = _parse_segment_metrics(nums[-6:])
                candidate_total = candidate[-1]
                candidate_parts = [value for value in candidate[:-1] if value is not None]
                current_total = table_metrics["net_premiums_earned"]
                if candidate_parts and candidate_total >= max(candidate_parts) and (
                    current_total is None
                    or table_metrics["npe_agency"] is None
                    or candidate_total < current_total
                ):
                    (
                        table_metrics["npe_agency"],
                        table_metrics["npe_direct"],
                        table_metrics["npe_commercial"],
                        table_metrics["npe_property"],
                        table_metrics["net_premiums_earned"],
                    ) = candidate
            elif label.startswith("net premiums written") and nums and table_metrics["net_premiums_written"] is None:
                table_metrics["net_premiums_written"] = nums[0]
            elif label.startswith("net premiums earned") and nums and table_metrics["net_premiums_earned"] is None:
                table_metrics["net_premiums_earned"] = nums[0]
            elif label.startswith("net income") and "available to common" not in label and nums and table_metrics["net_income"] is None:
                table_metrics["net_income"] = nums[0]
            elif "per share available to common shareholders" in label and nums and table_metrics["eps_diluted"] is None:
                table_metrics["eps_diluted"] = nums[0]
            elif label == "basic" and nums and table_metrics["eps_basic"] is None:
                table_metrics["eps_basic"] = nums[0]
            elif label == "diluted" and nums:
                value = nums[0]
                if table_metrics["eps_diluted"] is None:
                    table_metrics["eps_diluted"] = value
                elif value != table_metrics["eps_diluted"]:
                    table_metrics["comprehensive_eps_diluted"] = value
            elif "combined ratio" in label and nums and table_metrics["combined_ratio"] is None:
                table_metrics["combined_ratio"] = nums[-1] if len(nums) >= 6 else nums[0]
            elif "loss/lae ratio" in label and nums:
                table_metrics["loss_lae_ratio"] = nums[-1] if len(nums) >= 6 else nums[0]
            elif "expense ratio" in label and "net catastrophe" not in label and nums:
                table_metrics["expense_ratio"] = nums[-1] if len(nums) >= 6 else nums[0]
            elif "agency" in label and "auto" in label and nums and table_metrics["pif_agency_auto"] is None:
                table_metrics["pif_agency_auto"] = nums[0]
            elif "direct" in label and "auto" in label and nums and table_metrics["pif_direct_auto"] is None:
                table_metrics["pif_direct_auto"] = nums[0]
            elif "special lines" in label and nums and table_metrics["pif_special_lines"] is None:
                table_metrics["pif_special_lines"] = nums[0]
            elif "property business" in label and nums and table_metrics["pif_property"] is None:
                table_metrics["pif_property"] = nums[0]
            elif "total personal lines" in label and nums and table_metrics["pif_total_personal_lines"] is None:
                table_metrics["pif_total_personal_lines"] = nums[0]
            elif "commercial lines" in label and nums and table_metrics["pif_commercial_lines"] is None:
                table_metrics["pif_commercial_lines"] = nums[0]
            elif label in ("total", "companywide", "companywide total") and nums:
                candidate = _normalise_pif_value(nums[0])
                if candidate >= 10_000:
                    table_metrics["pif_total"] = candidate
            elif label == "investment income" and nums and table_metrics["investment_income"] is None:
                table_metrics["investment_income"] = nums[0]
            elif "total net realized gains" in label and nums and table_metrics["total_net_realized_gains"] is None:
                table_metrics["total_net_realized_gains"] = nums[0]
            elif label == "service revenues" and nums and table_metrics["service_revenues"] is None:
                table_metrics["service_revenues"] = nums[0]
            elif "fees and other revenues" in label and nums and table_metrics["fees_and_other_revenues"] is None:
                table_metrics["fees_and_other_revenues"] = nums[0]
            elif label == "total revenues" and nums and table_metrics["total_revenues"] is None:
                table_metrics["total_revenues"] = nums[0]
            elif "losses and loss adjustment expenses" in label and nums and table_metrics["losses_lae"] is None:
                table_metrics["losses_lae"] = nums[0]
            elif "policy acquisition costs" in label and nums and table_metrics["policy_acquisition_costs"] is None:
                table_metrics["policy_acquisition_costs"] = nums[0]
            elif "other underwriting expenses" in label and nums and table_metrics["other_underwriting_expenses"] is None:
                table_metrics["other_underwriting_expenses"] = nums[0]
            elif label == "interest expense" and nums and table_metrics["interest_expense"] is None:
                table_metrics["interest_expense"] = nums[0]
            elif label == "total expenses" and nums and table_metrics["total_expenses"] is None:
                table_metrics["total_expenses"] = nums[0]
            elif "income before income taxes" in label and nums and table_metrics["income_before_income_taxes"] is None:
                table_metrics["income_before_income_taxes"] = nums[0]
            elif "provision for income taxes" in label and nums and table_metrics["provision_for_income_taxes"] is None:
                table_metrics["provision_for_income_taxes"] = nums[0]
            elif "total comprehensive income" in label and nums and table_metrics["total_comprehensive_income"] is None:
                table_metrics["total_comprehensive_income"] = nums[0]
            elif label == "total investments2" and nums and table_metrics["total_investments"] is None:
                table_metrics["total_investments"] = nums[0]
            elif label == "total assets" and nums and table_metrics["total_assets"] is None:
                table_metrics["total_assets"] = nums[0]
            elif "loss and loss adjustment expense reserves" in label and nums and table_metrics["loss_lae_reserves"] is None:
                table_metrics["loss_lae_reserves"] = nums[0]
            elif label == "unearned premiums" and nums and table_metrics["unearned_premiums"] is None:
                table_metrics["unearned_premiums"] = nums[0]
            elif label == "debt" and nums and table_metrics["debt"] is None:
                table_metrics["debt"] = nums[0]
            elif label == "total liabilities" and nums and table_metrics["total_liabilities"] is None:
                table_metrics["total_liabilities"] = nums[0]
            elif "shareholders' equity" in label or "shareholders’ equity" in label:
                if nums and table_metrics["shareholders_equity"] is None:
                    table_metrics["shareholders_equity"] = nums[0]
            elif "common shares outstanding" in label and nums and table_metrics["common_shares_outstanding"] is None:
                table_metrics["common_shares_outstanding"] = nums[0]
            elif "common shares repurchased" in label and nums and table_metrics["shares_repurchased"] is None:
                raw = nums[0]
                table_metrics["shares_repurchased"] = raw / 1_000_000.0 if raw >= 1_000 else raw
            elif "average cost per common share" in label and nums and table_metrics["avg_cost_per_share"] is None:
                table_metrics["avg_cost_per_share"] = nums[0]
            elif "book value per common share" in label and nums and table_metrics["book_value_per_share"] is None:
                table_metrics["book_value_per_share"] = nums[0]
            elif label == "net income" and nums and table_metrics["roe_net_income_trailing_12m"] is None:
                if tokens[-1].endswith("%") or "%" in tokens:
                    table_metrics["roe_net_income_trailing_12m"] = nums[0]
            elif "comprehensive income" == label and nums and table_metrics["roe_comprehensive_trailing_12m"] is None:
                if tokens[-1].endswith("%") or "%" in tokens:
                    table_metrics["roe_comprehensive_trailing_12m"] = nums[0]
            elif "net unrealized pretax gains" in label and nums and table_metrics["net_unrealized_gains_fixed"] is None:
                table_metrics["net_unrealized_gains_fixed"] = nums[0]
            elif "debt-to-total capital ratio" in label and nums and table_metrics["debt_to_total_capital"] is None:
                table_metrics["debt_to_total_capital"] = nums[0]
            elif "fixed-income portfolio duration" in label and nums and table_metrics["fixed_income_duration"] is None:
                table_metrics["fixed_income_duration"] = nums[0]
            elif "weighted average credit quality" in label and table_metrics["weighted_avg_credit_quality"] is None:
                table_metrics["weighted_avg_credit_quality"] = tokens[-1].replace(".", "").strip()
            elif label == "fixed-income securities" and nums and table_metrics["fte_return_fixed_income"] is None:
                table_metrics["fte_return_fixed_income"] = nums[0]
            elif label == "common stocks" and nums and table_metrics["fte_return_common_stocks"] is None:
                table_metrics["fte_return_common_stocks"] = nums[0]
            elif label == "total portfolio" and nums and table_metrics["fte_return_total_portfolio"] is None:
                table_metrics["fte_return_total_portfolio"] = nums[0]
            elif "pretax annualized investment income book yield" in label and nums and table_metrics["investment_book_yield"] is None:
                table_metrics["investment_book_yield"] = nums[0] / 100.0

    # -----------------------------------------------------------------------
    # Combined Ratio (always present; required for usability check)
    # -----------------------------------------------------------------------
    combined_ratio = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)combined\s+ratio[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*([\d]+\.[\d]+)",
            r"(?i)combined\s+ratio\b[^<]{0,80}?([\d]{2,3}\.[\d]{1,2})",
            r"(?i)\bCR\b[^<]{0,40}</t[dh]>\s*<t[dh][^>]*>\s*([\d]+\.[\d]+)",
            r"(?si)combined.{0,300}?(\b\d{2,3}\.\d{1,2}\b)",
        ],
        lo=60.0, hi=140.0,
    )

    # -----------------------------------------------------------------------
    # Policies in Force — total (always present in PGR supplements)
    # -----------------------------------------------------------------------
    pif_total = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)policies\s+in\s+force[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*([\d,]+)",
            r"(?i)total\s+pif\b[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*([\d,]+)",
            r"(?i)policies\s+in\s+force\b[^<]{0,80}?([\d,]{6,})",
            r"(?i)\bpif\b[^<]{0,40}</t[dh]>\s*<t[dh][^>]*>\s*([\d,]+)",
            r"(?si)policies\s+in\s+force.{0,300}?(\b[\d,]{6,}\b)",
        ],
        lo=10_000, hi=30_000_000,
    )
    pif_total = _normalise_pif_value(pif_total)

    if combined_ratio is None and pif_total is None:
        return None  # Not a monthly supplement — skip

    # -----------------------------------------------------------------------
    # P2.6 — Segment Net Premiums Written (in $millions)
    # PGR reports NPW in three segments: Agency (Personal Lines Auto),
    # Direct (Personal Lines Auto), Commercial Lines, and Property.
    # The monthly supplement typically has a "Net Premiums Written" table
    # with one row per segment.  Dollar amounts are in millions.
    # -----------------------------------------------------------------------

    # Agency NPW
    npw_agency = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)agency[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d*)",
            r"(?i)personal\s+lines\s+(?:auto\s+)?agency[^<]{0,80}([\d,]+\.?\d*)",
            r"(?si)agency.{0,200}?net\s+premiums?\s+written.{0,200}?([\d,]+\.?\d*)",
        ],
        lo=100.0, hi=20_000.0,   # $100M–$20B realistic range (figures in $M)
        scale=1.0,
    )

    # Direct NPW
    npw_direct = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)direct[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d*)",
            r"(?i)personal\s+lines\s+(?:auto\s+)?direct[^<]{0,80}([\d,]+\.?\d*)",
            r"(?si)direct.{0,200}?net\s+premiums?\s+written.{0,200}?([\d,]+\.?\d*)",
        ],
        lo=100.0, hi=20_000.0,
        scale=1.0,
    )

    # Commercial Lines NPW
    npw_commercial = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)commercial\s+lines?[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d*)",
            r"(?i)commercial\s+auto[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d*)",
            r"(?si)commercial\s+lines?.{0,200}?net\s+premiums?\s+written.{0,200}?([\d,]+\.?\d*)",
        ],
        lo=10.0, hi=5_000.0,
        scale=1.0,
    )

    # Property NPW
    npw_property = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)property[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d*)",
            r"(?si)property.{0,200}?net\s+premiums?\s+written.{0,200}?([\d,]+\.?\d*)",
        ],
        lo=1.0, hi=3_000.0,
        scale=1.0,
    )

    # Total company NPW
    net_premiums_written = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)(?:total\s+)?(?:company\s+)?net\s+premiums?\s+written[^<]{0,80}\$?\s*([\d,]+\.?\d*)",
            r"(?i)net\s+premiums?\s+written[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d*)",
            r"(?si)net\s+premiums?\s+written.{0,300}?(\b[\d,]{3,}\.\d\b)",
        ],
        lo=500.0, hi=30_000.0,
        scale=1.0,
    )

    # Net Premiums Earned — total
    net_premiums_earned = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)net\s+premiums?\s+earned[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d*)",
            r"(?i)(?:total\s+)?net\s+premiums?\s+earned[^<]{0,80}\$?\s*([\d,]+\.?\d*)",
            r"(?si)net\s+premiums?\s+earned.{0,300}?(\b[\d,]{3,}\.\d\b)",
        ],
        lo=500.0, hi=30_000.0,
        scale=1.0,
    )

    # -----------------------------------------------------------------------
    # Net Investment Income ($M)
    # -----------------------------------------------------------------------
    investment_income = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)net\s+investment\s+income[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d*)",
            r"(?i)investment\s+income[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d*)",
            r"(?si)investment\s+income.{0,200}?([\d,]+\.\d)",
        ],
        lo=5.0, hi=3_000.0,
        scale=1.0,
    )

    # -----------------------------------------------------------------------
    # Book Value per Share ($)
    # -----------------------------------------------------------------------
    book_value_per_share = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)book\s+value\s+per\s+(?:common\s+)?share[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d+)",
            r"(?i)book\s+value\s+per\s+share[^<]{0,80}\$\s*([\d,]+\.\d{2})",
            r"(?si)book\s+value\s+per\s+share.{0,200}?\$\s*([\d,]+\.\d{2})",
        ],
        lo=5.0, hi=500.0,
        scale=1.0,
    )

    # -----------------------------------------------------------------------
    # EPS — basic ($ per share, monthly)
    # -----------------------------------------------------------------------
    eps_basic = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)(?:basic\s+)?earnings\s+per\s+(?:common\s+)?share[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.?\d+)",
            r"(?i)\beps\b[^<]{0,40}basic[^<]{0,60}\$?\s*([\d]+\.\d{2})",
            r"(?si)basic.{0,100}?earnings\s+per\s+share.{0,200}?\$\s*([\d]+\.\d{2})",
        ],
        lo=0.0, hi=20.0,
        scale=1.0,
    )

    # -----------------------------------------------------------------------
    # Shares Repurchased (stored as millions of shares) and Average Cost per
    # Share ($).  PGR's exhibit format changes in 2023-08 from decimal
    # "millions of shares" (e.g. 0.21) to whole-share counts (e.g. 46,822).
    # Normalize both formats into millions of shares for downstream features.
    # -----------------------------------------------------------------------
    whole_share_count = _try_parse_last_text_match(
        text,
        patterns=[
            r"(?i)common\s+shares\s+repurchased(?:\s*-\s*[a-z0-9 ]+)?\s+([\d,]{4,})",
            r"(?i)shares\s+repurchased(?:\s*-\s*[a-z0-9 ]+)?\s+([\d,]{4,})",
        ],
        lo=1_000.0,
        hi=50_000_000.0,
        scale=1.0,
    )
    if whole_share_count is not None:
        shares_repurchased = whole_share_count / 1_000_000.0
    else:
        shares_repurchased = _try_parse_last_text_match(
            text,
            patterns=[
                r"(?i)common\s+shares\s+repurchased(?:\s*-\s*[a-z0-9 ]+)?\s+([\d,]+(?:\.\d+)?)",
                r"(?i)shares\s+repurchased(?:\s*-\s*[a-z0-9 ]+)?\s+([\d,]+(?:\.\d+)?)",
            ],
            lo=0.0,
            hi=50.0,
            scale=1.0,
        )
    if shares_repurchased is None:
        shares_repurchased = _try_parse_dollar(
            html,
            patterns=[
                r"(?i)shares?\s+repurchased[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*([\d,]+\.?\d*)",
                r"(?i)(?:common\s+)?shares?\s+(?:re)?purchased[^<]{0,60}([\d,]+\.?\d*)",
                r"(?si)repurchase.{0,200}?shares?.{0,100}?([\d,]+\.\d)",
            ],
            lo=0.0, hi=50.0,  # millions of shares
            scale=1.0,
        )

    avg_cost_per_share = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)average\s+(?:purchase\s+)?(?:cost|price)\s+per\s+share[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.\d{2})",
            r"(?i)avg\.\s+(?:cost|price)\s+per\s+share[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*\$?\s*([\d,]+\.\d{2})",
            r"(?si)average.{0,50}(?:cost|price)\s+per\s+share.{0,200}?\$\s*([\d,]+\.\d{2})",
        ],
        lo=10.0, hi=600.0,
        scale=1.0,
    )
    if avg_cost_per_share is None:
        avg_cost_per_share = _try_parse_last_text_match(
            text,
            patterns=[
                r"(?i)average\s+cost\s+per\s+(?:common\s+)?share\s+\$?\s*([\d,]+\.\d{2})",
                r"(?i)average\s+price\s+per\s+(?:common\s+)?share\s+\$?\s*([\d,]+\.\d{2})",
                r"(?i)avg\.\s+(?:cost|price)\s+per\s+(?:common\s+)?share\s+\$?\s*([\d,]+\.\d{2})",
            ],
            lo=10.0,
            hi=600.0,
            scale=1.0,
        )

    # -----------------------------------------------------------------------
    # Investment Book Yield (as %)
    # -----------------------------------------------------------------------
    investment_book_yield = _try_parse_dollar(
        html,
        patterns=[
            r"(?i)(?:investment\s+)?book\s+yield[^<]{0,60}</t[dh]>\s*<t[dh][^>]*>\s*([\d]+\.[\d]+)\s*%?",
            r"(?i)book\s+yield[^<]{0,80}([\d]+\.[\d]+)\s*%",
        ],
        lo=0.5, hi=15.0,  # percentage points
        scale=0.01,        # convert from % to decimal
    )

    return {
        "month_end": month_end,
        "filing_date": filing_date,
        "filing_type": table_metrics["filing_type"],
        # Core
        "combined_ratio": table_metrics["combined_ratio"] if table_metrics["combined_ratio"] is not None else combined_ratio,
        "pif_total": table_metrics["pif_total"] if table_metrics["pif_total"] is not None else pif_total,
        # v6.2 phase 1 fields parsed from HTML (P2.6)
        "net_premiums_written": table_metrics["net_premiums_written"] if table_metrics["net_premiums_written"] is not None else net_premiums_written,
        "net_premiums_earned": table_metrics["net_premiums_earned"] if table_metrics["net_premiums_earned"] is not None else net_premiums_earned,
        "net_income": table_metrics["net_income"],
        "eps_diluted": table_metrics["eps_diluted"],
        "avg_diluted_equivalent_shares": table_metrics["avg_diluted_equivalent_shares"],
        "investment_income": table_metrics["investment_income"] if table_metrics["investment_income"] is not None else investment_income,
        "total_net_realized_gains": table_metrics["total_net_realized_gains"],
        "service_revenues": table_metrics["service_revenues"],
        "fees_and_other_revenues": table_metrics["fees_and_other_revenues"],
        "total_revenues": table_metrics["total_revenues"],
        "losses_lae": table_metrics["losses_lae"],
        "policy_acquisition_costs": table_metrics["policy_acquisition_costs"],
        "other_underwriting_expenses": table_metrics["other_underwriting_expenses"],
        "interest_expense": table_metrics["interest_expense"],
        "total_expenses": table_metrics["total_expenses"],
        "income_before_income_taxes": table_metrics["income_before_income_taxes"],
        "provision_for_income_taxes": table_metrics["provision_for_income_taxes"],
        "total_comprehensive_income": table_metrics["total_comprehensive_income"],
        "eps_basic": table_metrics["eps_basic"] if table_metrics["eps_basic"] is not None else eps_basic,
        "comprehensive_eps_diluted": table_metrics["comprehensive_eps_diluted"],
        "avg_shares_basic": table_metrics["avg_shares_basic"],
        "avg_shares_diluted": table_metrics["avg_shares_diluted"],
        "loss_lae_ratio": table_metrics["loss_lae_ratio"],
        "expense_ratio": table_metrics["expense_ratio"],
        "pif_agency_auto": table_metrics["pif_agency_auto"],
        "pif_direct_auto": table_metrics["pif_direct_auto"],
        "pif_special_lines": table_metrics["pif_special_lines"],
        "pif_property": table_metrics["pif_property"],
        "pif_total_personal_lines": table_metrics["pif_total_personal_lines"],
        "pif_commercial_lines": table_metrics["pif_commercial_lines"],
        "npw_agency": table_metrics["npw_agency"] if table_metrics["npw_agency"] is not None else npw_agency,
        "npw_direct": table_metrics["npw_direct"] if table_metrics["npw_direct"] is not None else npw_direct,
        "npw_property": table_metrics["npw_property"] if table_metrics["npw_property"] is not None else npw_property,
        "npw_commercial": table_metrics["npw_commercial"] if table_metrics["npw_commercial"] is not None else npw_commercial,
        "npe_agency": table_metrics["npe_agency"],
        "npe_direct": table_metrics["npe_direct"],
        "npe_property": table_metrics["npe_property"],
        "npe_commercial": table_metrics["npe_commercial"],
        "total_investments": table_metrics["total_investments"],
        "total_assets": table_metrics["total_assets"],
        "loss_lae_reserves": table_metrics["loss_lae_reserves"],
        "unearned_premiums": table_metrics["unearned_premiums"],
        "debt": table_metrics["debt"],
        "total_liabilities": table_metrics["total_liabilities"],
        "shareholders_equity": table_metrics["shareholders_equity"],
        "common_shares_outstanding": table_metrics["common_shares_outstanding"],
        "shares_repurchased": table_metrics["shares_repurchased"] if table_metrics["shares_repurchased"] is not None else shares_repurchased,
        "avg_cost_per_share": table_metrics["avg_cost_per_share"] if table_metrics["avg_cost_per_share"] is not None else avg_cost_per_share,
        "book_value_per_share": table_metrics["book_value_per_share"] if table_metrics["book_value_per_share"] is not None else book_value_per_share,
        "roe_net_income_trailing_12m": table_metrics["roe_net_income_trailing_12m"],
        "roe_comprehensive_trailing_12m": table_metrics["roe_comprehensive_trailing_12m"],
        "debt_to_total_capital": table_metrics["debt_to_total_capital"],
        "fixed_income_duration": table_metrics["fixed_income_duration"],
        "fte_return_fixed_income": table_metrics["fte_return_fixed_income"],
        "fte_return_common_stocks": table_metrics["fte_return_common_stocks"],
        "fte_return_total_portfolio": table_metrics["fte_return_total_portfolio"],
        "investment_book_yield": table_metrics["investment_book_yield"] if table_metrics["investment_book_yield"] is not None else investment_book_yield,
        "net_unrealized_gains_fixed": table_metrics["net_unrealized_gains_fixed"],
        "weighted_avg_credit_quality": table_metrics["weighted_avg_credit_quality"],
        # Derived fields — populated by _compute_derived_fields after time-series is assembled
        "pif_growth_yoy": None,
        "gainshare_estimate": None,
        "channel_mix_agency_pct": None,
        "underwriting_income": None,
        "npw_growth_yoy": None,
        "unearned_premium_growth_yoy": None,
        "buyback_yield": None,
    }


# ---------------------------------------------------------------------------
# Derived field computation
# ---------------------------------------------------------------------------

def _prior_year_key(month_end: str) -> str:
    """Return the month_end string for the same month one year prior.

    Handles the Feb-29 edge case: if month_end is 2024-02-29 (leap year),
    the prior-year key is 2023-02-28.
    """
    dt = datetime.strptime(month_end, "%Y-%m-%d")
    try:
        prior = dt.replace(year=dt.year - 1)
    except ValueError:
        # Feb 29 in a leap year → Feb 28 in a non-leap year
        prior = dt.replace(year=dt.year - 1, day=28)
    return prior.strftime("%Y-%m-%d")


def _compute_derived_fields(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute all derived fields for the full sorted 8-K time series.

    These fields require a full sorted time series so that year-ago values can
    be looked up.  Must be called after all records are collected and sorted
    ascending by ``month_end``.

    Derived fields computed:
      - ``pif_growth_yoy``          — YoY PIF growth (% change vs same month prior year)
      - ``gainshare_estimate``       — PGR Gainshare multiplier estimate (0–2 scale)
      - ``channel_mix_agency_pct``   — Agency NPW / (Agency + Direct NPW)
      - ``underwriting_income``      — NPE × (1 − CR/100)  (core P&C profitability $)
      - ``npw_growth_yoy``           — YoY % change in total Net Premiums Written
      - ``unearned_premium_growth_yoy`` — YoY % change in unearned premium reserve
        NOTE: unearned_premiums is rarely in the monthly supplement HTML; this will
        typically remain None for the live-fetch path (available via CSV backfill).

    The gainshare formula mirrors ``pgr_monthly_loader.py``:
      - ``cr_score``    = clip((96 − CR) / 10,  0, 2)
      - ``pif_score``   = clip(pif_growth / 0.10, 0, 2)
      - ``gainshare``   = 0.5 × cr_score + 0.5 × pif_score

    Args:
        records: List of record dicts, already sorted ascending by ``month_end``.

    Returns:
        The same list with all computable derived fields filled in.
    """
    # Build lookup dicts for all fields that feed YoY computations
    pif_by_month: dict[str, float] = {
        r["month_end"]: r["pif_total"]
        for r in records
        if r.get("pif_total") is not None
    }
    npw_by_month: dict[str, float] = {
        r["month_end"]: r["net_premiums_written"]
        for r in records
        if r.get("net_premiums_written") is not None
    }
    unprem_by_month: dict[str, float] = {
        r["month_end"]: r["unearned_premiums"]
        for r in records
        if r.get("unearned_premiums") is not None
    }

    for rec in records:
        me_key = rec["month_end"]
        prior_key = _prior_year_key(me_key)

        # --- YoY PIF growth ---
        pif_cur = rec.get("pif_total")
        if pif_cur is not None:
            prior_pif = pif_by_month.get(prior_key)
            if prior_pif and prior_pif != 0.0:
                rec["pif_growth_yoy"] = (pif_cur - prior_pif) / prior_pif

        # --- YoY NPW growth ---
        npw_cur = rec.get("net_premiums_written")
        if npw_cur is not None:
            prior_npw = npw_by_month.get(prior_key)
            if prior_npw and prior_npw != 0.0:
                rec["npw_growth_yoy"] = (npw_cur - prior_npw) / prior_npw

        # --- YoY unearned premium growth ---
        unprem_cur = rec.get("unearned_premiums")
        if unprem_cur is not None:
            prior_unprem = unprem_by_month.get(prior_key)
            if prior_unprem and prior_unprem != 0.0:
                rec["unearned_premium_growth_yoy"] = (
                    (unprem_cur - prior_unprem) / prior_unprem
                )

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

        # --- Per-row derived fields (no time series needed) ---

        # channel_mix_agency_pct = npw_agency / (npw_agency + npw_direct)
        npw_ag = rec.get("npw_agency")
        npw_di = rec.get("npw_direct")
        if npw_ag is not None and npw_di is not None:
            denom = npw_ag + npw_di
            if denom > 0.0:
                rec["channel_mix_agency_pct"] = npw_ag / denom

        # underwriting_income = net_premiums_earned × (1 − CR / 100)
        npe = rec.get("net_premiums_earned")
        if npe is not None and cr is not None:
            rec["underwriting_income"] = npe * (1.0 - cr / 100.0)

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
# v7.2 — Parsed record cross-validator
# ---------------------------------------------------------------------------

def _validate_parsed_record(
    record: dict[str, Any],
    filing_date: str,
    accession: str,
) -> dict[str, Any]:
    """Cross-validate parsed 8-K fields for internal consistency.

    Checks:
      1. combined_ratio ≈ loss_lae_ratio + expense_ratio (within 5pp).
         If both sub-ratios are present and the sum deviates > 5pp from CR,
         log a WARNING and set combined_ratio = None (prefer missing over wrong).
      2. net_premiums_written >= sum of segment NPW (agency + direct +
         commercial + property).  If total < sum of parts, log WARNING.
      3. pif_total is stored in thousands of policies for PGR monthly data.
         If parsed pif_total < 10,000, likely a mis-parse; set to None.
      4. eps_basic should be in range [-5.0, 15.0] for monthly figures.
         Out-of-range values are set to None.

    Args:
        record:       Parsed record dict from _parse_html_exhibit().
        filing_date:  ISO date string for logging context.
        accession:    Accession number for logging context.

    Returns:
        The record dict, possibly with some fields set to None.
    """
    cr = record.get("combined_ratio")
    lr = record.get("loss_lae_ratio")
    er = record.get("expense_ratio")

    if cr is not None and lr is not None and er is not None:
        expected_cr = lr + er
        if abs(cr - expected_cr) > 5.0:
            log.warning(
                "VALIDATION: CR=%.1f but loss_ratio+expense_ratio=%.1f+%.1f=%.1f "
                "(delta=%.1f) in %s (filed %s). Setting CR=None.",
                cr, lr, er, expected_cr, abs(cr - expected_cr),
                accession, filing_date,
            )
            record["combined_ratio"] = None

    # NPW segment check
    npw_total = record.get("net_premiums_written")
    npw_parts = sum(
        record.get(k) or 0.0
        for k in ("npw_agency", "npw_direct", "npw_commercial", "npw_property")
    )
    if npw_total is not None and npw_parts > 0 and npw_total < npw_parts * 0.9:
        log.warning(
            "VALIDATION: NPW_total=%.1f < sum_of_segments=%.1f in %s (filed %s).",
            npw_total, npw_parts, accession, filing_date,
        )

    # PIF floor
    pif = record.get("pif_total")
    if pif is not None and pif < 10_000:
        log.warning(
            "VALIDATION: pif_total=%.0f < 10,000 floor in %s. Setting None.",
            pif, accession,
        )
        record["pif_total"] = None

    # EPS range
    eps = record.get("eps_basic")
    if eps is not None and (eps < -5.0 or eps > 15.0):
        log.warning(
            "VALIDATION: eps_basic=%.2f out of [-5, 15] range in %s. Setting None.",
            eps, accession,
        )
        record["eps_basic"] = None

    return record


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
        "Backfill window: %s -> %s  (backfill_years=%d)",
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

            # v7.2: cross-validate parsed fields; nullify inconsistent ones.
            parsed = _validate_parsed_record(parsed, filing_date, accession)
            parsed["accession_number"] = accession
            # If validation nullified both core fields, skip this filing.
            if parsed["combined_ratio"] is None and parsed["pif_total"] is None:
                log.debug(
                    "Validation nullified both CR and PIF for %s — skipping.",
                    accession,
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
            log.exception(
                "SKIP %s (filed %s) due to parse failure. Error=%r",
                accession,
                filing_date,
                exc,
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

    # v7.2: prefer the filing with the most non-null fields when deduplicating.
    def _completeness(rec: dict[str, Any]) -> int:
        """Count non-None fields (excluding month_end)."""
        return sum(1 for k, v in rec.items() if v is not None and k != "month_end")

    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        me = rec["month_end"]
        if me not in seen or _completeness(rec) >= _completeness(seen[me]):
            seen[me] = rec
    deduped = sorted(seen.values(), key=lambda r: r["month_end"])

    # Coverage report
    n_total = len(deduped)

    def _cov(field: str) -> str:
        n = sum(1 for r in deduped if r.get(field) is not None)
        return f"{n}/{n_total}"

    log.info(
        "Coverage  combined_ratio=%s  pif_total=%s  gainshare=%s  "
        "npw=%s  npw_agency=%s  investment_income=%s  bvps=%s  "
        "channel_mix=%s  underwriting_income=%s  "
        "date_range=%s->%s",
        _cov("combined_ratio"),
        _cov("pif_total"),
        _cov("gainshare_estimate"),
        _cov("net_premiums_written"),
        _cov("npw_agency"),
        _cov("investment_income"),
        _cov("book_value_per_share"),
        _cov("channel_mix_agency_pct"),
        _cov("underwriting_income"),
        deduped[0]["month_end"],
        deduped[-1]["month_end"],
    )

    if dry_run:
        log.info("Dry run — skipping DB write (%d rows would be upserted).", n_total)
        return 0

    n = db_client.upsert_pgr_edgar_monthly(conn, deduped)
    log.info("Upserted %d rows to pgr_edgar_monthly.", n)

    # v7.2: warn when no new months were added to alert on format changes.
    existing_months_row = conn.execute(
        "SELECT COUNT(DISTINCT month_end) FROM pgr_edgar_monthly"
    ).fetchone()
    existing_count = existing_months_row[0] if existing_months_row else 0

    if n == 0 or (existing_count > 0 and n <= existing_count):
        log.warning(
            "NOTE: No new months added this run (upserted %d rows into "
            "a table with %d existing months). If this persists, check "
            "whether PGR has changed its 8-K filing format.",
            n, existing_count,
        )

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
        "filing_date":                  "filing_date",
        "filing_type":                  "filing_type",
        "accession_number":             "accession_number",
        "combined_ratio":              "combined_ratio",
        "pif_total":                   "pif_total",
        "net_premiums_written":        "net_premiums_written",
        "net_premiums_earned":         "net_premiums_earned",
        "net_income":                  "net_income",
        "eps_diluted":                 "eps_diluted",
        "eps_basic":                   "eps_basic",
        "avg_diluted_equivalent_shares": "avg_diluted_equivalent_shares",
        "total_net_realized_gains":    "total_net_realized_gains",
        "service_revenues":            "service_revenues",
        "fees_and_other_revenues":     "fees_and_other_revenues",
        "losses_lae":                  "losses_lae",
        "policy_acquisition_costs":    "policy_acquisition_costs",
        "other_underwriting_expenses": "other_underwriting_expenses",
        "interest_expense":            "interest_expense",
        "provision_for_income_taxes":  "provision_for_income_taxes",
        "total_comprehensive_income":  "total_comprehensive_income",
        "comprehensive_eps_diluted":   "comprehensive_eps_diluted",
        "avg_shares_basic":            "avg_shares_basic",
        "avg_shares_diluted":          "avg_shares_diluted",
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
        "pif_special_lines":           "pif_special_lines",
        "pif_property":                "pif_property",
        "pif_commercial_lines":        "pif_commercial_lines",
        "pif_total_personal_lines":    "pif_total_personal_lines",
        # Company-level operating metrics
        "investment_income":           "investment_income",
        "total_revenues":              "total_revenues",
        "total_expenses":              "total_expenses",
        "income_before_income_taxes":  "income_before_income_taxes",
        "roe_net_income_trailing_12m": "roe_net_income_ttm",  # CSV name differs
        "roe_comprehensive_trailing_12m": "roe_comprehensive_trailing_12m",
        "shareholders_equity":         "shareholders_equity",
        "total_assets":                "total_assets",
        "total_investments":           "total_investments",
        "loss_lae_reserves":           "loss_lae_reserves",
        "unearned_premiums":           "unearned_premiums",
        "debt":                        "debt",
        "total_liabilities":           "total_liabilities",
        "common_shares_outstanding":   "common_shares_outstanding",
        "shares_repurchased":          "shares_repurchased",
        "avg_cost_per_share":          "avg_cost_per_share",
        # Investment portfolio metrics
        "fte_return_fixed_income":     "fte_return_fixed_income",
        "fte_return_common_stocks":    "fte_return_common_stocks",
        "fte_return_total_portfolio":  "fte_return_total_portfolio",
        "investment_book_yield":       "investment_book_yield",
        "net_unrealized_gains_fixed":  "net_unrealized_gains_fixed",
        "fixed_income_duration":       "fixed_income_duration",
        "debt_to_total_capital":       "debt_to_total_capital",
        "weighted_avg_credit_quality": "weighted_avg_credit_quality",
    }

    text_cols = {
        "filing_date",
        "filing_type",
        "accession_number",
        "weighted_avg_credit_quality",
    }
    for csv_col, db_col in DIRECT_MAP.items():
        if csv_col in df.columns and db_col not in text_cols:
            df[db_col] = pd.to_numeric(df[csv_col], errors="coerce")
        elif csv_col in df.columns:
            df[db_col] = df[csv_col].astype(str)
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
        "CSV loaded: %d rows  date_range=%s->%s",
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
    configure_logging()
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
