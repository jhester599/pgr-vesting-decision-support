"""
PGR monthly investor 8-K fetcher — SEC EDGAR submissions API.

PGR files a monthly investor report as an 8-K (Regulation FD) around the
17th–19th of each month.  These filings are identified by:

  * ``form``  == "8-K"
  * ``items`` contains "7.01"   (Reg FD Disclosure)

Each filing carries an Exhibit 99 HTML document with the full month's
operating results.  This module:

1. Hits ``data.sec.gov/submissions/CIK0000080661.json`` to enumerate filings.
2. For each monthly 8-K, fetches the filing index to locate the EX-99 HTML.
3. Parses the exhibit with ``pandas.read_html`` to extract:
   - ``report_period``         month-end date for the reporting period
   - ``combined_ratio``        GAAP combined ratio (loss + expense)
   - ``pif_total``             total policies in force (thousands)
   - ``net_premiums_written``  monthly NPW (millions)
   - ``net_premiums_earned``   monthly NPE (millions)
   - ``net_income``            monthly net income (millions)
   - ``eps_diluted``           diluted EPS for the month
   - ``loss_ratio``            companywide GAAP loss/LAE ratio
   - ``expense_ratio``         companywide GAAP expense ratio
4. Derives ``pif_growth_yoy`` and ``gainshare_estimate`` over the full series.
5. Writes to ``pgr_edgar_monthly`` via ``db_client.upsert_pgr_edgar_monthly``.

Rate-limit policy (SEC EDGAR fair-use):
  - Maximum 10 requests/second; this module sleeps 0.15 s between requests.
  - ``User-Agent`` header required: name + contact email.
"""

from __future__ import annotations

import io
import logging
import re
import time
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PGR_CIK: str = "0000080661"
_SUBMISSIONS_URL: str = (
    f"https://data.sec.gov/submissions/CIK{_PGR_CIK}.json"
)
_EDGAR_ARCHIVE: str = "https://www.sec.gov/Archives/edgar/data/80661"
_INDEX_SUFFIX: str = "-index.html"

# Polite request delay (seconds) to stay well under the 10 req/s EDGAR limit.
_REQUEST_DELAY: float = 0.15

# Gainshare calibration constants (from PGR proxy disclosures).
_CR_TARGET: float = 96.0          # Below this = positive CR contribution
_CR_SCALE: float = 10.0           # 10 pts below target = max score
_PIF_GROWTH_MAX: float = 0.10     # 10 % YoY PIF growth = max PIF score


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(session: requests.Session, url: str, host: str = "www.sec.gov") -> str:
    """Fetch ``url``, honour rate limit, raise on HTTP error."""
    if "data.sec.gov" in url:
        headers = config.build_edgar_headers("data.sec.gov")
    else:
        headers = config.build_edgar_headers(host)

    time.sleep(_REQUEST_DELAY)
    resp = session.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Step 1: enumerate monthly 8-K filings from submissions JSON
# ---------------------------------------------------------------------------

def fetch_submissions(session: requests.Session) -> list[dict[str, Any]]:
    """
    Return all PGR 8-K filings whose ``items`` field contains "7.01".

    These are the monthly investor-report 8-Ks (Regulation FD).

    Returns:
        List of dicts with keys: ``accession``, ``filing_date``.
        Sorted descending by filing date (most recent first).
    """
    headers = config.build_edgar_headers("data.sec.gov")
    time.sleep(_REQUEST_DELAY)
    resp = session.get(_SUBMISSIONS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    forms = recent["form"]
    items_list = recent["items"]
    filing_dates = recent["filingDate"]
    accessions = recent["accessionNumber"]

    results: list[dict[str, Any]] = []
    for form, items, fdate, accession in zip(
        forms, items_list, filing_dates, accessions
    ):
        if form == "8-K" and "7.01" in items:
            results.append(
                {
                    "accession": accession,
                    "filing_date": fdate,
                }
            )

    # Also check older filings pages if present.
    # The page['name'] is a bare filename like 'CIK0000080661-submissions-001.json';
    # it lives under the /submissions/ path on data.sec.gov.
    for page in data["filings"].get("files", []):
        name = page["name"]
        if not name.startswith("/"):
            name = "/submissions/" + name
        page_url = f"https://data.sec.gov{name}"
        try:
            time.sleep(_REQUEST_DELAY)
            page_resp = session.get(
                page_url,
                headers=config.build_edgar_headers("data.sec.gov"),
                timeout=30,
            )
            page_resp.raise_for_status()
            page_data = page_resp.json()
            for form, items, fdate, accession in zip(
                page_data.get("form", []),
                page_data.get("items", []),
                page_data.get("filingDate", []),
                page_data.get("accessionNumber", []),
            ):
                if form == "8-K" and "7.01" in items:
                    results.append(
                        {
                            "accession": accession,
                            "filing_date": fdate,
                        }
                    )
        except Exception as exc:
            logger.warning("Could not fetch older filings page %s: %s", page_url, exc, exc_info=True)

    results.sort(key=lambda r: r["filing_date"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Step 2: locate the EX-99 exhibit URL from the filing index
# ---------------------------------------------------------------------------

def _accession_to_folder(accession: str) -> str:
    """Convert '0000080661-26-000096' → '000008066126000096'."""
    return accession.replace("-", "")


def get_exhibit_url(
    session: requests.Session, accession: str
) -> str | None:
    """
    Fetch the filing index HTML and return the URL of the first EX-99 document.

    Returns ``None`` if no EX-99 is found (unexpected filing layout).
    """
    folder = _accession_to_folder(accession)
    index_url = f"{_EDGAR_ARCHIVE}/{folder}/{accession}{_INDEX_SUFFIX}"

    try:
        html = _get(session, index_url)
    except requests.HTTPError as exc:
        logger.warning("Could not fetch filing index for %s: %s", accession, exc)
        return None

    # Each document row looks like:
    # <tr>... seq | description | filename | type | size ...</tr>
    # We want the row where type = EX-99 (or EX-99.1, EX-99.2, etc.)
    rows = re.findall(r"<tr[^>]*>.*?</tr>", html, re.S | re.I)
    for row in rows:
        if re.search(r"EX-99", row, re.I):
            href = re.search(r'href="([^"]*\.htm[^"]*)"', row, re.I)
            if href:
                path = href.group(1)
                if path.startswith("/"):
                    return f"https://www.sec.gov{path}"
                return f"{_EDGAR_ARCHIVE}/{folder}/{path}"

    logger.warning("No EX-99 document found in filing index for %s", accession)
    return None


# ---------------------------------------------------------------------------
# Step 3: parse the exhibit HTML
# ---------------------------------------------------------------------------

def _dedup_row(row: pd.Series) -> list[str]:
    """
    Collapse duplicate consecutive values (EDGAR HTML artefact) into a unique list.

    Returns a list of non-NaN string tokens, with consecutive duplicates removed.
    """
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
    """Parse a number string, handling parentheses for negatives.

    Returns ``None`` for non-numeric strings (including ``'nan'``, ``'NaN'``).
    """
    cleaned = val.replace(",", "").replace("$", "").strip()
    if cleaned.lower() in ("nan", "none", "n/a", ""):
        return None
    cleaned = cleaned.replace("(", "-").replace(")", "").strip()
    try:
        result = float(cleaned)
        if result != result:  # float NaN check
            return None
        return result
    except ValueError:
        return None


def _extract_numerics(tokens: list[str]) -> list[float]:
    """Return all parseable float values from a token list (skip labels)."""
    nums: list[float] = []
    for t in tokens:
        v = _try_float(t)
        if v is not None:
            nums.append(v)
    return nums


def _extract_report_period(html: str) -> date | None:
    """
    Parse the 'For the month ended <Month> <Day>, <Year>' string from HTML.

    Returns the calendar month-end as a ``date``, or ``None`` on failure.
    """
    text = re.sub(r"&[a-z0-9#]+;", " ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    pattern = r"month\s+ended\s+(\w+)\s+(\d{1,2}),?\s+(\d{4})"
    m = re.search(pattern, text, re.I)
    if not m:
        # Some older filings say "ended Month Day, Year" without 'month'
        pattern2 = r"(\w+)\s+(\d{1,2}),\s+(\d{4})"
        m = re.search(pattern2, text, re.I)
        if not m:
            return None

    month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
    try:
        dt = datetime.strptime(
            f"{month_str} {int(day_str):02d} {year_str}", "%B %d %Y"
        )
        return dt.date()
    except ValueError:
        return None


def parse_exhibit(html: str) -> dict[str, Any]:
    """
    Parse an 8-K Exhibit 99 HTML document and return a metrics dict.

    Returns dict with keys:
        report_period        (date | None)
        combined_ratio       (float | None)
        pif_total            (float | None)  — in thousands
        net_premiums_written (float | None)  — in millions
        net_premiums_earned  (float | None)  — in millions
        net_income           (float | None)  — in millions
        eps_diluted          (float | None)
        loss_ratio           (float | None)
        expense_ratio        (float | None)
    """
    result: dict[str, Any] = {
        "report_period": None,
        "combined_ratio": None,
        "pif_total": None,
        "net_premiums_written": None,
        "net_premiums_earned": None,
        "net_income": None,
        "eps_diluted": None,
        "loss_ratio": None,
        "expense_ratio": None,
    }

    # --- Report period ---
    result["report_period"] = _extract_report_period(html)

    # --- Parse HTML tables ---
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception as exc:
        logger.warning("read_html failed: %s", exc, exc_info=True)
        return result

    # Scan all tables; use label matching rather than table-index heuristics
    # so the parser is robust to minor layout changes across filing years.
    for table in tables:
        for _, row in table.iterrows():
            tokens = _dedup_row(row)
            if not tokens:
                continue
            label = tokens[0].lower()
            nums = _extract_numerics(tokens[1:])

            # --- Summary table metrics (single-month values come before YTD,
            #     so the first match is always the current-month value) ---
            if "combined ratio" in label and nums:
                if result["combined_ratio"] is None:
                    result["combined_ratio"] = nums[0]

            elif label.startswith("net premium") and "written" in label and nums:
                if result["net_premiums_written"] is None:
                    result["net_premiums_written"] = nums[0]

            elif label.startswith("net premium") and "earned" in label and nums:
                if result["net_premiums_earned"] is None:
                    result["net_premiums_earned"] = nums[0]

            elif "net income" in label and "per share" not in label and nums:
                if result["net_income"] is None:
                    result["net_income"] = nums[0]

            elif "per share available to common" in label and nums:
                if result["eps_diluted"] is None:
                    result["eps_diluted"] = nums[0]

            # --- PIF table: grand-total row ---
            # PGR uses several label variants across filing years:
            #   recent  : "Total"
            #   2024+   : "Companywide" or "Companywide Total"
            # Distinguish from NPW/NPE "Total" rows using the heuristic that
            # PIF counts (in thousands) are ≥ 10 000, while monthly NPW/NPE
            # companywide totals are typically < 10 000 (in millions).
            # NOTE: older filings report fractional PIF (e.g. 34364.3),
            # so we do NOT require the value to be a whole number.
            elif (
                label in ("total", "companywide", "companywide total")
                and nums
            ):
                candidate = nums[0]
                if candidate >= 10_000:
                    if result["pif_total"] is None or candidate > result["pif_total"]:
                        result["pif_total"] = candidate

            # --- Supplemental table: companywide GAAP ratios ---
            # The supplemental table columns are ordered:
            # Agency | Direct | Property | PL Total | Commercial | Companywide
            # So the LAST numeric value in the row = companywide figure.
            elif "loss/lae ratio" in label and nums:
                result["loss_ratio"] = nums[-1]

            elif "expense ratio" in label and "net catastrophe" not in label and nums:
                result["expense_ratio"] = nums[-1]

    return result


# ---------------------------------------------------------------------------
# Step 4: orchestrate fetch + parse for N months
# ---------------------------------------------------------------------------

def fetch_monthly_8ks(
    lookback_months: int = 24,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Fetch and parse PGR monthly 8-Ks for the most recent ``lookback_months``.

    Args:
        lookback_months: How many months of history to fetch (default 24).
        session:         Optional ``requests.Session`` (created if None).

    Returns:
        DataFrame indexed by ``report_period`` (month-end date, DatetimeIndex),
        with columns: combined_ratio, pif_total, net_premiums_written,
        net_premiums_earned, net_income, eps_diluted, loss_ratio, expense_ratio.
        Sorted ascending.

    Raises:
        requests.HTTPError: if the EDGAR submissions JSON cannot be fetched.
    """
    if session is None:
        session = requests.Session()

    logger.info("Fetching PGR submissions JSON from EDGAR…")
    filings = fetch_submissions(session)
    filings = filings[:lookback_months]
    logger.info("Found %d monthly 8-K filings (7.01); processing %d", len(filings), len(filings))

    rows: list[dict[str, Any]] = []
    for filing in filings:
        acc = filing["accession"]
        fdate = filing["filing_date"]
        logger.debug("Processing %s (filed %s)", acc, fdate)

        exhibit_url = get_exhibit_url(session, acc)
        if exhibit_url is None:
            logger.warning("Skipping %s — no EX-99 found", acc)
            continue

        try:
            html = _get(session, exhibit_url)
        except requests.HTTPError as exc:
            logger.warning("Could not fetch exhibit for %s: %s", acc, exc)
            continue

        metrics = parse_exhibit(html)
        if metrics["report_period"] is None:
            logger.warning("Could not parse report period for %s — skipping", acc)
            continue

        row = {
            "report_period": pd.Timestamp(metrics["report_period"]),
            "filing_date": fdate,
            "combined_ratio": metrics["combined_ratio"],
            "pif_total": metrics["pif_total"],
            "net_premiums_written": metrics["net_premiums_written"],
            "net_premiums_earned": metrics["net_premiums_earned"],
            "net_income": metrics["net_income"],
            "eps_diluted": metrics["eps_diluted"],
            "loss_ratio": metrics["loss_ratio"],
            "expense_ratio": metrics["expense_ratio"],
        }
        rows.append(row)
        logger.info(
            "  %s: CR=%.1f  PIF=%s  NPW=$%s M",
            metrics["report_period"].strftime("%Y-%m"),
            metrics["combined_ratio"] or float("nan"),
            f"{metrics['pif_total']:,.0f}" if metrics["pif_total"] else "n/a",
            f"{metrics['net_premiums_written']:,.0f}" if metrics["net_premiums_written"] else "n/a",
        )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "combined_ratio", "pif_total", "net_premiums_written",
                "net_premiums_earned", "net_income", "eps_diluted",
                "loss_ratio", "expense_ratio",
            ]
        )
        empty.index = pd.DatetimeIndex([], name="report_period")
        return empty

    df = pd.DataFrame(rows)
    df = df.sort_values("report_period")

    # Deduplicate: if two filings cover the same month-end, keep the later filing.
    df["month_end"] = df["report_period"].dt.to_period("M").dt.to_timestamp("M")
    df = df.drop_duplicates(subset="month_end", keep="last")

    df = df.set_index("month_end")
    df.index.name = "report_period"
    df.index = pd.DatetimeIndex(df.index)

    numeric_cols = [
        "combined_ratio", "pif_total", "net_premiums_written",
        "net_premiums_earned", "net_income", "eps_diluted",
        "loss_ratio", "expense_ratio",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Step 5: derive Gainshare estimate + write to DB
# ---------------------------------------------------------------------------

def _compute_gainshare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append ``pif_growth_yoy`` and ``gainshare_estimate`` columns.

    Uses the same calibration as ``pgr_monthly_loader.py``.
    """
    df = df.copy()

    if "pif_total" in df.columns and not df["pif_total"].isna().all():
        df["pif_growth_yoy"] = df["pif_total"].pct_change(periods=12)
    else:
        df["pif_growth_yoy"] = np.nan

    cr = df.get("combined_ratio", pd.Series(np.nan, index=df.index))
    pif_growth = df.get("pif_growth_yoy", pd.Series(np.nan, index=df.index))

    if not cr.isna().all():
        cr_score = ((_CR_TARGET - cr) / _CR_SCALE).clip(lower=0.0, upper=2.0)
    else:
        cr_score = pd.Series(np.nan, index=df.index)

    if not pif_growth.isna().all():
        pif_score = (pif_growth / _PIF_GROWTH_MAX).clip(lower=0.0, upper=2.0)
    else:
        pif_score = pd.Series(np.nan, index=df.index)

    df["gainshare_estimate"] = 0.5 * cr_score + 0.5 * pif_score
    return df


def backfill_to_db(
    conn,
    lookback_months: int = 24,
    session: requests.Session | None = None,
) -> int:
    """
    Fetch monthly 8-Ks, compute Gainshare, and upsert into ``pgr_edgar_monthly``.

    Args:
        conn:             Open SQLite connection from ``db_client.get_connection()``.
        lookback_months:  Number of months to back-fill (default 24).
        session:          Optional ``requests.Session``.

    Returns:
        Number of rows written.
    """
    from src.database import db_client  # local import to avoid circular

    df = fetch_monthly_8ks(lookback_months=lookback_months, session=session)
    if df.empty:
        logger.warning("No monthly 8-K data fetched — nothing written to DB.")
        return 0

    df = _compute_gainshare(df)

    records = [
        {
            "month_end": idx.strftime("%Y-%m-%d"),
            "combined_ratio": row.get("combined_ratio"),
            "pif_total": row.get("pif_total"),
            "pif_growth_yoy": row.get("pif_growth_yoy"),
            "gainshare_estimate": row.get("gainshare_estimate"),
        }
        for idx, row in df.iterrows()
    ]

    n = db_client.upsert_pgr_edgar_monthly(conn, records)
    logger.info("Wrote %d rows to pgr_edgar_monthly.", n)
    return n
