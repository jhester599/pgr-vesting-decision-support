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

Both runs are idempotent.  Every parsed value is appended to
``pgr_edgar_monthly_raw`` (keyed by accession, field and ``PARSER_VERSION``;
an existing key is left alone), ``db_client.upsert_pgr_edgar_monthly`` never
mixes filings within a row, and derived fields are recomputed over the whole
table (``recompute_derived_fields``).

EDGAR pagination:
  The primary submissions JSON (CIK0000080661.json) only contains the ~1,000
  most recent filings under ``filings.recent``.  Older filings are in the
  pagination files listed in ``filings.files``
  (``CIK0000080661-submissions-001.json``, …), which are *flat*: the parallel
  arrays sit at the top level.  Files whose ``filingTo`` precedes the cutoff
  are skipped.

HTML parsing coverage:
  Every monthly release since August 2004 parses (the August 2004 exhibit is
  plain text and is converted to tables first).  Parse exceptions are caught
  per filing and logged; one bad filing never aborts the full run.
  ``scripts/repair_edgar_history.py`` re-parses the full history.

SEC EDGAR rate limits: 10 requests/second max (this script stays at 4);
``User-Agent`` header required (``EDGAR_USER_AGENT``).
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
from bs4 import BeautifulSoup

# Resolve project root so this script can be run directly or via GitHub Actions.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.processing import pgr_edgar_derived
from src.logging_config import configure_logging


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PGR_CIK: str = "CIK0000080661"
PGR_CIK_NUMERIC: int = 80661          # numeric CIK for archive URLs

SUBMISSIONS_BASE_URL: str = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES_URL: str = "https://www.sec.gov/Archives/edgar/data/80661"

# Earliest filing date a ``--backfill-years`` run reaches.  (The full history
# back to August 2004 is rebuilt by scripts/repair_edgar_history.py.)
BACKFILL_EARLIEST_DATE: str = "2010-01-01"

# Polite rate limit: at most 4 requests/second, well under SEC's 10 req/s.
_SEC_MIN_INTERVAL_SECONDS: float = 0.25
_last_request_monotonic: float | None = None

# Version tag stored with every parsed value in ``pgr_edgar_monthly_raw``.
# Bump it whenever a change to the parser can change a stored value.
PARSER_VERSION: str = "8k-html/2026-09-25"

# Optional on-disk response cache (set by ``set_http_cache_dir``).  EDGAR
# filings are immutable, so a cached body never goes stale; only the
# submissions index changes, and callers can bypass the cache for it.
_http_cache_dir: str | None = None


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

class CachedResponse:
    """Minimal stand-in for ``requests.Response`` served from the disk cache."""

    def __init__(self, url: str, content: bytes, fetched_at: str) -> None:
        self.url = url
        self.content = content
        self.status_code = 200
        self.fetched_at = fetched_at

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        return None


def set_http_cache_dir(path: str | None) -> None:
    """Enable (or with ``None`` disable) the on-disk EDGAR response cache."""
    global _http_cache_dir
    _http_cache_dir = path
    if path is not None:
        os.makedirs(path, exist_ok=True)


def _cache_paths(url: str) -> tuple[str, str]:
    import hashlib

    assert _http_cache_dir is not None
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return (
        os.path.join(_http_cache_dir, f"{key}.body"),
        os.path.join(_http_cache_dir, f"{key}.json"),
    )


def _throttle() -> None:
    """Sleep so that consecutive network requests start >= 0.25 s apart."""
    global _last_request_monotonic
    now = time.monotonic()
    if _last_request_monotonic is not None:
        wait = _SEC_MIN_INTERVAL_SECONDS - (now - _last_request_monotonic)
        if wait > 0:
            time.sleep(wait)
    _last_request_monotonic = time.monotonic()


def _get(url: str, retries: int = 3, use_cache: bool = True) -> Any:
    """GET with retry, a 4 req/s throttle and the optional disk cache.

    Returns a ``requests.Response`` for a network fetch, or a
    ``CachedResponse`` when the body is served from the cache.  Both expose
    ``text``, ``content``, ``json()`` and a ``fetched_at`` ISO timestamp.
    """
    if _http_cache_dir is not None and use_cache:
        body_path, meta_path = _cache_paths(url)
        if os.path.exists(body_path) and os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            with open(body_path, "rb") as fh:
                return CachedResponse(url, fh.read(), meta.get("fetched_at", ""))
    resp = _get_network(url, retries=retries)
    fetched_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    try:
        resp.fetched_at = fetched_at
    except AttributeError:
        pass
    if _http_cache_dir is not None:
        body_path, meta_path = _cache_paths(url)
        with open(body_path, "wb") as fh:
            fh.write(resp.content)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"url": url, "fetched_at": fetched_at}, fh)
    return resp


def _get_network(url: str, retries: int = 3) -> requests.Response:
    """GET with exponential back-off retry and polite inter-request delay."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            _throttle()
            resp = requests.get(
                url,
                headers=config.build_edgar_headers(),
                timeout=30,
            )
            resp.raise_for_status()
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
    # The submissions index grows with every filing: never serve it from cache.
    resp = _get(url, use_cache=False)
    return resp.json()


def _filings_block(page: dict) -> dict:
    """Return the parallel-array filings block of a submissions page.

    The primary ``CIK##########.json`` nests it at ``filings.recent``; the
    pagination files (``CIK##########-submissions-001.json`` …) are flat and
    carry ``accessionNumber``, ``form``, ``items`` … at the top level (F17).
    """
    nested = page.get("filings", {}).get("recent")
    if nested:
        return nested
    if "accessionNumber" in page:
        return page
    return {}


def _match_item_code(item_str: str) -> str | None:
    """Return the item code that makes an 8-K a candidate monthly release.

    * ``2.02`` — Results of Operations (quarter-end months); wins when a
      filing lists both 2.02 and 7.01.
    * ``7.01`` — Regulation FD monthly supplement (non-quarter-end months).
    * ``9.01`` alone — exhibits only.  PGR filed at least one monthly
      release this way (May 2015, 0000080661-15-000034).  Such a filing is
      accepted only if its index lists an EX-99 exhibit, and only kept if
      that exhibit parses as an earnings release.
    """
    parsed_items = [i.strip() for i in str(item_str).split(",") if i.strip()]
    # A filing listing both is the quarter-end results release.
    if "2.02" in parsed_items:
        return "2.02"
    if "7.01" in parsed_items:
        return "7.01"
    if parsed_items == ["9.01"]:
        return "9.01"
    return None


def _collect_8k_filings(
    recent: dict,
    cutoff_date: str,
    out: list[dict[str, Any]],
) -> bool:
    """Extract PGR operating-metrics 8-K filings from one filings block.

    Accepts item 7.01 (Regulation FD monthly supplement, used for
    non-quarter-end months), item 2.02 (Results of Operations, used for
    quarter-end months: March, June, September, December) and 9.01-only
    filings (see ``_match_item_code``).  The matched item code is stored in
    the ``"item_code"`` key of each output dict so that
    ``_parse_html_exhibit`` can set ``filing_type`` correctly.

    Modifies ``out`` in-place with matching filings.

    Args:
        recent: Dict with parallel arrays ``form``, ``filingDate``, ``items``,
            ``accessionNumber`` as returned by the EDGAR submissions endpoint
            (``filings.recent`` or a flat pagination file).
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
        item_code = _match_item_code(item_str)
        if item_code is None:
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
                "item_code": item_code,
            }
        )

    return passed_cutoff


def fetch_all_8k_filings(cutoff_date: str) -> list[dict[str, Any]]:
    """Fetch all PGR operating-metrics 8-K filings (items 7.01 and 2.02) back to ``cutoff_date``.

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
    _collect_8k_filings(_filings_block(primary), cutoff_date, results)

    # Pagination overflow files are listed in primary["filings"]["files"]
    extra_files = primary.get("filings", {}).get("files", [])
    for file_entry in extra_files:
        name = file_entry.get("name", "")
        m = re.search(r"-submissions-(\d+)\.json$", name)
        if not m:
            continue
        page_id = m.group(1)
        # Skip pages that end before the cutoff (the primary lists their range).
        filing_to = file_entry.get("filingTo")
        if filing_to and filing_to < cutoff_date:
            continue
        log.info("Fetching pagination file %s …", name)
        page_data = _fetch_submissions_page(page_id)
        stop = _collect_8k_filings(_filings_block(page_data), cutoff_date, results)
        if stop:
            log.debug("Oldest filing in %s precedes cutoff — stopping pagination.", name)
            break

    unique = {r["accession_number"]: r for r in results}
    results = sorted(unique.values(), key=lambda r: r["filing_date"])
    log.info(
        "Found %d candidate 8-K (items 7.01/2.02/9.01) filings back to %s.",
        len(results), cutoff_date,
    )
    return results


# ---------------------------------------------------------------------------
# Filing document URL resolution
# ---------------------------------------------------------------------------

def _get_all_filing_doc_urls(
    accession_number: str,
    accession_dashed: str,
    require_ex99: bool = False,
) -> list[str]:
    """Return all candidate HTML exhibit URLs for an 8-K filing, sorted by preference.

    Fetches the filing index page and returns every suitable ``.htm`` document
    in preference order: PGR-named files first, then 8-K-named files, then
    everything else.  This allows callers to try multiple exhibits (important
    for quarterly earnings 8-Ks where the main 8-K form cover is listed before
    the Exhibit 99.1 operating supplement that contains the actual data).

    Args:
        accession_number: Cleaned accession (no dashes), e.g.
            ``"000008066124000001"``.
        accession_dashed: Original dashed form, e.g.
            ``"0000080661-24-000001"``.
        require_ex99: Return an empty list unless the index lists an EX-99
            ``.htm`` exhibit (used for 9.01-only filings).

    Returns:
        Ordered list of full exhibit URLs (EX-99 exhibits first); empty list
        if the index cannot be fetched or no suitable ``.htm`` files are found.
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
        return []

    # EX-99 exhibits (the earnings release) come first, whatever their name.
    ex99 = [
        f"https://www.sec.gov{href}"
        for href, doc_type in _index_documents(html)
        if doc_type.upper().startswith("EX-99") and href.lower().endswith(".htm")
    ]
    # Plain-text EX-99 exhibits (only Aug-2004 among the monthly releases)
    # are tried last; ``parse_filing`` converts them with
    # ``_text_exhibit_to_html``.
    ex99_text = [
        f"https://www.sec.gov{href}"
        for href, doc_type in _index_documents(html)
        if doc_type.upper().startswith("EX-99") and href.lower().endswith(".txt")
    ]
    if require_ex99 and not ex99 and not ex99_text:
        return []

    # Extract all .htm hrefs from the filing index
    pattern = re.compile(
        r'href="(/Archives/edgar/data/80661/[^"]+\.htm)"',
        re.IGNORECASE,
    )
    candidates = pattern.findall(html)

    pgr_named: list[str] = []
    k8_named: list[str] = []
    other: list[str] = []

    for href in candidates:
        if f"https://www.sec.gov{href}" in ex99:
            continue
        fname = href.split("/")[-1].lower()
        # Skip XBRL viewer and inline XBRL files
        if any(skip in href for skip in ("ix?doc=", "R1.htm", "R2.htm")):
            continue
        if fname.startswith("r") and fname[1:].isdigit():
            continue
        url = f"https://www.sec.gov{href}"
        if "pgr" in fname:
            pgr_named.append(url)
        elif "8k" in fname or "8-k" in fname:
            k8_named.append(url)
        else:
            other.append(url)

    return ex99 + pgr_named + k8_named + other + ex99_text


def _text_exhibit_to_html(text: str) -> str:
    """Wrap a plain-text (pre-2005 EDGAR) exhibit so the HTML parser can read it.

    Each ``<TABLE>`` block becomes an HTML table with one row per line and
    cells split on runs of two or more spaces (the fixed-width column gaps).
    The full text is kept in a ``<pre>`` block for the text-mode fallbacks.
    """
    from html import escape

    tables: list[str] = []
    for block in re.findall(r"<TABLE>(.*?)</TABLE>", text, flags=re.IGNORECASE | re.DOTALL):
        rows: list[str] = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or re.fullmatch(r"(<[A-Z]+>\s*)+", stripped):
                continue
            if re.fullmatch(r"[-=\s]+", stripped):
                continue
            cells = re.split(r"\s{2,}", stripped)
            rows.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")
        if rows:
            tables.append("<table>" + "".join(rows) + "</table>")
    body = re.sub(r"<TABLE>.*?</TABLE>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    return (
        "<html><body>" + "".join(tables) + "<pre>" + escape(body) + "</pre></body></html>"
    )


_INDEX_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_INDEX_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)


def _index_documents(index_html: str) -> list[tuple[str, str]]:
    """Return ``(href, type)`` for each row of a filing index's document table.

    The table columns are Seq, Description, Document, Type, Size; ``type`` is
    e.g. ``"8-K"``, ``"EX-99"``, ``"EX-99.1"`` or ``"GRAPHIC"``.
    """
    docs: list[tuple[str, str]] = []
    for row in _INDEX_ROW_RE.findall(index_html):
        cells = _INDEX_CELL_RE.findall(row)
        if len(cells) < 4:
            continue
        href = re.search(r'href="([^"]+)"', cells[2])
        if href is None:
            continue
        doc_type = re.sub(r"<[^>]+>", "", cells[3]).strip()
        docs.append((href.group(1), doc_type))
    return docs


def _get_filing_doc_url(
    accession_number: str,
    accession_dashed: str,
) -> str | None:
    """Return the top-priority HTML exhibit URL for an 8-K filing.

    Thin wrapper around ``_get_all_filing_doc_urls`` that returns only the
    first (highest-priority) candidate.  For quarterly 8-Ks you should call
    ``_get_all_filing_doc_urls`` directly so all exhibits can be tried.

    Returns:
        Full URL to the primary HTML exhibit, or ``None`` if none found.
    """
    urls = _get_all_filing_doc_urls(accession_number, accession_dashed)
    return urls[0] if urls else None


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
    cleaned = _UNICODE_MINUS_RE.sub("-", cleaned)
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


# ---------------------------------------------------------------------------
# BeautifulSoup-based table-classification helpers (ported from parse_pgr_8k.py)
# ---------------------------------------------------------------------------

import re as _re  # already imported at module level but kept local alias for clarity

# Dash-like characters that PGR's filings use as a minus sign when they sit
# directly before a number: U+2212 minus, U+2012 figure dash, U+2013 en dash,
# U+2011 non-breaking hyphen, U+FE63 small hyphen-minus, U+FF0D fullwidth.
_UNICODE_MINUS_RE = _re.compile(r"[\u2212\u2012\u2013\u2011\ufe63\uff0d](?=\s*[\d.])")

# A cell holding only an opening parenthesis (optionally with "$"): the
# number follows in the next cell and is negative.
_OPEN_PAREN_CELL_RE = _re.compile(r"^\$?\s*\(\s*\$?$")


_LEADING_NUMBER_RE = _re.compile(
    r"^[^\d(.\-]*(?P<paren>\()?\s*(?P<minus>-)?\s*(?P<num>\d+(?:\.\d+)?|\.\d+)"
)


def _parse_number(text: str) -> float | None:
    """Parse a financial number from a cell's text.

    Handles ``$1,234.5``, ``89.2%``, parenthesised negatives ``(56.7)`` and
    ``$(56.7)``, a split-cell opening parenthesis ``(56.7`` (the closing
    ``)`` sits in the next cell), and Unicode minus signs (``−56.7``).
    A cell holding only the closing half (``56.7)``) is returned positive;
    ``_row_numbers`` applies the sign from the preceding ``(`` cell.  Only
    the first number in the cell is read, so a trailing footnote marker
    (``2.9% 1``) is ignored.
    """
    if not text:
        return None
    t = _UNICODE_MINUS_RE.sub("-", text.replace("\xa0", " "))
    t = t.replace(",", "").replace("$", "").replace("%", "").strip()
    match = _LEADING_NUMBER_RE.match(t)
    if match is None:
        return None
    value = float(match.group("num"))
    if match.group("paren") or match.group("minus"):
        return -value
    return value


def _row_numbers(cells: list[str], decimals_only: bool = False) -> list[float]:
    """Return every number in a row's cells, in order, with split-cell signs.

    Some filings put the parentheses of a negative in their own cells:
    ``["(", "16.8", ")"]`` or ``["$(", "16.8", ")"]``.  A lone opening
    parenthesis marks the next number negative; ``$`` cells in between are
    skipped.

    With ``decimals_only`` a cell counts only if it holds a decimal point.
    Ratios are always printed to one decimal, so this drops footnote
    markers that sit in their own cells between ratio columns (2025-09:
    "36.5 | 2 | 37.0 | 2 | …" shifted the companywide column; F34).
    """
    numbers: list[float] = []
    pending_negative = False
    for cell in cells:
        text = cell.replace("\xa0", " ").strip()
        if not text:
            continue
        if _OPEN_PAREN_CELL_RE.match(text):
            pending_negative = True
            continue
        value = _parse_number(text)
        if value is None:
            if text not in ("$", ")", ")%", "%)", "%"):
                pending_negative = False
            continue
        if decimals_only and "." not in text:
            continue
        if pending_negative:
            value = -abs(value)
            pending_negative = False
        numbers.append(value)
    return numbers


def _get_first_numeric(cells: list[str]) -> float | None:
    """Return the first parseable number from a list of cell-text strings."""
    numbers = _row_numbers(cells)
    return numbers[0] if numbers else None


def _cells_text(row) -> list[str]:
    """Return stripped text of every td/th in a BS4 row, from the first non-empty cell.

    Leading empty cells are dropped so that ``cells[0]`` is the row label even
    in layouts with a spacer column before it (e.g. the 2007-08 balance sheet).
    Runs of whitespace, including line breaks inside a label ("Other\nPersonal
    Lines" in 2004), become one space.
    """
    cells = [
        re.sub(r"\s+", " ", c.get_text(separator=" ", strip=True))
        for c in row.find_all(["td", "th"])
    ]
    for idx, cell in enumerate(cells):
        if cell:
            return cells[idx:]
    return []


def _find_row_value(rows, *keywords) -> float | None:
    """Scan rows; return first numeric from the row whose first cell contains ALL keywords."""
    kws = [k.lower() for k in keywords]
    for row in rows:
        ctexts = _cells_text(row)
        if not ctexts:
            continue
        label = ctexts[0].lower()
        if all(k in label for k in kws):
            return _get_first_numeric(ctexts[1:])
    return None


def _normalise_label(label: str) -> str:
    """Lower-case a row label, unify apostrophes and drop footnote markers."""
    t = label.replace("\u2019", "'").replace("\x92", "'").replace("\u2018", "'")
    t = re.sub(r"\s+", " ", t.replace("\xa0", " ")).strip().lower()
    t = re.sub(r"(\s*\(\d\)|\s+\d)+$", "", t)  # trailing "(1)" / " 2" footnotes
    return t.rstrip(":").strip()


# Anchored balance-sheet labels (F18).  A bare substring match picked up
# "Return on average shareholders' equity" and "Debt to total capital ratio".
_EQUITY_LABEL_RE = re.compile(
    r"^(total )?(common )?shareholders'? equity( \(deficit\))?$"
)
_DEBT_LABEL_RE = re.compile(r"^(total )?debt( outstanding)?$")


def _find_anchored_row_value(rows, label_re: re.Pattern[str]) -> float | None:
    """Return the first number from a row whose whole label matches ``label_re``.

    Rows whose label matches but carry no number (section headers) are
    skipped, so a later "Total shareholders' equity" row is still found.
    """
    for row in rows:
        ctexts = _cells_text(row)
        if not ctexts:
            continue
        if not label_re.match(_normalise_label(ctexts[0])):
            continue
        value = _get_first_numeric(ctexts[1:])
        if value is not None:
            return value
    return None


def _find_row_value_multi(rows, keyword_sets) -> float | None:
    """Try multiple keyword sets in order; return first match."""
    for kws in keyword_sets:
        v = _find_row_value(rows, *kws)
        if v is not None:
            return v
    return None


def _table_header_text(tbl, n_rows: int = 3) -> str:
    """Get lowercase joined text of the first n_rows of a table."""
    rows = tbl.find_all("tr")[:n_rows]
    return " ".join(r.get_text(separator=" ", strip=True) for r in rows).lower()


def _table_full_text(tbl) -> str:
    """Full lowercase text of the table."""
    return tbl.get_text(separator=" ", strip=True).lower()


_EXCL_KEYWORDS = [
    "year-to-date", "year to date", "full year", "annual",
    "year (",
]


def _is_excluded_table(tbl) -> bool:
    """True if this table should be skipped (year-to-date, full-year sections)."""
    hdr = _table_header_text(tbl, 4)
    for kw in _EXCL_KEYWORDS:
        if kw in hdr:
            if "current month" in hdr or "comments on monthly" in hdr:
                return False
            return True
    return False


def _is_income_stmt_table(tbl_text: str, n: int) -> bool:
    # Strict form: has NPE, losses, total revenues, income taxes, enough rows
    strict = (
        "net premiums earned" in tbl_text
        and "losses and loss adjustment" in tbl_text
        and "total revenues" in tbl_text
        and "before income taxes" in tbl_text
        and n >= 15
    )
    # Relaxed form: losses + total revenues + income taxes is a unique IS fingerprint
    # even when NPE is in a separate summary table or row count is smaller
    relaxed = (
        "losses and loss adjustment" in tbl_text
        and "total revenues" in tbl_text
        and "before income taxes" in tbl_text
        and n >= 8
    )
    return strict or relaxed


def _is_eps_table(tbl_text: str) -> bool:
    has_eps_signal = (
        "per share" in tbl_text
        or "comprehensive income" in tbl_text
        or ("basic" in tbl_text and "diluted" in tbl_text)
    )
    return has_eps_signal and "average" in tbl_text and "shares outstanding" in tbl_text


def _is_investment_returns_table(tbl_text: str) -> bool:
    return (
        "fully taxable equivalent" in tbl_text
        or ("fixed income securities" in tbl_text and "total portfolio" in tbl_text)
        or ("fixed-income securities" in tbl_text and "total portfolio" in tbl_text)
    )


def _is_balance_sheet_table(tbl_text: str) -> bool:
    return (
        (
            "total assets" in tbl_text
            or "book value per share" in tbl_text
            or "book value per common share" in tbl_text
        )
        and (
            "shareholders" in tbl_text
            or "common shares outstanding" in tbl_text
        )
    )


def _is_policies_table(tbl_text: str, n: int) -> bool:
    return (
        "policies in force" in tbl_text
        and ("agency" in tbl_text or "direct" in tbl_text)
        and n < 25
    )


def _is_summary_table(tbl_text: str, n: int) -> bool:
    return (
        "net premiums written" in tbl_text
        and "combined ratio" in tbl_text
        and "net income" in tbl_text
        and n < 60
    )


def _is_segment_table(tbl_text: str, n: int) -> bool:
    # Full-label form: explicit Agency/Direct/Commercial column headers
    has_labels = (
        "agency" in tbl_text
        and "direct" in tbl_text
        and "commercial" in tbl_text
        and "net premiums written" in tbl_text
        and n >= 10
    )
    # Compact form: NPW + ratio rows without explicit segment column headers
    # (used in some test fixtures and condensed release formats)
    has_ratio_rows = (
        "net premiums written" in tbl_text
        and "combined ratio" in tbl_text
        and "net income" not in tbl_text  # exclude summary/IS tables
        and n >= 3
    )
    return has_labels or has_ratio_rows


def _extract_income_stmt(tbl) -> dict:
    """Extract income statement fields from an income-statement table."""
    rows = tbl.find_all("tr")
    d: dict = {}

    def rv(*kws: str) -> float | None:
        return _find_row_value(rows, *kws)

    d["net_premiums_written"]       = rv("net premiums written")
    d["net_premiums_earned"]        = rv("net premiums earned")
    d["investment_income"]          = rv("investment income")
    d["total_net_realized_gains"]   = _find_row_value_multi(rows, [
        ("total net realized gains",),
        ("net realized gains (losses) on securities",),
        ("net realized gains on securities",),
    ])
    d["fees_and_other_revenues"]    = rv("fees and other revenues")
    d["service_revenues"]           = rv("service revenues")
    d["total_revenues"]             = rv("total revenues")
    d["losses_lae"]                 = rv("losses and loss adjustment expenses")
    d["policy_acquisition_costs"]   = rv("policy acquisition costs")
    d["other_underwriting_expenses"] = rv("other underwriting expenses")
    d["interest_expense"]           = rv("interest expense")
    d["total_expenses"]             = rv("total expenses")
    d["income_before_income_taxes"] = _find_row_value_multi(rows, [
        ("income before income taxes",),
        ("income", "before income taxes"),
    ])
    d["provision_for_income_taxes"] = _find_row_value_multi(rows, [
        ("provision for income taxes",),
        ("provision", "income taxes"),
        ("benefit for income taxes",),
        ("for income taxes",),
    ])
    d["net_income"]                 = rv("net income")
    d["total_comprehensive_income"] = _find_row_value_multi(rows, [
        ("total comprehensive income",),
        ("comprehensive income",),
    ])

    return {k: v for k, v in d.items() if v is not None}


def _extract_eps_comprehensive(tbl) -> dict:
    """Extract EPS and comprehensive-income fields using a state-machine row scan."""
    rows = tbl.find_all("tr")
    d: dict = {}

    in_net_income_eps  = False
    in_comprehensive   = False
    in_basic_section   = False
    in_diluted_section = False

    for row in rows:
        ctexts = _cells_text(row)
        label = ctexts[0].lower().strip() if ctexts else ""
        num   = _get_first_numeric(ctexts[1:]) if len(ctexts) > 1 else None

        if "net income" in label and "comprehensive" not in label:
            in_net_income_eps  = True
            in_comprehensive   = False
            in_basic_section   = False
            in_diluted_section = False
            if num is not None and "net_income" not in d:
                d["net_income"] = num
            continue

        if "comprehensive income" in label or "comprehensive loss" in label:
            in_comprehensive   = True
            in_net_income_eps  = False
            in_basic_section   = False
            in_diluted_section = False
            if num is not None:
                d.setdefault("total_comprehensive_income", num)
            continue

        if "per common share" in label:
            continue

        bare = label.rstrip(":").strip()
        bare_root = re.sub(r"\s+\d+$", "", bare)

        if bare_root == "basic":
            in_basic_section   = True
            in_diluted_section = False
            if in_net_income_eps and num is not None:
                d.setdefault("eps_basic", num)
            continue

        if bare_root == "diluted":
            in_diluted_section = True
            in_basic_section   = False
            if in_net_income_eps and num is not None:
                d.setdefault("eps_diluted", num)
            elif in_comprehensive and num is not None:
                d.setdefault("comprehensive_eps_diluted", num)
            continue

        if "per share" in label and num is not None:
            if in_net_income_eps:
                if in_basic_section:
                    d.setdefault("eps_basic", num)
                elif in_diluted_section:
                    d.setdefault("eps_diluted", num)
            elif in_comprehensive and in_diluted_section:
                d.setdefault("comprehensive_eps_diluted", num)
            continue

        if "average common shares outstanding" in label and "basic" in label:
            if num is not None:
                d.setdefault("avg_shares_basic", num)
        elif "average shares outstanding" in label:
            if num is not None:
                d.setdefault("avg_shares_basic", num)
        elif "total average equivalent" in label or "total equivalent shares" in label:
            if num is not None:
                d.setdefault("avg_shares_diluted", num)
        elif ("after-tax" in label or "unrealized" in label
              or "forecast" in label or "foreign currency" in label):
            in_net_income_eps  = False
            in_comprehensive   = False
            in_basic_section   = False
            in_diluted_section = False

    if "eps_basic" not in d:
        d["eps_basic"] = _find_row_value_multi(rows, [("per share", "basic")])
    if "eps_diluted" not in d:
        d["eps_diluted"] = _find_row_value_multi(rows, [("per share", "diluted")])
    if "avg_shares_diluted" not in d:
        d["avg_shares_diluted"] = _find_row_value_multi(rows, [
            ("total equivalent shares",), ("total average equivalent",),
        ])

    # Positional fallback for standalone EPS tables where "net income" / "comprehensive
    # income" trigger rows are absent (e.g. compact or test-fixture formats).
    # Scan for bare "Basic" / "Diluted" rows in order; first Diluted = eps_diluted,
    # second Diluted = comprehensive_eps_diluted.
    if "eps_basic" not in d or "eps_diluted" not in d or "comprehensive_eps_diluted" not in d:
        _diluted_count = 0
        for row in rows:
            ctexts = _cells_text(row)
            if not ctexts:
                continue
            bare = re.sub(r"\s+\d+$", "", ctexts[0].lower().rstrip(":").strip())
            num = _get_first_numeric(ctexts[1:]) if len(ctexts) > 1 else None
            if num is None or not (0.0 < abs(num) < 50.0):
                continue
            if bare == "basic" and "eps_basic" not in d:
                d["eps_basic"] = num
            elif bare == "diluted":
                _diluted_count += 1
                if _diluted_count == 1 and "eps_diluted" not in d:
                    d["eps_diluted"] = num
                elif _diluted_count == 2 and "comprehensive_eps_diluted" not in d:
                    d["comprehensive_eps_diluted"] = num

    return {k: v for k, v in d.items() if v is not None}


def _extract_investment_returns(tbl) -> dict:
    """Extract FTE total-return and book-yield rows.

    Rows are read from the "fully taxable equivalent" header onward when it
    is present, so a table that also holds EPS or balance-sheet rows (e.g.
    2023-04..2023-09) does not feed its earlier rows to these label matches.
    """
    rows = tbl.find_all("tr")
    for idx, row in enumerate(rows):
        if "fully taxable equivalent" in row.get_text(" ", strip=True).lower():
            rows = rows[idx:]
            break
    d: dict = {}

    d["fte_return_fixed_income"]    = _find_row_value_multi(rows, [
        ("fixed-income securities",),
        ("fixed income securities",),
    ])
    d["fte_return_common_stocks"]   = _find_row_value(rows, "common stocks")
    d["fte_return_total_portfolio"] = _find_row_value(rows, "total portfolio")
    _yield_raw = _find_row_value_multi(rows, [
        ("investment income book yield",),
        ("recurring investment book yield",),
        ("pretax recurring",),
        ("pretax annualized",),
    ])
    # Stored in percent (e.g. "3.3%" -> 3.3), matching the historical CSV.
    # Migration 004 rescales rows written by the old /100 conversion (F10).
    d["investment_book_yield"] = _yield_raw

    return {k: v for k, v in d.items() if v is not None}


def _extract_shares_repurchased_raw(rows) -> float | None:
    """Return raw numeric share count (no unit conversion) for _resolve_share_count."""
    kws = ("common shares repurchased", "shares repurchased")
    for row in rows:
        ctexts = _cells_text(row)
        if not ctexts:
            continue
        label = ctexts[0].lower()
        if any(k in label for k in kws):
            val = _get_first_numeric(ctexts[1:])
            return val  # return raw; _resolve_share_count handles normalisation
    return None


def _find_total_liabilities_only(rows) -> float | None:
    """Return explicit 'Total liabilities' value, skipping the equity-combined row."""
    for row in rows:
        ctexts = _cells_text(row)
        label = ctexts[0].lower() if ctexts else ""
        if "total liabilities" in label and "shareholders" not in label and "equity" not in label:
            return _get_first_numeric(ctexts[1:])
    return None


def _extract_roe_net_income(rows) -> float | None:
    """Locate the trailing-12m net-income ROE value."""
    roe_section = False
    for row in rows:
        ctexts = _cells_text(row)
        label = ctexts[0].lower() if ctexts else ""
        if "trailing 12" in label or "return on average" in label:
            roe_section = True
            v = _get_first_numeric(ctexts[1:])
            if v is not None:
                return v
            continue
        if roe_section:
            if "net income" in label:
                v = _get_first_numeric(ctexts[1:])
                if v is not None:
                    return v
            elif "comprehensive income" in label:
                pass
            elif label.strip():
                roe_section = False
    return None


def _extract_roe_comprehensive(rows) -> float | None:
    """Locate the trailing-12m comprehensive-income ROE value."""
    roe_section = False
    for row in rows:
        ctexts = _cells_text(row)
        label = ctexts[0].lower() if ctexts else ""
        if "trailing 12" in label or "return on average" in label:
            roe_section = True
            continue
        if roe_section:
            if "comprehensive income" in label:
                v = _get_first_numeric(ctexts[1:])
                if v is not None:
                    return v
            elif label.strip() and "net income" not in label:
                roe_section = False
    return None


def _extract_credit_quality(rows) -> str | None:
    """Extract weighted-average credit-quality string (e.g. 'AA-')."""
    for row in rows:
        ctexts = _cells_text(row)
        label = ctexts[0].lower() if ctexts else ""
        if "weighted average credit quality" in label or "weighted-average credit" in label:
            for c in ctexts[1:]:
                c = c.strip()
                if c and c not in ("", "$"):
                    if not re.match(r"^[\d\.\-]+$", c):
                        return c
    return None


def _set_if_absent(d: dict, key: str, val: Any) -> None:
    """Set key only if not already present (first assignment wins)."""
    if val is not None and key not in d:
        d[key] = val


def _map_segment_nums(
    d: dict,
    prefix: str,
    nums: list[float],
    has_property: bool,
) -> None:
    """Map ordered segment values into d[prefix_agency], d[prefix_direct], etc."""
    _set_if_absent(d, f"{prefix}_agency", nums[0])
    _set_if_absent(d, f"{prefix}_direct", nums[1])

    if has_property and len(nums) >= 6:
        expected_pl = nums[0] + nums[1]
        old_layout = abs(nums[2] - expected_pl) < max(5.0, 0.02 * expected_pl)
        if old_layout:
            _set_if_absent(d, f"{prefix}_commercial", nums[3])
            _set_if_absent(d, f"{prefix}_property",   nums[4])
        else:
            _set_if_absent(d, f"{prefix}_property",   nums[2])
            _set_if_absent(d, f"{prefix}_commercial", nums[4])
    elif not has_property:
        if len(nums) == 5:
            _set_if_absent(d, f"{prefix}_commercial", nums[3])
        elif len(nums) == 6:
            _set_if_absent(d, f"{prefix}_commercial", nums[3])
        elif len(nums) == 4:
            _set_if_absent(d, f"{prefix}_commercial", nums[2])


def _select_companywide_ratio(nums: list[float], expected: float | None = None) -> float:
    """Pick the current-period companywide ratio from monthly or quarterly rows."""
    candidate = nums[-2] if len(nums) >= 7 else nums[-1]
    if expected is None or abs(candidate - expected) <= 5.0:
        return candidate

    for tol in (0.5, 5.0):
        for value in nums:
            if 60.0 <= value <= 140.0 and abs(value - expected) <= tol:
                return value
    return candidate


def _extract_balance_sheet(tbl, month_end: str) -> dict:
    """Extract balance-sheet and capital metrics."""
    rows = tbl.find_all("tr")
    d: dict = {}

    def rv(*kws: str) -> float | None:
        return _find_row_value(rows, *kws)

    d["total_investments"]   = rv("total investments")
    d["total_assets"]        = rv("total assets")
    d["loss_lae_reserves"]   = rv("loss and loss adjustment expense reserves")
    d["unearned_premiums"]   = rv("unearned premiums")
    d["debt"]                = _find_anchored_row_value(rows, _DEBT_LABEL_RE)
    d["total_liabilities"]   = _find_total_liabilities_only(rows)
    d["shareholders_equity"] = _find_anchored_row_value(rows, _EQUITY_LABEL_RE)
    d["common_shares_outstanding"] = rv("common shares outstanding")

    # Shares repurchased: extract raw, then normalise via _resolve_share_count.
    # Extract avg_cost first so _resolve_share_count can use it for cross-validation.
    _raw_repurchased = _extract_shares_repurchased_raw(rows)
    avg_cost = _find_row_value_multi(rows, [
        ("average cost per common share",),
        ("average cost per share",),
    ])
    d["avg_cost_per_share"] = avg_cost

    if _raw_repurchased is not None:
        d["shares_repurchased"] = _resolve_share_count(_raw_repurchased, avg_cost, month_end)

    d["book_value_per_share"] = _find_row_value_multi(rows, [
        ("book value per common share",),
        ("book value per share",),
    ])
    d["roe_net_income_trailing_12m"]     = _extract_roe_net_income(rows)
    d["roe_comprehensive_trailing_12m"]  = _extract_roe_comprehensive(rows)
    d["debt_to_total_capital"]           = _find_row_value_multi(rows, [
        ("debt-to-total capital ratio",),
        ("debt to total capital ratio",),
        ("debt to total capital",),
    ])
    d["fixed_income_duration"]           = _find_row_value_multi(rows, [
        ("fixed-income portfolio duration",),
        ("fixed income portfolio duration",),
    ])
    d["net_unrealized_gains_fixed"]      = _find_row_value_multi(rows, [
        ("net unrealized pretax gains (losses) on fixed",),
        ("net unrealized pretax gains (losses)",),
        ("net unrealized pretax gains on investments",),
        ("net unrealized pre-tax gains",),
    ])
    cq = _extract_credit_quality(rows)
    if cq:
        d["weighted_avg_credit_quality"] = cq.rstrip(". \t")

    return {k: v for k, v in d.items() if v is not None}


def _extract_policies_in_force(tbl) -> dict:
    """Extract Policies in Force (thousands).

    Only rows from the first "policies in force" row onward are read, so the
    block can sit inside a larger table whose earlier rows reuse the same
    labels (e.g. "Agency – Auto" net premiums written).
    """
    rows = tbl.find_all("tr")
    for idx, row in enumerate(rows):
        if "policies in force" in row.get_text(" ", strip=True).lower():
            rows = rows[idx:]
            break
    d: dict = {}

    d["pif_agency_auto"]  = _find_row_value_multi(rows, [
        ("agency", "auto"),
        ("agency \u2013 auto",),
        ("agency  auto",),
        # 2006-01..2007-01 releases label the agency channel by its brand.
        ("drive", "auto"),
    ])
    d["pif_direct_auto"]  = _find_row_value_multi(rows, [
        ("direct", "auto"),
        ("direct \u2013 auto",),
        ("direct  auto",),
    ])
    d["pif_special_lines"] = _find_row_value_multi(rows, [
        ("special lines",),
        ("other personal lines",),
        ("total special lines",),
    ])
    d["pif_property"] = _find_row_value_multi(rows, [("property",)])
    d["pif_total_personal_lines"] = _find_row_value_multi(rows, [("total personal lines",)])
    d["pif_commercial_lines"] = _find_row_value_multi(rows, [
        ("commercial lines",),
        ("commercial auto business",),
        ("total commercial auto",),
        ("commercial auto",),
    ])
    d["pif_total"] = _find_row_value_multi(rows, [
        ("total", "policies in force"),
        ("companywide total",),
        ("companywide",),
    ])
    if d.get("pif_total") is None:
        pl = d.get("pif_total_personal_lines")
        cl = d.get("pif_commercial_lines")
        if pl is not None and cl is not None:
            d["pif_total"] = round(pl + cl, 1)

    return {k: v for k, v in d.items() if v is not None}


def _extract_segment(tbl) -> dict:
    """Extract NPW/NPE by segment and companywide GAAP ratios from the segment table."""
    rows = tbl.find_all("tr")
    if not rows:
        return {}

    grid = [_cells_text(r) for r in rows]
    header_text = " ".join(" ".join(g) for g in grid[:6]).lower()
    has_property = "property" in header_text

    d: dict = {}

    for row_texts in grid:
        label = row_texts[0].lower().strip() if row_texts else ""

        if any(x in label for x in ("calendar year", "accident year",
                                     "prior accident", "current accident",
                                     "reserve", "development")):
            continue

        is_ratio_row = "ratio" in label
        nums = _row_numbers(row_texts[1:], decimals_only=is_ratio_row)

        if not nums:
            continue

        if "net premiums written" in label and len(nums) >= 4:
            _map_segment_nums(d, "npw", nums, has_property)
        elif "net premiums earned" in label and len(nums) >= 4:
            _map_segment_nums(d, "npe", nums, has_property)
        elif ("combined ratio" in label
              and "accident" not in label and "calendar" not in label):
            lr = d.get("loss_lae_ratio")
            er = d.get("expense_ratio")
            expected = lr + er if lr is not None and er is not None else None
            _set_if_absent(d, "combined_ratio", _select_companywide_ratio(nums, expected))
        elif (("loss/lae ratio" in label or ("loss" in label and "lae ratio" in label))
              and "accident" not in label and "calendar" not in label):
            _set_if_absent(d, "loss_lae_ratio", _select_companywide_ratio(nums))
        elif ("expense ratio" in label
              and "accident" not in label and "calendar" not in label):
            _set_if_absent(d, "expense_ratio", _select_companywide_ratio(nums))

    return d


def _extract_summary_table(tbl) -> dict:
    """Extract fields from the brief summary table at the top of each release."""
    rows = tbl.find_all("tr")
    d: dict = {}

    def rv(*kws: str) -> float | None:
        return _find_row_value(rows, *kws)

    d["net_premiums_written"]   = rv("net premiums written")
    d["net_premiums_earned"]    = rv("net premiums earned")
    d["net_income"]             = rv("net income")
    d["combined_ratio"]         = rv("combined ratio")
    d["avg_diluted_equivalent_shares"] = _find_row_value_multi(rows, [
        ("average diluted equivalent common shares",),
        ("average diluted equivalent shares",),
        ("average diluted",),
        ("diluted equivalent shares",),
    ])
    d["total_net_realized_gains"] = _find_row_value_multi(rows, [
        ("total pretax net realized",),
        ("pretax net realized",),
        ("net realized gains",),
    ])
    _eps_raw = _find_row_value_multi(rows, [
        ("per share available to common",),
        ("per share",),
    ])
    if _eps_raw is not None and abs(_eps_raw) < 50:
        d["eps_diluted"] = _eps_raw

    return {k: v for k, v in d.items() if v is not None}


# Maximum plausible single-month repurchase for PGR in $M.
# Set to $2B to accommodate the Oct-2004 ASR (~$1.49B), which is the
# largest single event in PGR's buyback history.
_MAX_REPURCHASE_DOLLARS_M: float = 2_000.0


def _resolve_share_count(
    raw: float,
    avg_cost: float | None,
    month_end: str,
) -> float:
    """Normalize a raw shares-repurchased figure to millions of shares.

    PGR's filings use three formats across their history:
      pre-2023-08 : decimal millions  (e.g. "0.21"    → 0.21M shares)
      post-2023-08 large : whole-share count ≥ 1,000
                           (e.g. "46,822"  → 0.046822M)
      post-2023-08 small : plain integer in *thousands* of shares
                           (e.g. "51"      → 0.051M)

    Resolution strategy (in priority order):
      1. raw ≥ 1,000  →  always a whole-share count; divide by 1,000,000.
      2. When avg_cost is available, test two interpretations and select the
         one whose implied repurchase dollar amount falls within the plausible
         range [0, _MAX_REPURCHASE_DOLLARS_M]:
           thousands : (raw / 1,000) × avg_cost   [$M]
           millions  :  raw          × avg_cost   [$M]
         If exactly one interpretation is in-range, use it.
      3. Date + integer fallback (used when both or neither are in-range):
         post-2023-08 plain integers are in thousands of shares; pre-2023-08
         decimal values (e.g. "0.21", "16.9") are not integers and are
         returned unchanged (already in millions).

    Args:
        raw:       Parsed numeric value from the filing.
        avg_cost:  Average cost per share ($), or None if not yet parsed.
        month_end: ISO date string for the reporting period ("YYYY-MM-DD").

    Returns:
        shares_repurchased in millions of shares.
    """
    if raw >= 1_000:
        return raw / 1_000_000.0

    if avg_cost is not None and avg_cost > 0.0:
        implied_thousands = (raw / 1_000.0) * avg_cost
        implied_millions  = raw * avg_cost

        thousands_ok = 0.0 <= implied_thousands <= _MAX_REPURCHASE_DOLLARS_M
        millions_ok  = 0.0 <= implied_millions  <= _MAX_REPURCHASE_DOLLARS_M

        if thousands_ok and not millions_ok:
            return raw / 1_000.0
        if millions_ok and not thousands_ok:
            return raw
        # Both or neither in-range: fall through to the date heuristic.

    # Date + integer heuristic: post-2023-08 small integers are in thousands.
    # Pre-2023-08 decimal values (e.g. "0.21", "16.9") fail is_integer() and
    # correctly fall through to return raw unchanged.
    if float(raw).is_integer() and raw >= 1 and month_end >= "2023-08-01":
        return raw / 1_000.0
    return raw


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
    item_code: str = "7.01",
) -> dict[str, Any] | None:
    """Parse a PGR 8-K HTML exhibit for operating metrics.

    PGR files the same operating-metrics supplement each month regardless of
    whether it is a quarter-end month.  Non-quarter-end months use item 7.01
    (Regulation FD); quarter-end months (March, June, September, December) use
    item 2.02 (Results of Operations).  Both formats contain the same tables:
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

    The filing date is used to derive ``month_end``: PGR files its supplement
    in the first 3 weeks of the following month, so the data period is the
    month immediately prior to the filing date.

    Args:
        html: Raw HTML text of the 8-K exhibit.
        filing_date: ISO date string (``"YYYY-MM-DD"``).
        item_code: EDGAR item code, either ``"7.01"`` (monthly Reg FD) or
            ``"2.02"`` (quarterly earnings).  Controls the ``filing_type``
            field in the returned record.

    Returns:
        Dict with all parseable field values set, None placeholders for derived
        fields.  Returns ``None`` if neither combined_ratio nor pif_total can
        be extracted (filing is likely not an operating-metrics supplement).
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
        "filing_type": "quarterly_earnings" if item_code == "2.02" else "monthly_results",
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

    # -----------------------------------------------------------------------
    # BeautifulSoup table-classification dispatch
    # -----------------------------------------------------------------------
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    for tbl in tables:
        rows = tbl.find_all("tr")
        n = len(rows)
        if n < 2:
            continue
        tbl_text = _table_full_text(tbl)

        # Balance sheet (check first — its text also matches eps/investment tests)
        if _is_balance_sheet_table(tbl_text):
            bs = _extract_balance_sheet(tbl, month_end)
            for k, v in bs.items():
                if table_metrics.get(k) is None:
                    table_metrics[k] = v
            continue

        # EPS / Comprehensive income
        if _is_eps_table(tbl_text) and not _is_income_stmt_table(tbl_text, n):
            eps = _extract_eps_comprehensive(tbl)
            for k, v in eps.items():
                if table_metrics.get(k) is None:
                    table_metrics[k] = v
            # Some releases put the investment results under the EPS rows.
            if "fully taxable equivalent" in tbl_text:
                for k, v in _extract_investment_returns(tbl).items():
                    if table_metrics.get(k) is None:
                        table_metrics[k] = v
            continue

        # Investment returns (some old exhibits embed these inside the IS table;
        # if so, fall through so IS fields are also extracted below)
        if _is_investment_returns_table(tbl_text):
            inv = _extract_investment_returns(tbl)
            for k, v in inv.items():
                if table_metrics.get(k) is None:
                    table_metrics[k] = v
            if not _is_income_stmt_table(tbl_text, n):
                continue

        # Skip year-to-date / full-year income and segment tables
        if _is_excluded_table(tbl):
            continue

        # Income statement (current-month only; first-wins so current-month table wins)
        if _is_income_stmt_table(tbl_text, n):
            for k, v in _extract_income_stmt(tbl).items():
                if table_metrics.get(k) is None:
                    table_metrics[k] = v
            # EPS may also be embedded at the bottom of the IS table
            if "average shares outstanding" in tbl_text or "per share" in tbl_text:
                eps = _extract_eps_comprehensive(tbl)
                for k, v in eps.items():
                    if table_metrics.get(k) is None:
                        table_metrics[k] = v
            continue

        # Policies in force
        if _is_policies_table(tbl_text, n):
            pif = _extract_policies_in_force(tbl)
            for k, v in pif.items():
                if table_metrics.get(k) is None:
                    table_metrics[k] = v
            continue

        # Summary table (top of release — has combined ratio, NPW, net income)
        if _is_summary_table(tbl_text, n):
            summ = _extract_summary_table(tbl)
            for k, v in summ.items():
                if table_metrics.get(k) is None:
                    table_metrics[k] = v
            continue

        # Segment table
        if _is_segment_table(tbl_text, n):
            seg = _extract_segment(tbl)
            for k, v in seg.items():
                if table_metrics.get(k) is None:
                    table_metrics[k] = v
            continue

    # -----------------------------------------------------------------------
    # Fallback sweep: if key fields are still missing after the typed dispatch,
    # scan all tables with a simple row-label pass.  This handles minimal test
    # fixtures and any real filing whose data falls in an unclassified table.
    # -----------------------------------------------------------------------
    # Policies in force embedded in a larger, unclassified table
    # (e.g. 2009-04, 2009-05, 2010-08).
    if table_metrics["pif_direct_auto"] is None:
        for tbl in tables:
            if "policies in force" not in _table_full_text(tbl):
                continue
            for k, v in _extract_policies_in_force(tbl).items():
                if table_metrics.get(k) is None:
                    table_metrics[k] = v
            if table_metrics["pif_direct_auto"] is not None:
                break

    _needs_fallback = any(
        table_metrics[f] is None
        for f in ("combined_ratio", "pif_total", "net_premiums_written",
                  "investment_income", "book_value_per_share", "eps_basic",
                  "shares_repurchased", "avg_cost_per_share")
    )
    if _needs_fallback:
        for tbl in tables:
            roe_context = False
            for row in tbl.find_all("tr"):
                ctexts = _cells_text(row)
                if not ctexts:
                    continue
                label = ctexts[0].lower()
                if "return on" in label or "trailing 12" in label:
                    roe_context = True
                val = _get_first_numeric(ctexts[1:])
                nums = _row_numbers(ctexts[1:], decimals_only="ratio" in label)
                ratio_values = [value for value in nums if 60.0 <= value <= 140.0]

                if (
                    table_metrics["loss_lae_ratio"] is None
                    and nums
                    and ("loss/lae ratio" in label or "loss ratio" in label)
                ):
                    table_metrics["loss_lae_ratio"] = _select_companywide_ratio(nums)

                if (
                    table_metrics["expense_ratio"] is None
                    and nums
                    and "expense ratio" in label
                    and "net catastrophe" not in label
                ):
                    table_metrics["expense_ratio"] = _select_companywide_ratio(nums)

                if table_metrics["combined_ratio"] is None and (
                    "combined ratio" in label
                    or "combined loss and expense ratio" in label
                ):
                    if ratio_values:
                        lr = table_metrics.get("loss_lae_ratio")
                        er = table_metrics.get("expense_ratio")
                        expected = lr + er if lr is not None and er is not None else None
                        table_metrics["combined_ratio"] = _select_companywide_ratio(
                            ratio_values,
                            expected,
                        )

                if table_metrics["pif_total"] is None and val is not None and (
                    "policies in force" in label or "total pif" in label
                ):
                    normed = _normalise_pif_value(val)
                    if normed is not None and normed >= 10_000:
                        table_metrics["pif_total"] = normed

                if val is None:
                    continue

                if table_metrics["net_premiums_written"] is None and "net premiums written" in label:
                    if 100.0 <= val <= 30_000.0:
                        table_metrics["net_premiums_written"] = val

                if table_metrics["investment_income"] is None and "investment income" in label:
                    if 5.0 <= val <= 3_000.0:
                        table_metrics["investment_income"] = val

                if table_metrics["book_value_per_share"] is None and "book value per" in label and "share" in label:
                    if 1.0 <= val <= 500.0:
                        table_metrics["book_value_per_share"] = val

                if table_metrics["eps_basic"] is None and (
                    "earnings per share" in label or ("per share" in label and "earnings" in label)
                ):
                    if 0.0 <= val <= 50.0:
                        table_metrics["eps_basic"] = val

                if table_metrics["avg_cost_per_share"] is None and "per share" in label and (
                    "average" in label or "avg" in label
                ):
                    if 5.0 <= val <= 2_000.0:
                        table_metrics["avg_cost_per_share"] = val

                if table_metrics["shares_repurchased"] is None and "shares repurchased" in label:
                    table_metrics["shares_repurchased"] = _resolve_share_count(
                        val, table_metrics.get("avg_cost_per_share"), month_end,
                    )

                if table_metrics["avg_shares_basic"] is None and "average" in label and "basic" in label and "shares" in label:
                    if 100.0 <= val <= 5_000.0:
                        table_metrics["avg_shares_basic"] = val

                if table_metrics["avg_shares_diluted"] is None and (
                    "total average equivalent" in label or ("average" in label and "diluted" in label and "shares" in label)
                ):
                    if 100.0 <= val <= 5_000.0:
                        table_metrics["avg_shares_diluted"] = val

                # ROE: "Net income" / "Comprehensive income" rows with a "%"
                # cell, only under a "return on …" / "trailing 12 …" header.
                # (A bare "Net income" row with a %-change column is not ROE:
                # 2007-08 stored 76.9 that way.)
                has_pct = any("%" in c for c in ctexts[1:])
                if roe_context and has_pct and 0.0 < val < 100.0:
                    if table_metrics["roe_net_income_trailing_12m"] is None and (
                        "net income" in label and "comprehensive" not in label
                    ):
                        table_metrics["roe_net_income_trailing_12m"] = val
                    if table_metrics["roe_comprehensive_trailing_12m"] is None and (
                        "comprehensive income" in label or "comprehensive loss" in label
                    ):
                        table_metrics["roe_comprehensive_trailing_12m"] = val

    # Text-mode CR fallback: strips all HTML tags first so label/value pairs
    # that span separate table rows (a common quarterly-earnings layout) are
    # visible as plain text.
    #
    # Strategy:
    #   1. Find ALL occurrences of "combined ratio" in the stripped text.
    #   2. For each occurrence, try a "near window" (first 100 chars after the
    #      label): if exactly 1–2 values appear there, the first is the answer
    #      (handles narrative prose: "combined ratio was 89.9%, vs 97.2% PY").
    #   3. If the near window is inconclusive (0 or 3+ values), use a wide
    #      window (−200 / +600 chars) to capture tabular layouts where values
    #      may be several rows below the label.  Apply column-position logic:
    #      ≥7 values → nums[-2] (quarterly tables append a prior-year company
    #      total as the last column); fewer → nums[-1].
    #   4. Cross-validate the candidate against loss_lae+expense if both are
    #      available; if the delta >5 pp, search the value list for a better
    #      match (handles 9-column tables where nums[-2] is a YTD total).
    #
    # Runs whenever the table scanner found nothing.
    combined_ratio: float | None = None
    if table_metrics["combined_ratio"] is None:
        text_lower = text.lower()
        cr_offsets = [
            m.start()
            for m in re.finditer(
                r"combined\s+(?:ratio|loss\s+and\s+expense\s+ratio)",
                text_lower,
            )
        ]
        best_cr_candidate: float | None = None
        best_cr_vals: list[float] = []

        for idx in cr_offsets:
            # --- Near window: narrative or single-value inline layout ---
            near = text[idx : min(len(text), idx + 100)]
            near_vals = [
                float(v)
                for v in re.findall(r"\b(\d{2,3}\.\d{1,2})\b", near)
                if 60.0 <= float(v) <= 140.0
            ]
            if 1 <= len(near_vals) <= 2:
                # Unambiguous: first value is the current-period CR
                best_cr_candidate = near_vals[0]
                best_cr_vals = near_vals
                break  # Clean near match; stop examining further occurrences

            # --- Wide window: tabular layout, values may be far from label ---
            start = max(0, idx - 200)
            end = min(len(text), idx + 600)
            wide_vals = [
                float(v)
                for v in re.findall(r"\b(\d{2,3}\.\d{1,2})\b", text[start:end])
                if 60.0 <= float(v) <= 140.0
            ]
            if len(wide_vals) > len(best_cr_vals):
                best_cr_vals = wide_vals

        # Apply column-position logic if no near match was found
        if best_cr_candidate is None and best_cr_vals:
            if len(best_cr_vals) >= 7:
                best_cr_candidate = best_cr_vals[-2]
            else:
                best_cr_candidate = best_cr_vals[-1]

        # Cross-validate against sub-ratios when both are available
        if best_cr_candidate is not None:
            lr = table_metrics.get("loss_lae_ratio")
            er = table_metrics.get("expense_ratio")
            if (
                lr is not None
                and er is not None
                and abs(best_cr_candidate - (lr + er)) > 5.0
            ):
                expected = lr + er
                # Tight match first (±0.5pp) to avoid picking segment values that
                # happen to fall within the loose 5pp window.
                found = False
                for tol in (0.5, 5.0):
                    for v in best_cr_vals:
                        if abs(v - expected) <= tol:
                            best_cr_candidate = v
                            found = True
                            break
                    if found:
                        break

            combined_ratio = best_cr_candidate

    # Narrative fallback for the fixed-income duration, which pre-2006
    # releases give only in the commentary ("the duration was 3.0 years").
    if table_metrics["fixed_income_duration"] is None:
        m = re.search(r"duration (?:was|of) (\d{1,2}\.\d) years", text, re.IGNORECASE)
        if m and 0.5 <= float(m.group(1)) <= 10.0:
            table_metrics["fixed_income_duration"] = float(m.group(1))

    # Sub-ratio fallback: combined_ratio = loss/LAE + expense by definition.
    # When all direct extraction paths fail but both sub-ratios were parsed
    # from the same exhibit, compute the combined ratio rather than leave it NULL.
    if table_metrics["combined_ratio"] is None and combined_ratio is None:
        _lr = table_metrics.get("loss_lae_ratio")
        _er = table_metrics.get("expense_ratio")
        if _lr is not None and _er is not None and 60.0 <= _lr + _er <= 140.0:
            combined_ratio = round(_lr + _er, 1)
            log.debug(
                "CR derived from sub-ratios: %.1f + %.1f = %.1f (filing %s)",
                _lr, _er, combined_ratio, filing_date,
            )

    # -----------------------------------------------------------------------
    # Derived: total_liabilities = assets - equity if not explicitly stated
    # -----------------------------------------------------------------------
    if table_metrics.get("total_liabilities") is None:
        ta = table_metrics.get("total_assets")
        se = table_metrics.get("shareholders_equity")
        if ta is not None and se is not None:
            table_metrics["total_liabilities"] = round(ta - se, 1)

    # -----------------------------------------------------------------------
    # F18: some releases (e.g. Dec-2004) have no total-equity line.  Use
    # book value per share x common shares outstanding (common equity; it
    # excludes the 2018-03..2024-01 preferred stock) or leave it NULL.
    # -----------------------------------------------------------------------
    derived_fields: list[str] = []
    if table_metrics.get("shareholders_equity") is None:
        bvps = table_metrics.get("book_value_per_share")
        shares = table_metrics.get("common_shares_outstanding")
        if bvps is not None and shares is not None and bvps > 0 and shares > 0:
            table_metrics["shareholders_equity"] = round(bvps * shares, 1)
            derived_fields.append("shareholders_equity")

    # -----------------------------------------------------------------------
    # Normalise pif_total to canonical thousands unit
    # -----------------------------------------------------------------------
    if table_metrics.get("pif_total") is not None:
        table_metrics["pif_total"] = _normalise_pif_value(table_metrics["pif_total"])

    # -----------------------------------------------------------------------
    # Usability gate: require at least combined_ratio or pif_total
    # -----------------------------------------------------------------------
    if (
        table_metrics["combined_ratio"] is None
        and combined_ratio is None
        and table_metrics["pif_total"] is None
    ):
        return None  # Not a monthly supplement — skip

    table_metrics["month_end"] = month_end

    # -----------------------------------------------------------------------
    # Return the unified dict with all parsed fields plus derived placeholders
    # -----------------------------------------------------------------------
    return {
        "month_end": month_end,
        "filing_date": filing_date,
        "filing_type": table_metrics["filing_type"],
        "combined_ratio": (
            table_metrics["combined_ratio"]
            if table_metrics["combined_ratio"] is not None
            else combined_ratio
        ),
        "pif_total": table_metrics["pif_total"],
        "net_premiums_written": table_metrics["net_premiums_written"],
        "net_premiums_earned": table_metrics["net_premiums_earned"],
        "net_income": table_metrics["net_income"],
        "eps_diluted": table_metrics["eps_diluted"],
        "avg_diluted_equivalent_shares": table_metrics["avg_diluted_equivalent_shares"],
        "investment_income": table_metrics["investment_income"],
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
        "eps_basic": table_metrics["eps_basic"],
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
        "npw_agency": table_metrics["npw_agency"],
        "npw_direct": table_metrics["npw_direct"],
        "npw_property": table_metrics["npw_property"],
        "npw_commercial": table_metrics["npw_commercial"],
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
        "shares_repurchased": table_metrics["shares_repurchased"],
        "avg_cost_per_share": table_metrics["avg_cost_per_share"],
        "book_value_per_share": table_metrics["book_value_per_share"],
        "roe_net_income_trailing_12m": table_metrics["roe_net_income_trailing_12m"],
        "roe_comprehensive_trailing_12m": table_metrics["roe_comprehensive_trailing_12m"],
        "debt_to_total_capital": table_metrics["debt_to_total_capital"],
        "fixed_income_duration": table_metrics["fixed_income_duration"],
        "fte_return_fixed_income": table_metrics["fte_return_fixed_income"],
        "fte_return_common_stocks": table_metrics["fte_return_common_stocks"],
        "fte_return_total_portfolio": table_metrics["fte_return_total_portfolio"],
        "investment_book_yield": table_metrics["investment_book_yield"],
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
        # Fields computed from other parsed fields rather than read directly.
        "derived_fields": derived_fields,
    }


# ---------------------------------------------------------------------------
# Derived field computation
# ---------------------------------------------------------------------------

def _prior_year_key(month_end: str) -> str:
    """Return the month_end string for the same month one year prior.

    Period arithmetic on the calendar month (F11): 2025-02-28 maps to
    2024-02-29, and 2024-02-29 maps to 2023-02-28.
    """
    return pgr_edgar_derived.prior_year_month_end(month_end)


# Fields that ``_compute_derived_fields`` owns.  They are always recomputed
# from the parsed fields, never read from a filing or the CSV.
DERIVED_FIELDS: tuple[str, ...] = (
    "pif_total",
    "pif_total_personal_lines",
    "pif_growth_yoy",
    "gainshare_estimate",
    "channel_mix_agency_pct",
    "underwriting_income",
    "npw_growth_yoy",
    "unearned_premium_growth_yoy",
)


def _compute_derived_fields(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute all derived fields for the full 8-K time series, in place.

    Uses the single definitions in ``src.processing.pgr_edgar_derived``:

      - ``pif_total``                — agency auto + direct auto + special lines
        + commercial lines (property excluded); None unless all are present
      - ``pif_total_personal_lines`` — agency auto + direct auto + special lines
      - ``pif_growth_yoy``           — calendar-month YoY of ``pif_total``
      - ``gainshare_estimate``       — 0.5 × CR score + 0.5 × PIF score (0–2);
        None unless both CR and PIF growth exist
      - ``channel_mix_agency_pct``   — Agency NPW / (Agency + Direct NPW)
      - ``underwriting_income``      — NPE × (1 − CR/100)
      - ``npw_growth_yoy``           — calendar-month YoY of total NPW
      - ``unearned_premium_growth_yoy`` — calendar-month YoY of unearned premiums

    YoY values are None when the same month one year earlier is missing, so
    a gap never produces a 13-month change (F16).  Every derived field is
    assigned (possibly None), so recomputing over the full table clears
    stale values.

    Args:
        records: Record dicts with a ``month_end`` key (any order).

    Returns:
        The same list with all derived fields assigned.
    """
    for rec in records:
        rec["pif_total"] = pgr_edgar_derived.pif_sum(
            rec, pgr_edgar_derived.PIF_TOTAL_COMPONENTS
        )
        rec["pif_total_personal_lines"] = pgr_edgar_derived.pif_sum(
            rec, pgr_edgar_derived.PIF_PERSONAL_LINES_COMPONENTS
        )

    def _yoy(field: str) -> dict[pd.Period, float | None]:
        return pgr_edgar_derived.yoy_growth_by_period(
            {r["month_end"]: r.get(field) for r in records}
        )

    pif_yoy = _yoy("pif_total")
    npw_yoy = _yoy("net_premiums_written")
    unprem_yoy = _yoy("unearned_premiums")

    for rec in records:
        period = pgr_edgar_derived.month_key(rec["month_end"])
        rec["pif_growth_yoy"] = pif_yoy.get(period)
        rec["npw_growth_yoy"] = npw_yoy.get(period)
        rec["unearned_premium_growth_yoy"] = unprem_yoy.get(period)

        cr = rec.get("combined_ratio")
        rec["gainshare_estimate"] = pgr_edgar_derived.gainshare_estimate(
            cr, rec["pif_growth_yoy"]
        )

        # channel_mix_agency_pct = npw_agency / (npw_agency + npw_direct)
        rec["channel_mix_agency_pct"] = None
        npw_ag = rec.get("npw_agency")
        npw_di = rec.get("npw_direct")
        if npw_ag is not None and npw_di is not None and npw_ag + npw_di > 0.0:
            rec["channel_mix_agency_pct"] = npw_ag / (npw_ag + npw_di)

        # underwriting_income = net_premiums_earned × (1 − CR / 100)
        rec["underwriting_income"] = None
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

_PIF_TOTAL_FLOOR: int = 5_000


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
         If parsed pif_total < 5,000, likely a mis-parse; set to None.
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

    # PIF floor (thousands of policies).  PGR had ~9,000K policies in 2004,
    # so the floor only catches unit mis-parses (e.g. a value in millions).
    pif = record.get("pif_total")
    if pif is not None and pif < _PIF_TOTAL_FLOOR:
        log.warning(
            "VALIDATION: pif_total=%.0f < %d floor in %s. Setting None.",
            pif, _PIF_TOTAL_FLOOR, accession,
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

    # shares_repurchased plausibility: flag unit-scaling anomalies.
    # PGR's max single-month buyback has historically been ~17M shares (2004 ASR);
    # a value > 25M almost certainly means a raw dollar amount ($M) was stored instead
    # of a share count (millions).  A value so tiny that the implied dollar repurchase
    # is < $10K is also likely a 1000x under-scale (thousands stored as units).
    sr = record.get("shares_repurchased")
    acp = record.get("avg_cost_per_share")
    if sr is not None and sr > 25.0:
        log.warning(
            "VALIDATION: shares_repurchased=%.4f > 25M threshold in %s — "
            "possible dollar-amount mis-parse (avg_cost=%.2f). Setting None.",
            sr, accession, acp or 0.0,
        )
        record["shares_repurchased"] = None
    elif sr is not None and sr > 0.0 and acp is not None and acp > 0.0:
        implied_dollars = sr * 1_000_000.0 * acp
        if implied_dollars < 1_000.0:
            log.warning(
                "VALIDATION: shares_repurchased=%.6f × avg_cost=%.2f implies "
                "$%.0f repurchase — likely 1000x under-scale in %s. Setting None.",
                sr, acp, implied_dollars, accession,
            )
            record["shares_repurchased"] = None
        elif implied_dollars > 750_000_000.0:
            log.warning(
                "VALIDATION: shares_repurchased=%.4f × avg_cost=%.2f implies "
                "$%.0fM repurchase — unusually large for a single month in %s. "
                "Retaining value but flagging for review.",
                sr, acp, implied_dollars / 1_000_000.0, accession,
            )

    return record


# ---------------------------------------------------------------------------
# Main fetch-and-upsert logic
# ---------------------------------------------------------------------------

_RECORD_META_KEYS: frozenset[str] = frozenset({
    "month_end", "filing_date", "filing_type", "accession_number",
    "document_url", "fetched_at", "derived_fields",
})


def _completeness_score(rec: dict[str, Any]) -> int:
    """Count non-None fields; combined_ratio presence adds a large bonus."""
    base = sum(
        1 for k, v in rec.items() if v is not None and k not in _RECORD_META_KEYS
    )
    # Heavily weight having a combined_ratio — it's the most critical field.
    if rec.get("combined_ratio") is not None:
        base += 100
    return base


def parse_filing(filing: dict[str, Any]) -> dict[str, Any] | None:
    """Fetch and parse one candidate 8-K; return a validated record or None.

    Every exhibit of the filing is tried, EX-99 first, and the most complete
    parse wins (stopping at the first one with a combined ratio).  A 9.01-only
    filing is considered only when its index lists an EX-99 exhibit.

    The record carries ``accession_number`` (dashed), ``document_url`` and
    ``fetched_at`` for the provenance table.
    """
    accession = filing["accession_number"]
    accession_dashed = filing["accession_dashed"]
    filing_date = filing["filing_date"]
    item_code = filing.get("item_code", "7.01")
    log.debug("Processing %s (filed %s, item %s) …", accession, filing_date, item_code)

    doc_urls = _get_all_filing_doc_urls(
        accession, accession_dashed, require_ex99=item_code == "9.01",
    )
    if not doc_urls:
        log.debug("No HTML exhibit found for %s — skipping.", accession)
        return None

    parsed: dict[str, Any] | None = None
    for url_idx, doc_url in enumerate(doc_urls):
        try:
            resp = _get(doc_url)
            html = resp.text
            if doc_url.lower().endswith(".txt"):
                html = _text_exhibit_to_html(html)
            candidate = _parse_html_exhibit(html, filing_date, item_code=item_code)
        except Exception as exc:
            log.warning(
                "Failed to fetch/parse exhibit %d for %s (%s): %r",
                url_idx + 1, accession, doc_url, exc,
            )
            continue

        if candidate is None:
            log.debug(
                "Exhibit %d for %s yielded no parseable data (%s).",
                url_idx + 1, accession, doc_url,
            )
            continue
        candidate["document_url"] = doc_url
        candidate["fetched_at"] = getattr(resp, "fetched_at", None)

        if parsed is None or _completeness_score(candidate) > _completeness_score(parsed):
            parsed = candidate

        # Stop as soon as we have a combined_ratio — core field satisfied.
        if parsed.get("combined_ratio") is not None:
            break

    if parsed is None:
        log.debug("No parseable data in %s (filed %s).", accession, filing_date)
        return None

    # v7.2: cross-validate parsed fields; nullify inconsistent ones.
    parsed = _validate_parsed_record(parsed, filing_date, accession)
    parsed["accession_number"] = accession_dashed

    # If validation nullified combined_ratio but sub-ratios survived,
    # recover it from loss/LAE + expense (combined ratio = their sum by definition).
    if parsed.get("combined_ratio") is None:
        _lr = parsed.get("loss_lae_ratio")
        _er = parsed.get("expense_ratio")
        if _lr is not None and _er is not None and 60.0 <= _lr + _er <= 140.0:
            parsed["combined_ratio"] = round(_lr + _er, 1)
            log.info(
                "CR recovered from sub-ratios for %s: %.1f + %.1f = %.1f",
                accession, _lr, _er, _lr + _er,
            )

    # If validation nullified both core fields, skip this filing.
    if parsed["combined_ratio"] is None and parsed["pif_total"] is None:
        log.debug("Validation nullified both CR and PIF for %s — skipping.", accession)
        return None

    log.info(
        "Parsed %s  month_end=%-12s  CR=%-6s  PIF=%s  item=%s",
        accession,
        parsed["month_end"],
        f"{parsed['combined_ratio']:.1f}" if parsed["combined_ratio"] else "n/a",
        f"{parsed['pif_total']:,.0f}" if parsed["pif_total"] else "n/a",
        item_code,
    )
    return parsed


def fetch_and_upsert(
    conn: sqlite3.Connection,
    backfill_years: int = 2,
    dry_run: bool = False,
) -> int:
    """Fetch PGR 8-K operating metrics and upsert them to the DB.

    Workflow:
      1. Compute cutoff date (today minus backfill_years, floored at
         BACKFILL_EARLIEST_DATE).
      2. Fetch all 8-K (items 7.01/2.02) filings from EDGAR submissions (with
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
        log.info("No 8-K (items 7.01/2.02) filings found in the requested date range.")
        return 0

    records: list[dict[str, Any]] = []
    parse_errors = 0

    for filing in filings:
        try:
            parsed = parse_filing(filing)
        except Exception as exc:
            parse_errors += 1
            log.exception(
                "SKIP %s (filed %s) due to parse failure. Error=%r",
                filing["accession_number"],
                filing["filing_date"],
                exc,
            )
            continue
        if parsed is not None:
            records.append(parsed)

    if parse_errors > 0:
        log.warning("%d filing(s) skipped due to parse errors.", parse_errors)

    if not records:
        log.info("No records to upsert.")
        return 0

    deduped = select_monthly_releases(records)

    # Coverage report
    n_total = len(deduped)

    def _cov(field: str) -> str:
        n = sum(1 for r in deduped if r.get(field) is not None)
        return f"{n}/{n_total}"

    log.info(
        "Coverage  combined_ratio=%s  pif_total=%s  npw=%s  npw_agency=%s  "
        "investment_income=%s  bvps=%s  date_range=%s->%s",
        _cov("combined_ratio"),
        _cov("pif_total"),
        _cov("net_premiums_written"),
        _cov("npw_agency"),
        _cov("investment_income"),
        _cov("book_value_per_share"),
        deduped[0]["month_end"],
        deduped[-1]["month_end"],
    )

    if dry_run:
        log.info("Dry run — skipping DB write (%d rows would be upserted).", n_total)
        return 0

    months_before = conn.execute(
        "SELECT COUNT(*) FROM pgr_edgar_monthly"
    ).fetchone()[0]
    # Every parse is kept, append-only, before the monthly table is touched.
    n_raw = db_client.record_pgr_edgar_raw(conn, deduped, PARSER_VERSION)
    n = db_client.upsert_pgr_edgar_monthly(conn, deduped)
    recompute_derived_fields(conn)
    months_after = conn.execute(
        "SELECT COUNT(*) FROM pgr_edgar_monthly"
    ).fetchone()[0]
    log.info(
        "Recorded %d raw values; upserted %d rows to pgr_edgar_monthly "
        "(%d new months).",
        n_raw, n, months_after - months_before,
    )

    # v7.2: warn when no new months were added to alert on format changes.
    if months_after == months_before:
        log.warning(
            "NOTE: No new months added this run (table has %d months). If this "
            "persists, check whether PGR has changed its 8-K filing format.",
            months_after,
        )

    return n


def select_monthly_releases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one parsed record per ``month_end``: the month's release.

    Prefers a record with a combined ratio, then the earliest filing (the
    monthly release; a later filing for the same month is a quarterly letter
    or an unrelated 8-K whose numbers can parse as a partial record), then
    the most complete.  Returns records sorted by ``month_end``.
    """
    def _rank(rec: dict[str, Any]) -> tuple[int, str, int]:
        return (
            0 if rec.get("combined_ratio") is not None else 1,
            rec.get("filing_date") or "",
            -_completeness_score(rec),
        )

    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        me = rec["month_end"]
        if me not in seen or _rank(rec) < _rank(seen[me]):
            seen[me] = rec
    return sorted(seen.values(), key=lambda r: r["month_end"])


def recompute_derived_fields(conn: sqlite3.Connection) -> int:
    """Recompute every derived column over the whole ``pgr_edgar_monthly`` table.

    YoY windows need the full history, so derived fields are never computed
    on a partial fetch window (F16/F33).  Returns the number of rows updated.
    """
    cur = conn.execute("SELECT * FROM pgr_edgar_monthly ORDER BY month_end")
    names = [d[0] for d in cur.description]
    records = [dict(zip(names, row)) for row in cur.fetchall()]
    if not records:
        return 0
    _compute_derived_fields(records)
    sql = (
        "UPDATE pgr_edgar_monthly SET "
        + ", ".join(f"{f} = ?" for f in DERIVED_FIELDS)
        + " WHERE month_end = ?"
    )
    conn.executemany(
        sql,
        [tuple(rec[f] for f in DERIVED_FIELDS) + (rec["month_end"],) for rec in records],
    )
    conn.commit()
    return len(records)


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
    ``month_end`` (last calendar day of that month, ``"YYYY-MM-DD"``) and maps
    the CSV columns into the DB schema.

    Only months missing from the table are inserted, so re-running it on a
    live-populated DB changes nothing (F33).  Inserted values are recorded in
    ``pgr_edgar_monthly_raw`` with ``method='csv'``.  Derived fields are then
    recomputed over the whole table.

    No network calls are made.  The regular ``fetch_and_upsert`` EDGAR fetch
    covers recent months not yet in the CSV.

    Args:
        conn: Open SQLite connection.
        csv_path: Path to ``pgr_edgar_cache.csv``.
        dry_run: If True, parse but skip the DB write.

    Returns:
        Number of months inserted (0 for dry runs).

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
    # Build records.  Derived fields (YoY growth, Gainshare, PIF totals,
    # channel mix, underwriting income) are not taken from the CSV: they are
    # recomputed over the whole table after the insert, by calendar month (F16).
    # -----------------------------------------------------------------------
    unique_cols = ["month_end"] + list(dict.fromkeys(DIRECT_MAP.values()))
    df_out = df[unique_cols].copy()

    def _nan_to_none(val: Any) -> Any:
        """Convert float NaN to None for SQLite NULL storage."""
        try:
            if val != val:  # NaN check
                return None
        except TypeError:
            pass
        if isinstance(val, str) and val.lower() == "nan":
            return None
        return val

    records_raw: list[dict[str, Any]] = [
        {col: _nan_to_none(row[col]) for col in unique_cols}
        for _, row in df_out.iterrows()
    ]

    log.info(
        "CSV loaded: %d rows  date_range=%s->%s",
        len(records_raw),
        records_raw[0]["month_end"] if records_raw else "n/a",
        records_raw[-1]["month_end"] if records_raw else "n/a",
    )

    if dry_run:
        log.info("Dry run — skipping DB write (%d rows read).", len(records_raw))
        return 0

    # Idempotent against newer rows (F33): only months absent from the table
    # are inserted; rows written by the live parser are never overwritten.
    existing = {
        str(row[0])
        for row in conn.execute("SELECT month_end FROM pgr_edgar_monthly").fetchall()
    }
    new_records = [r for r in records_raw if r["month_end"] not in existing]
    db_client.record_pgr_edgar_raw(
        conn, new_records, parser_version=os.path.basename(csv_path), method="csv",
    )
    n = db_client.upsert_pgr_edgar_monthly(conn, new_records, mode="insert_missing")
    recompute_derived_fields(conn)
    log.info(
        "Inserted %d missing months from CSV into pgr_edgar_monthly "
        "(%d months already present were left unchanged).",
        n, len(records_raw) - len(new_records),
    )
    return n


# Column order of data/processed/pgr_edgar_cache.csv.  ``report_period`` is
# ``YYYY-MM``; ``roe_net_income_trailing_12m`` is the DB's ``roe_net_income_ttm``.
EDGAR_CACHE_CSV_COLUMNS: tuple[str, ...] = (
    "report_period",
    "filing_date",
    "filing_type",
    "accession_number",
    "net_premiums_written",
    "net_premiums_earned",
    "combined_ratio",
    "avg_diluted_equivalent_shares",
    "investment_income",
    "total_net_realized_gains",
    "service_revenues",
    "fees_and_other_revenues",
    "total_revenues",
    "losses_lae",
    "policy_acquisition_costs",
    "other_underwriting_expenses",
    "interest_expense",
    "total_expenses",
    "income_before_income_taxes",
    "provision_for_income_taxes",
    "net_income",
    "total_comprehensive_income",
    "eps_basic",
    "eps_diluted",
    "comprehensive_eps_diluted",
    "avg_shares_basic",
    "avg_shares_diluted",
    "loss_lae_ratio",
    "expense_ratio",
    "pif_agency_auto",
    "pif_direct_auto",
    "pif_special_lines",
    "pif_property",
    "pif_total_personal_lines",
    "pif_commercial_lines",
    "pif_total",
    "npw_agency",
    "npw_direct",
    "npw_property",
    "npw_commercial",
    "npe_agency",
    "npe_direct",
    "npe_property",
    "npe_commercial",
    "total_investments",
    "total_assets",
    "loss_lae_reserves",
    "unearned_premiums",
    "debt",
    "total_liabilities",
    "shareholders_equity",
    "common_shares_outstanding",
    "shares_repurchased",
    "avg_cost_per_share",
    "book_value_per_share",
    "roe_net_income_trailing_12m",
    "roe_comprehensive_trailing_12m",
    "debt_to_total_capital",
    "fixed_income_duration",
    "fte_return_fixed_income",
    "fte_return_common_stocks",
    "fte_return_total_portfolio",
    "investment_book_yield",
    "net_unrealized_gains_fixed",
    "weighted_avg_credit_quality",
)


def export_edgar_cache_csv(conn: sqlite3.Connection, csv_path: str) -> int:
    """Write ``pgr_edgar_monthly`` to ``pgr_edgar_cache.csv`` (the committed seed).

    The CSV is a snapshot of the DB table, so re-seeding an empty DB from it
    with ``load_from_csv`` reproduces the table.  Returns the number of rows.
    """
    df = pd.read_sql_query("SELECT * FROM pgr_edgar_monthly ORDER BY month_end", conn)
    df["report_period"] = df["month_end"].str.slice(0, 7)
    df["roe_net_income_trailing_12m"] = df["roe_net_income_ttm"]
    df[list(EDGAR_CACHE_CSV_COLUMNS)].to_csv(csv_path, index=False, float_format="%.6f")
    return len(df)


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
        help=(
            "Parse/read data but do not write to the database. The DB is "
            "opened read-only and migrations are not applied."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        default=None,
        help=(
            "Cache EDGAR responses (filing indexes and exhibits) in DIR. "
            "Filings are immutable; the submissions index is always re-fetched."
        ),
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    if args.cache_dir:
        set_http_cache_dir(args.cache_dir)
    if args.dry_run:
        conn = db_client.get_connection(config.DB_PATH, read_only=True)
    else:
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
