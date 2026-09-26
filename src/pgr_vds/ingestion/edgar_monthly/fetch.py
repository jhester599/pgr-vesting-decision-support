"""SEC EDGAR access for PGR's monthly 8-K releases.

The primary submissions JSON (``CIK0000080661.json``) only contains the
~1,000 most recent filings under ``filings.recent``. Older filings are in the
pagination files listed in ``filings.files``
(``CIK0000080661-submissions-001.json``, ...), which are *flat*: the parallel
arrays sit at the top level. Files whose ``filingTo`` precedes the cutoff are
skipped.

SEC EDGAR rate limits: 10 requests/second max (this module stays at 4);
``User-Agent`` header required (``EDGAR_USER_AGENT``, via
``config.build_edgar_headers``). ``set_http_cache_dir`` caches filing
indexes and exhibits on disk (filings are immutable).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests  # type: ignore[import-untyped]  # no stubs installed

import config

log = logging.getLogger(__name__)


PGR_CIK: str = "CIK0000080661"


PGR_CIK_NUMERIC: int = 80661          # numeric CIK for archive URLs


SUBMISSIONS_BASE_URL: str = "https://data.sec.gov/submissions"


EDGAR_ARCHIVES_URL: str = "https://www.sec.gov/Archives/edgar/data/80661"


# Polite rate limit: at most 4 requests/second, well under SEC's 10 req/s.
_SEC_MIN_INTERVAL_SECONDS: float = 0.25


_last_request_monotonic: float | None = None


# Optional on-disk response cache (set by ``set_http_cache_dir``).  EDGAR
# filings are immutable, so a cached body never goes stale; only the
# submissions index changes, and callers can bypass the cache for it.
_http_cache_dir: str | None = None


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


def get(url: str, retries: int = 3, use_cache: bool = True) -> Any:
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
    resp = get(url, use_cache=False)
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
    ``parse.parse_html_exhibit`` can set ``filing_type`` correctly.

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


def get_all_filing_doc_urls(
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
        resp = get(index_url)
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
    # are tried last; ``parse.parse_filing`` converts them with
    # ``parse.text_exhibit_to_html``.
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


def _get_filing_doc_url(
    accession_number: str,
    accession_dashed: str,
) -> str | None:
    """Return the top-priority HTML exhibit URL for an 8-K filing.

    Thin wrapper around ``get_all_filing_doc_urls`` that returns only the
    first (highest-priority) candidate.  For quarterly 8-Ks you should call
    ``get_all_filing_doc_urls`` directly so all exhibits can be tried.

    Returns:
        Full URL to the primary HTML exhibit, or ``None`` if none found.
    """
    urls = get_all_filing_doc_urls(accession_number, accession_dashed)
    return urls[0] if urls else None
