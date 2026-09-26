"""PGR's monthly 8-K operating results from SEC EDGAR (was ``scripts/edgar_8k_fetcher.py``).

Review 2026-09-25, section 5, phase 5 merged the live fetcher script and its
diverged, test-only copy ``src/ingestion/edgar_8k_fetcher.py`` (deleted, with
the legacy ``src/ingestion/pgr_monthly_loader.py``) into:

- ``fetch``: SEC EDGAR HTTP access (rate limit, User-Agent, response cache),
  the submissions index and filing documents;
- ``parse``: the Exhibit 99 parser, record validation and ``parse_filing``;
- ``derive``: derived fields over the whole table;
- ``load``: ``fetch_and_upsert``, the CSV loader ``load_from_csv``, the CSV
  export and the staleness check.

The command-line entry point is ``cli/edgar_monthly_fetch.py``.
"""
