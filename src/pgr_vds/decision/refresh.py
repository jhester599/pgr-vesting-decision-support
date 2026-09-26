"""Data refresh step of the monthly decision run (FRED macro series)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def fetch_fred_step(conn, dry_run: bool = False, skip_fred: bool = False) -> None:
    """Fetch the latest FRED macro series and upsert into the DB."""
    if skip_fred or dry_run:
        if skip_fred:
            logger.info("[FRED] Skipping FRED fetch (--skip-fred).")
        else:
            logger.info("[FRED] Dry run - skipping FRED HTTP calls.")
        return

    from src.ingestion.fred_loader import (
        fetch_all_fred_macro,
        production_fred_series,
        upsert_fred_to_db,
    )

    series_list = production_fred_series()
    logger.info("[FRED] Fetching %s FRED series...", len(series_list))
    try:
        df = fetch_all_fred_macro(series_list)
        n = upsert_fred_to_db(conn, df)
        logger.info("[FRED] %s rows upserted.", n)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[FRED] Fetch failed. Continuing with cached data. Error=%r",
            exc,
        )
