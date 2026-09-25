"""
Rebuild ``fred_macro_monthly`` from raw (unlagged) FRED observations.

Review 2026-09-25 step 3a (WP3; F06, F07, F27). Before this step the
production loader stored FRED values already shifted by their publication
lag, the feature builder lagged them again by row, and the v19 research
loader had added calendar-month-end copies of 64 months per macro series.
In order this script:

  1. applies pending migrations, including ``005_fred_one_row_per_month``
     (one row per series and month, relabelled to the business month-end,
     plus a unique (series, month) index);
  2. refetches every FRED series the table holds from FRED, unlagged:
     the production series (``production_fred_series()``, from 1990-01-01)
     and the v19 research series served by FRED (from 2008-01-01). It uses
     the FRED API when ``FRED_API_KEY`` is set and the public
     ``fredgraph.csv`` endpoint otherwise (same current-vintage values);
  3. replaces every stored row of each refetched series with one row per
     month (the last observation of the month);
  4. prints (and with ``--report`` writes) a per-series diff against the
     table as it was before step 1.

Series that FRED does not serve (``CUSR0000SETE`` from BLS and the Multpl
valuation series) are only relabelled by the migration.

FRED serves only the last three years of ICE BofA series
(``BAMLH0A0HYM2``). When a fetched series starts later than the stored one,
the earlier stored months are kept, un-lagged: the script measures the lag
the legacy loader stored them with on the overlap (stored(M) = raw(M - k))
and shifts them back by k. It refuses if the overlap is shorter than 12
months or matches on fewer than 90 % of them.

Raw responses are cached under ``--cache-dir`` (default ``data/raw/fred``,
gitignored); ``--offline`` uses only the cache, ``--refresh`` ignores it.
Requests are spaced ``--pause`` seconds apart (default 0.5 s).

Always run on a copy first:
    cp data/pgr_financials.db /tmp/copy.db
    python scripts/rebuild_fred_macro.py --db /tmp/copy.db --report /tmp/fred.md

The DB is finalised (WAL folded, DELETE journal mode) so it can be committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client, migration_runner
from src.ingestion.fred_loader import (
    fetch_fred_series,
    fetch_fred_series_csv,
    production_fred_series,
    to_monthly_observations,
)
from src.ingestion.http_utils import build_retry_session
from src.research.v19 import FREDGRAPH_SERIES

logger = logging.getLogger(__name__)

PRODUCTION_START = "1990-01-01"
RESEARCH_START = "2008-01-01"


def rebuild_plan() -> dict[str, str]:
    """Return {series_id: observation_start} for every series to refetch."""
    plan = {sid: PRODUCTION_START for sid in production_fred_series()}
    for sid in FREDGRAPH_SERIES:
        plan.setdefault(sid, RESEARCH_START)
    return plan


def _snapshot(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT series_id, month_end, value FROM fred_macro_monthly", conn
    )
    df["month"] = df["month_end"].str[:7]
    return df


def load_raw_series(
    series_id: str,
    cache_dir: Path,
    offline: bool = False,
    refresh: bool = False,
    pause: float = 0.5,
    session=None,
) -> tuple[pd.DataFrame, str]:
    """Return raw observations for ``series_id`` from cache or FRED.

    Returns the one-column observation frame and a label naming its source.
    """
    cache_path = cache_dir / f"{series_id}.csv"
    meta_path = cache_dir / f"{series_id}.json"
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        source = "cache"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            source = f"cache ({meta.get('source')}, fetched {meta.get('fetched_at')})"
        return cached, source
    if offline:
        raise FileNotFoundError(f"--offline and no cached observations for {series_id}")

    time.sleep(pause)
    if config.FRED_API_KEY:
        df = fetch_fred_series(series_id, observation_start=PRODUCTION_START)
        source = "FRED API"
    else:
        df = fetch_fred_series_csv(
            series_id, observation_start=PRODUCTION_START, session=session
        )
        source = "fredgraph.csv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index_label="date")
    meta_path.write_text(
        json.dumps(
            {
                "series_id": series_id,
                "source": source,
                "fetched_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                "observations": int(df[series_id].notna().sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return df, source


MIN_OVERLAP_MONTHS = 12
MIN_OVERLAP_MATCH = 0.9


def _by_month(series: pd.Series) -> pd.Series:
    out = series.dropna().copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).to_period("M")
    return out[~out.index.duplicated(keep="last")].sort_index()


def measure_stored_lag(
    stored: pd.Series, raw: pd.Series, max_lag: int = 3
) -> tuple[int, float, int]:
    """Return (k, match share, overlap) for stored(M) = raw(M - k).

    Both series are indexed by month. The k with the highest share of
    matching overlapping months wins.
    """
    best = (0, 0.0, 0)
    for lag in range(max_lag + 1):
        shifted = raw.copy()
        shifted.index = shifted.index + lag
        common = stored.index.intersection(shifted.index)
        if len(common) == 0:
            continue
        match = float(
            np.isclose(stored.loc[common], shifted.loc[common], rtol=1e-6, atol=1e-9).mean()
        )
        if match > best[1]:
            best = (lag, match, len(common))
    return best


def recover_truncated_history(
    series_id: str, stored: pd.Series, raw: pd.Series
) -> tuple[pd.Series, str]:
    """Prepend stored months that the fetched history no longer covers.

    ``stored`` and ``raw`` are indexed by month; ``stored`` holds the legacy
    (lagged) values. Returns the combined raw series and a note for the log.
    """
    if stored.empty or raw.empty:
        return raw, ""
    lag, match, overlap = measure_stored_lag(stored, raw)
    recovered = stored[stored.index < raw.index.min() + lag].copy()
    recovered.index = recovered.index - lag
    if recovered.empty:
        return raw, ""
    if overlap < MIN_OVERLAP_MONTHS or match < MIN_OVERLAP_MATCH:
        raise RuntimeError(
            f"{series_id}: fetched history starts {raw.index.min()} but the stored lag "
            f"cannot be established (best k={lag}, {match:.0%} of {overlap} months)"
        )
    note = (
        f"kept {len(recovered)} earlier months {recovered.index.min()}..{recovered.index.max()} "
        f"from the stored table, un-lagged by {lag} (overlap {overlap} months, {match:.0%} match)"
    )
    return pd.concat([recovered, raw]).sort_index(), note


def replace_series(conn, series_id: str, monthly: pd.Series) -> int:
    """Replace every stored row of ``series_id`` with ``monthly``."""
    records = [
        {"series_id": series_id, "month_end": ts.strftime("%Y-%m-%d"), "value": float(v)}
        for ts, v in monthly.dropna().items()
    ]
    conn.execute("DELETE FROM fred_macro_monthly WHERE series_id = ?", (series_id,))
    return db_client.upsert_fred_macro(conn, records)


def diff_tables(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Per-series summary of rows, duplicate months and value changes."""
    rows = []
    for sid in sorted(set(before["series_id"]) | set(after["series_id"])):
        b = before[before["series_id"] == sid]
        a = after[after["series_id"] == sid]
        b_month = b.sort_values("month_end").groupby("month")["value"].last()
        a_month = a.set_index("month")["value"]
        common = b_month.index.intersection(a_month.index)
        delta = (a_month.loc[common] - b_month.loc[common]).abs()
        rows.append(
            {
                "series_id": sid,
                "rows_before": len(b),
                "duplicate_months_before": int(len(b) - b["month"].nunique()),
                "rows_after": len(a),
                "first_month": a_month.index.min() if len(a) else None,
                "last_month_before": b_month.index.max() if len(b) else None,
                "last_month_after": a_month.index.max() if len(a) else None,
                "months_added": int(len(a_month.index.difference(b_month.index))),
                "months_removed": int(len(b_month.index.difference(a_month.index))),
                "months_changed": int((delta > 1e-9).sum()),
                "max_abs_change": float(delta.max()) if len(delta) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    """Render ``df`` as a GitHub markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(
    db_path: str,
    cache_dir: str = os.path.join("data", "raw", "fred"),
    offline: bool = False,
    refresh: bool = False,
    pause: float = 0.5,
    report: str | None = None,
) -> pd.DataFrame:
    """Rebuild ``fred_macro_monthly`` in ``db_path`` and return the diff."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database does not exist: {db_path}")
    conn = db_client.get_connection(db_path)
    try:
        before = _snapshot(conn)
        applied = migration_runner.apply_migrations(conn)
        if applied:
            print(f"Applied migrations: {', '.join(applied)}")

        session = build_retry_session()
        sources: dict[str, str] = {}
        for series_id, start in rebuild_plan().items():
            raw, source = load_raw_series(
                series_id, Path(cache_dir), offline=offline, refresh=refresh,
                pause=pause, session=session,
            )
            raw = raw.loc[raw.index >= pd.Timestamp(start)]
            monthly = _by_month(to_monthly_observations(raw)[series_id])
            stored = _by_month(
                pd.read_sql_query(
                    "SELECT month_end, value FROM fred_macro_monthly WHERE series_id = ?",
                    conn, params=(series_id,), index_col="month_end",
                )["value"]
            )
            monthly, note = recover_truncated_history(series_id, stored, monthly)
            if note:
                source = f"{source}; {note}"
            monthly.index = monthly.index.to_timestamp()
            n = replace_series(conn, series_id, monthly)
            sources[series_id] = source
            print(f"{series_id:<18} {n:>4} months  {source}")

        after = _snapshot(conn)
        dup = after.groupby(["series_id", "month"]).size()
        if (dup > 1).any():
            raise RuntimeError("fred_macro_monthly still has duplicate (series, month) rows")
    finally:
        conn.close()
    db_client.finalize_for_commit(db_path)

    diff = diff_tables(before, after)
    diff["source"] = diff["series_id"].map(sources).fillna("not refetched")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(diff.to_string(index=False))
    if report:
        lines = [
            "# fred_macro_monthly rebuild",
            "",
            f"- Database: `{db_path}`",
            f"- Rows before: {len(before)}; after: {len(after)}",
            f"- Duplicate (series, month) rows before: "
            f"{int(diff['duplicate_months_before'].sum())}; after: 0",
            "",
            _markdown_table(diff),
            "",
        ]
        Path(report).write_text("\n".join(lines), encoding="utf-8")
    return diff


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--db", required=True, help="Database to rebuild (use a copy).")
    parser.add_argument("--cache-dir", default=os.path.join("data", "raw", "fred"))
    parser.add_argument("--offline", action="store_true", help="Use cached responses only.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached responses.")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds between requests.")
    parser.add_argument("--report", default=None, help="Write a markdown diff here.")
    args = parser.parse_args()
    main(
        args.db,
        cache_dir=args.cache_dir,
        offline=args.offline,
        refresh=args.refresh,
        pause=args.pause,
        report=args.report,
    )
