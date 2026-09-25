"""
Rebuild ``monthly_relative_returns`` from stored prices, dividends and splits.

Review 2026-09-25 step 2 (WP1 + WP4; F03, F05, F08, F22). In order:

  1. seed ``split_history`` from the canonical ``config.KNOWN_SPLITS``
     (adds VOO 2013-10-24 1-for-2 and VGT 2026-04-21 8-for-1);
  2. delete partial-week duplicate bars, keeping the latest bar per
     ticker-ISO-week;
  3. rebuild both horizons with business-month-end windows, replacing every
     stored row of each benchmark/horizon;
  4. write a per-benchmark diff against the table as it was before step 1.

With ``--attribute`` the diff also splits each change into the part caused by
steps 1-2 (data: splits and duplicate bars, legacy windows) and the part
caused by step 3's window change, by recomputing the targets in memory with
the legacy ``t + DateOffset(months=h)`` window end.

Always run on a copy first:
    cp data/pgr_financials.db /tmp/copy.db
    python scripts/rebuild_relative_returns.py --db /tmp/copy.db \\
        --report /tmp/rebuild.md --rows-csv /tmp/rebuild_rows.csv --attribute

No API calls are made. The DB is finalised (WAL folded, DELETE journal mode)
so it can be committed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.processing import multi_total_return
from src.processing.multi_total_return import build_relative_return_targets

logger = logging.getLogger(__name__)

_KEY = ["benchmark", "target_horizon", "date"]


def _snapshot(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT date, benchmark, target_horizon, pgr_return, benchmark_return, "
        "relative_return FROM monthly_relative_returns",
        conn,
    )


def _legacy_window_end(t: pd.Timestamp, months: int) -> pd.Timestamp:
    return pd.Timestamp(t) + pd.DateOffset(months=months)


def _in_memory_targets(conn, legacy_window: bool) -> pd.DataFrame:
    frames = []
    ctx = (
        patch.object(multi_total_return, "forward_window_end", _legacy_window_end)
        if legacy_window else nullcontext()
    )
    with ctx:
        for horizon in (6, 12):
            wide = build_relative_return_targets(conn, horizon, upsert=False)
            long = wide.stack().dropna().rename("relative_return").reset_index()
            long.columns = ["date", "benchmark", "relative_return"]
            long["date"] = long["date"].dt.strftime("%Y-%m-%d")
            long["target_horizon"] = horizon
            frames.append(long)
    return pd.concat(frames, ignore_index=True)


def diff_tables(
    before: pd.DataFrame,
    after: pd.DataFrame,
    data_only: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (row-level diff of changed/added/removed rows, per-benchmark summary)."""
    b = before.set_index(_KEY)["relative_return"].rename("before")
    a = after.set_index(_KEY)["relative_return"].rename("after")
    rows = pd.concat([b, a], axis=1)
    if data_only is not None:
        rows = rows.join(data_only.set_index(_KEY)["relative_return"].rename("data_only"))
    rows["delta"] = rows["after"] - rows["before"]
    rows["status"] = np.select(
        [rows["before"].isna(), rows["after"].isna(), rows["delta"].abs() > 1e-12],
        ["added", "removed", "changed"],
        default="unchanged",
    )
    rows["sign_flip"] = (
        (rows["status"] == "changed") & (np.sign(rows["before"]) != np.sign(rows["after"]))
    )
    if data_only is not None:
        rows["delta_data"] = rows["data_only"] - rows["before"]
        rows["delta_window"] = rows["after"] - rows["data_only"]
    rows = rows.reset_index()

    def summarise(g: pd.DataFrame) -> pd.Series:
        changed = g[g["status"] == "changed"]
        absd = changed["delta"].abs()
        out = {
            "rows_before": int(g["before"].notna().sum()),
            "rows_after": int(g["after"].notna().sum()),
            "added": int((g["status"] == "added").sum()),
            "removed": int((g["status"] == "removed").sum()),
            "changed": int(len(changed)),
            "changed_gt_1pp": int((absd > 0.01).sum()),
            "mean_abs_delta_pp": float(absd.mean() * 100) if len(changed) else 0.0,
            "max_abs_delta_pp": float(absd.max() * 100) if len(changed) else 0.0,
            "max_delta_date": (
                changed.loc[absd.idxmax(), "date"] if len(changed) else ""
            ),
            "sign_flips": int(g["sign_flip"].sum()),
        }
        if "delta_data" in g:
            dd = changed["delta_data"].abs()
            dw = changed["delta_window"].abs()
            out["rows_data_change"] = int((dd > 1e-12).sum())
            out["max_abs_data_delta_pp"] = float(dd.max() * 100) if len(changed) else 0.0
            out["rows_window_change"] = int((dw > 1e-12).sum())
            out["mean_abs_window_delta_pp"] = float(dw.mean() * 100) if len(changed) else 0.0
        return pd.Series(out)

    summary = (
        rows.groupby(["benchmark", "target_horizon"])
        .apply(summarise, include_groups=False)
        .reset_index()
    )
    return rows[rows["status"] != "unchanged"].reset_index(drop=True), summary


def _markdown_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild monthly_relative_returns.")
    parser.add_argument("--db", default=config.DB_PATH)
    parser.add_argument("--report", default=None, help="Write a Markdown diff report here.")
    parser.add_argument("--rows-csv", default=None, help="Write changed rows as CSV here.")
    parser.add_argument("--attribute", action="store_true",
                        help="Split each change into data (splits/bars) and window parts.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = db_client.get_connection(args.db)
    before = _snapshot(conn)
    logger.info("monthly_relative_returns rows before: %s", len(before))

    n_splits = db_client.upsert_splits(conn, config.KNOWN_SPLITS)
    logger.info("Seeded %s canonical split rows.", n_splits)
    n_dupes = db_client.dedupe_weekly_price_bars(conn)
    logger.info("Deleted %s superseded partial-week price bars.", n_dupes)

    data_only = _in_memory_targets(conn, legacy_window=True) if args.attribute else None

    # Replace the whole table so benchmarks no longer in the universe and rows
    # without a complete window cannot linger.
    with conn:
        conn.execute("DELETE FROM monthly_relative_returns")
    for horizon in (6, 12):
        build_relative_return_targets(conn, forward_months=horizon, upsert=True)
    after = _snapshot(conn)
    logger.info("monthly_relative_returns rows after: %s", len(after))
    conn.close()
    db_client.finalize_for_commit(args.db)

    rows, summary = diff_tables(before, after, data_only)
    logger.info("Rows changed/added/removed: %s", rows["status"].value_counts().to_dict())
    if args.rows_csv:
        rows.to_csv(args.rows_csv, index=False, float_format="%.10g")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("# monthly_relative_returns rebuild diff\n\n")
            fh.write(f"- Rows before: {len(before)}; rows after: {len(after)}\n")
            fh.write(f"- Split rows seeded: {n_splits}; duplicate week bars deleted: {n_dupes}\n")
            counts = rows["status"].value_counts().to_dict()
            fh.write(f"- Row status counts: {counts}\n\n")
            fh.write(_markdown_table(summary))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
