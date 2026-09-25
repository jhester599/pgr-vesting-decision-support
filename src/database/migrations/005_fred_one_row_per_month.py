"""Migration 005: one ``fred_macro_monthly`` row per series and month.

Review 2026-09-25, findings F06 and F27. The production FRED loader labels
each month with its last business day, but ``src/research/v19.py`` wrote
calendar month-ends, so 64 months (2008-05 to 2026-02) of each macro series
were stored twice (for example 2020-05-29 and 2020-05-31). The feature
builder lagged by row, so a duplicated month cost an extra month of lag.

This migration keeps one row per (series_id, calendar month): the row
labelled with the month's last business day when there is one (written by
the production loader), otherwise the latest-labelled row. It relabels the
survivor to the last business day, the label ``db_client.fred_month_label``
now gives every write, and adds a unique index on (series_id, month) so a
second label for the same month can never be stored again.

It does not change values. The values stored before this review were
publication-lagged by the loader; ``scripts/rebuild_fred_macro.py`` replaces
them with raw observations.
"""

from __future__ import annotations

import sqlite3
from calendar import monthrange
from datetime import date, timedelta


def _business_month_end(year: int, month: int) -> str:
    day = date(year, month, monthrange(year, month)[1])
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.isoformat()


def upgrade(conn: sqlite3.Connection) -> None:
    """Collapse duplicate month rows and add the (series, month) unique index."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'fred_macro_monthly'"
    ).fetchone()
    if not exists:
        return

    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for rowid, series_id, month_end in conn.execute(
        "SELECT rowid, series_id, month_end FROM fred_macro_monthly"
    ):
        key = (str(series_id), str(month_end)[:7])
        groups.setdefault(key, []).append((int(rowid), str(month_end)))

    for (_series_id, month), rows in groups.items():
        label = _business_month_end(int(month[:4]), int(month[5:7]))
        keep = next((r for r in rows if r[1] == label), max(rows, key=lambda r: r[1]))
        for rowid, _ in rows:
            if rowid != keep[0]:
                conn.execute("DELETE FROM fred_macro_monthly WHERE rowid = ?", (rowid,))
        if keep[1] != label:
            conn.execute(
                "UPDATE fred_macro_monthly SET month_end = ? WHERE rowid = ?",
                (label, keep[0]),
            )

    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_fred_macro_monthly_series_month
        ON fred_macro_monthly (series_id, substr(month_end, 1, 7))
        """
    )
