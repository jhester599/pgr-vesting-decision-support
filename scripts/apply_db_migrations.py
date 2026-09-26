"""Apply pending SQL migrations to a database, then finalize it for commit.

This is the explicit, reviewable way to change the committed database: the
migrations (``.sql`` scripts or ``.py`` modules with ``upgrade(conn)``) live in
``src/database/migrations/`` and this script applies whatever is
pending, records it in ``schema_migrations``, and leaves the file in
``journal_mode=DELETE``. Normal read-write runs apply the same migrations via
``db_client.initialize_schema``.

Usage:
    python scripts/apply_db_migrations.py [--db PATH]

Migrations:
    004_investment_book_yield_percent  Rescale investment_book_yield rows
                                       stored as fractions to percent (F10).
    005_fred_one_row_per_month         Keep one fred_macro_monthly row per
                                       series and month; add a unique index
                                       (F06/F27).
    008_model_performance_log_metrics_version
                                       Tag model-health rows with the metric
                                       definitions they used; existing rows
                                       become 'pre-2026-09-25' (WP7).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client, migration_runner


def main(db_path: str | None = None) -> list[str]:
    """Apply pending migrations to ``db_path`` and return their ids."""
    path = db_path or config.DB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database does not exist: {path}")
    conn = db_client.get_connection(path)
    try:
        applied = migration_runner.apply_migrations(conn)
    finally:
        conn.close()
    db_client.finalize_for_commit(path)
    if applied:
        print(f"{path}: applied {', '.join(applied)}")
    else:
        print(f"{path}: no pending migrations")
    return applied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=None, help="Database path (default: config.DB_PATH).")
    args = parser.parse_args()
    main(args.db)
