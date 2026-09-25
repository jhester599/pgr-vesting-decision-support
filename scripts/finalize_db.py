"""Prepare the SQLite database for ``git add``.

Runs ``PRAGMA wal_checkpoint(TRUNCATE)`` and sets ``journal_mode=DELETE`` so
the committed file is self-contained and does not leave ``-wal``/``-shm``
sidecars behind. Every workflow that commits ``data/pgr_financials.db`` runs
this immediately before ``git add``.

Usage:
    python scripts/finalize_db.py [--db PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client


def main(db_path: str | None = None) -> None:
    """Checkpoint the WAL and switch the database to DELETE journal mode."""
    path = db_path or config.DB_PATH
    mode = db_client.finalize_for_commit(path)
    print(f"{path}: WAL checkpointed, journal_mode={mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=None, help="Database path (default: config.DB_PATH).")
    args = parser.parse_args()
    main(args.db)
