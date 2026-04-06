from __future__ import annotations

import sqlite3
from pathlib import Path

from src.database import db_client


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def test_initialize_schema_applies_migration_on_fresh_db(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.db"
    conn = db_client.get_connection(str(db_path))

    db_client.initialize_schema(conn)

    version = db_client.get_schema_version(conn)
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    migration_rows = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    conn.close()

    assert version == "002_model_performance_log"
    assert "pgr_edgar_monthly" in tables
    assert "model_performance_log" in tables
    assert "schema_migrations" in tables
    assert migration_rows == 2


def test_initialize_schema_reconciles_legacy_db_shape(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE pgr_edgar_monthly (
            month_end TEXT PRIMARY KEY,
            combined_ratio REAL,
            pif_total REAL
        )
        """
    )
    conn.commit()
    conn.close()

    conn2 = db_client.get_connection(str(db_path))
    db_client.initialize_schema(conn2)
    cols = _columns(conn2, "pgr_edgar_monthly")
    version = db_client.get_schema_version(conn2)
    conn2.close()

    assert "book_value_per_share" in cols
    assert "buyback_yield" in cols
    assert version == "002_model_performance_log"


def test_initialize_schema_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "idempotent.db"
    conn = db_client.get_connection(str(db_path))

    db_client.initialize_schema(conn)
    db_client.initialize_schema(conn)

    rows = [str(row[0]) for row in conn.execute("SELECT migration_id FROM schema_migrations").fetchall()]
    conn.close()

    assert rows == ["001_initial", "002_model_performance_log"]
