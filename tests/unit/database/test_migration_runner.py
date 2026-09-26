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

    assert version == "008_model_performance_log_metrics_version"
    assert "pgr_edgar_monthly" in tables
    assert "model_performance_log" in tables
    assert "model_retrain_log" in tables
    assert "schema_migrations" in tables
    assert migration_rows == 8


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
    assert "investment_book_yield" in cols
    assert version == "008_model_performance_log_metrics_version"


def test_initialize_schema_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "idempotent.db"
    conn = db_client.get_connection(str(db_path))

    db_client.initialize_schema(conn)
    db_client.initialize_schema(conn)

    rows = [str(row[0]) for row in conn.execute("SELECT migration_id FROM schema_migrations").fetchall()]
    conn.close()

    assert rows == [
        "001_initial",
        "002_model_performance_log",
        "003_model_retrain_log",
        "004_investment_book_yield_percent",
        "005_fred_one_row_per_month",
        "006_pgr_edgar_monthly_raw",
        "007_pgr_fundamentals_quarterly_rebuild",
        "008_model_performance_log_metrics_version",
    ]


def test_list_migrations_includes_python_migrations_in_order() -> None:
    from src.database import migration_runner

    migrations = migration_runner.list_migrations()
    ids = [m.migration_id for m in migrations]
    assert ids == sorted(ids)
    assert "004_investment_book_yield_percent" in ids
    assert "006_pgr_edgar_monthly_raw" in ids
    assert "007_pgr_fundamentals_quarterly_rebuild" in ids
    assert all(not m.path.name.startswith("_") for m in migrations)


def test_book_yield_migration_is_noop_when_column_missing(tmp_path: Path) -> None:
    """A pre-v6 table without investment_book_yield must not break migration 004."""
    db_path = tmp_path / "very_old.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE pgr_edgar_monthly (month_end TEXT PRIMARY KEY, combined_ratio REAL)")
    conn.execute("INSERT INTO pgr_edgar_monthly VALUES ('2020-01-31', 92.0)")
    conn.commit()
    conn.close()

    conn2 = db_client.get_connection(str(db_path))
    db_client.initialize_schema(conn2)
    row = conn2.execute(
        "SELECT combined_ratio, investment_book_yield FROM pgr_edgar_monthly"
    ).fetchone()
    conn2.close()
    assert tuple(row) == (92.0, None)
