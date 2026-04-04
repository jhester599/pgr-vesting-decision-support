"""Schema migration runner for the SQLite operational database."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Migration:
    """One ordered SQL migration file."""

    migration_id: str
    path: Path


def migrations_dir() -> Path:
    """Return the on-disk migrations directory."""
    return Path(__file__).with_name("migrations")


def list_migrations() -> list[Migration]:
    """Return migration files sorted by filename."""
    paths = sorted(migrations_dir().glob("*.sql"))
    return [Migration(migration_id=path.stem, path=path) for path in paths]


def ensure_migration_table(conn: sqlite3.Connection) -> None:
    """Create the schema_migrations table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at   TEXT NOT NULL
        )
        """
    )
    conn.commit()


def get_applied_migration_ids(conn: sqlite3.Connection) -> set[str]:
    """Return the set of already-applied migration identifiers."""
    ensure_migration_table(conn)
    rows = conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
    return {str(row[0]) for row in rows}


def current_schema_version(conn: sqlite3.Connection) -> str | None:
    """Return the latest applied migration id, or None if nothing is applied."""
    ensure_migration_table(conn)
    row = conn.execute(
        "SELECT migration_id FROM schema_migrations ORDER BY migration_id DESC LIMIT 1"
    ).fetchone()
    return str(row[0]) if row else None


def apply_migrations(conn: sqlite3.Connection) -> list[str]:
    """Apply all unapplied ordered SQL migrations and return their ids."""
    ensure_migration_table(conn)
    applied = get_applied_migration_ids(conn)
    applied_now: list[str] = []

    for migration in list_migrations():
        if migration.migration_id in applied:
            continue
        sql = migration.path.read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.execute(
            """
            INSERT INTO schema_migrations (migration_id, applied_at)
            VALUES (?, ?)
            """,
            (
                migration.migration_id,
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        applied_now.append(migration.migration_id)
        applied.add(migration.migration_id)

    return applied_now
