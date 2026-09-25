"""Regression tests for review 2026-09-25 findings F10 and F35.

F10: the live 8-K parser divided the investment book yield by 100, so the
     committed DB mixed percent (3.8) and fraction (0.038) units in a live
     GBT feature. The parser now stores percent and migration 004 rescales
     the stored fractions.
F35: the parser emits ``roe_net_income_trailing_12m`` but the monthly upsert
     only read ``roe_net_income_ttm``, so the live fetch never wrote it.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.edgar_8k_fetcher import _parse_html_exhibit
from src.database import db_client, migration_runner
from tests.test_edgar_8k_parser_breadth import _broad_exhibit_html

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_DB = REPO_ROOT / "data" / "pgr_financials.db"

# Parser keys whose DB column has a different name.
_PARSER_TO_DB_COLUMN = {"roe_net_income_trailing_12m": "roe_net_income_ttm"}


def _fresh_db(tmp_path: Path) -> sqlite3.Connection:
    conn = db_client.get_connection(str(tmp_path / "edgar.db"))
    db_client.initialize_schema(conn)
    return conn


def _book_yield_html(yield_text: str) -> str:
    return f"""
    <html><body>
      <table>
        <tr><td>Combined ratio</td><td>94.1</td></tr>
        <tr><td>Net premiums written</td><td>$</td><td>6000.0</td></tr>
      </table>
      <table>
        <tr><td>Fixed-income securities</td><td>0.4%</td></tr>
        <tr><td>Common stocks</td><td>1.2%</td></tr>
        <tr><td>Total portfolio</td><td>0.5%</td></tr>
        <tr><td>Pretax annualized investment income book yield</td><td>{yield_text}</td></tr>
      </table>
    </body></html>
    """


# ---------------------------------------------------------------------------
# F35: parse -> upsert -> read back every field
# ---------------------------------------------------------------------------


def test_parse_upsert_read_back_round_trips_every_parsed_field(tmp_path: Path) -> None:
    parsed = _parse_html_exhibit(_broad_exhibit_html(), "2023-09-15")
    assert parsed is not None
    assert parsed["roe_net_income_trailing_12m"] == pytest.approx(10.4)

    conn = _fresh_db(tmp_path)
    db_client.upsert_pgr_edgar_monthly(conn, [parsed])
    row = conn.execute(
        "SELECT * FROM pgr_edgar_monthly WHERE month_end = ?",
        (parsed["month_end"],),
    ).fetchone()
    assert row is not None
    stored = dict(row)

    mismatches: dict[str, tuple[object, object]] = {}
    for key, value in parsed.items():
        # ``derived_fields`` is provenance metadata for pgr_edgar_monthly_raw.
        if value is None or key == "derived_fields":
            continue
        column = _PARSER_TO_DB_COLUMN.get(key, key)
        assert column in stored, f"parser key {key!r} has no DB column"
        got = stored[column]
        if isinstance(value, float):
            if got is None or got != pytest.approx(value):
                mismatches[key] = (value, got)
        elif got != value:
            mismatches[key] = (value, got)
    conn.close()
    assert not mismatches, f"fields lost or changed on upsert: {mismatches}"


def test_roe_net_income_ttm_is_readable_via_get_pgr_edgar_monthly(tmp_path: Path) -> None:
    parsed = _parse_html_exhibit(_broad_exhibit_html(), "2023-09-15")
    assert parsed is not None
    conn = _fresh_db(tmp_path)
    db_client.upsert_pgr_edgar_monthly(conn, [parsed])
    df = db_client.get_pgr_edgar_monthly(conn)
    conn.close()
    assert df["roe_net_income_ttm"].iloc[-1] == pytest.approx(10.4)


def test_explicit_roe_net_income_ttm_key_still_wins(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    db_client.upsert_pgr_edgar_monthly(
        conn,
        [
            {
                "month_end": "2024-01-31",
                "roe_net_income_ttm": 30.0,
                "roe_net_income_trailing_12m": 99.0,
            }
        ],
    )
    value = conn.execute(
        "SELECT roe_net_income_ttm FROM pgr_edgar_monthly WHERE month_end = '2024-01-31'"
    ).fetchone()[0]
    conn.close()
    assert value == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# F10: investment_book_yield units
# ---------------------------------------------------------------------------


def test_parsing_book_yield_percent_stores_percent(tmp_path: Path) -> None:
    parsed = _parse_html_exhibit(_book_yield_html("4.2%"), "2025-06-18")
    assert parsed is not None
    assert parsed["investment_book_yield"] == pytest.approx(4.2)

    conn = _fresh_db(tmp_path)
    db_client.upsert_pgr_edgar_monthly(conn, [parsed])
    stored = conn.execute(
        "SELECT investment_book_yield FROM pgr_edgar_monthly WHERE month_end = ?",
        (parsed["month_end"],),
    ).fetchone()[0]
    conn.close()
    assert stored == pytest.approx(4.2)


def test_migration_004_rescales_fractions_and_leaves_percent_untouched(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    migration_runner.ensure_migration_table(conn)
    for migration in migration_runner.list_migrations():
        if migration.migration_id >= "004":
            continue
        conn.executescript(migration.path.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO schema_migrations (migration_id, applied_at) VALUES (?, 'test')",
            (migration.migration_id,),
        )
    conn.executemany(
        "INSERT INTO pgr_edgar_monthly (month_end, investment_book_yield) VALUES (?, ?)",
        [
            ("2023-12-31", 2.6),
            ("2024-04-30", 0.038),
            ("2024-12-31", 0.040999999999999995),
            ("2025-01-31", None),
        ],
    )
    conn.commit()

    applied = migration_runner.apply_migrations(conn)
    assert "004_investment_book_yield_percent" in applied
    values = dict(
        conn.execute("SELECT month_end, investment_book_yield FROM pgr_edgar_monthly").fetchall()
    )
    # Re-running is a no-op: the migration is recorded and not re-applied.
    assert migration_runner.apply_migrations(conn) == []
    conn.close()

    assert values["2023-12-31"] == pytest.approx(2.6)
    assert values["2024-04-30"] == 3.8
    assert values["2024-12-31"] == 4.1
    assert values["2025-01-31"] is None


def test_committed_db_investment_book_yield_is_percent() -> None:
    """Every non-null stored book yield must be a plausible percent value."""
    conn = sqlite3.connect(f"{COMMITTED_DB.as_uri()}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT month_end, investment_book_yield FROM pgr_edgar_monthly "
            "WHERE investment_book_yield IS NOT NULL",
            conn,
        )
    finally:
        conn.close()
    assert not df.empty
    out_of_range = df.loc[
        (df["investment_book_yield"] < 0.5) | (df["investment_book_yield"] > 10.0)
    ]
    assert out_of_range.empty, (
        f"{len(out_of_range)} investment_book_yield values outside [0.5, 10]: "
        f"{out_of_range.head(5).to_dict('records')}"
    )
