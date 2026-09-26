"""
Tests for FRED macro DB operations in src/database/db_client.py (v3.0).

Verifies schema creation, upsert idempotence, and get_fred_macro() column alignment.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.database.db_client import (
    get_fred_macro,
    initialize_schema,
    upsert_fred_macro,
)


@pytest.fixture()
def mem_conn() -> sqlite3.Connection:
    """In-memory SQLite connection with schema initialized."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_schema(conn)
    yield conn
    conn.close()


def _sample_records() -> list[dict]:
    return [
        {"series_id": "T10Y2Y", "month_end": "2024-01-31", "value": 0.35},
        {"series_id": "T10Y2Y", "month_end": "2024-02-29", "value": 0.42},
        {"series_id": "GS10",   "month_end": "2024-01-31", "value": 4.10},
        {"series_id": "GS10",   "month_end": "2024-02-29", "value": 4.15},
    ]


class TestUpsertFredMacro:
    def test_empty_list_returns_zero(self, mem_conn):
        assert upsert_fred_macro(mem_conn, []) == 0

    def test_upsert_inserts_correct_count(self, mem_conn):
        records = _sample_records()
        n = upsert_fred_macro(mem_conn, records)
        assert n == 4

    def test_upsert_idempotent(self, mem_conn):
        records = _sample_records()
        upsert_fred_macro(mem_conn, records)
        n2 = upsert_fred_macro(mem_conn, records)
        # Second upsert should not raise and should not duplicate
        count = mem_conn.execute(
            "SELECT COUNT(*) FROM fred_macro_monthly"
        ).fetchone()[0]
        assert count == 4

    def test_null_value_stored(self, mem_conn):
        records = [{"series_id": "NFCI", "month_end": "2024-01-31", "value": None}]
        upsert_fred_macro(mem_conn, records)
        row = mem_conn.execute(
            "SELECT value FROM fred_macro_monthly WHERE series_id = 'NFCI'"
        ).fetchone()
        assert row is not None
        assert row["value"] is None


class TestGetFredMacro:
    def test_returns_empty_when_table_empty(self, mem_conn):
        df = get_fred_macro(mem_conn)
        assert df.empty

    def test_returns_wide_format(self, mem_conn):
        upsert_fred_macro(mem_conn, _sample_records())
        df = get_fred_macro(mem_conn)
        assert "T10Y2Y" in df.columns
        assert "GS10" in df.columns
        assert len(df) == 2  # 2 months

    def test_filter_by_series_ids(self, mem_conn):
        upsert_fred_macro(mem_conn, _sample_records())
        df = get_fred_macro(mem_conn, series_ids=["T10Y2Y"])
        assert "T10Y2Y" in df.columns
        assert "GS10" not in df.columns

    def test_index_is_datetime(self, mem_conn):
        upsert_fred_macro(mem_conn, _sample_records())
        df = get_fred_macro(mem_conn)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_sorted_ascending(self, mem_conn):
        # Insert in reversed order
        reversed_records = list(reversed(_sample_records()))
        upsert_fred_macro(mem_conn, reversed_records)
        df = get_fred_macro(mem_conn)
        assert df.index.is_monotonic_increasing

    def test_values_correct(self, mem_conn):
        upsert_fred_macro(mem_conn, _sample_records())
        df = get_fred_macro(mem_conn)
        assert abs(df.loc["2024-01-31", "T10Y2Y"] - 0.35) < 1e-9
        assert abs(df.loc["2024-02-29", "GS10"] - 4.15) < 1e-9
