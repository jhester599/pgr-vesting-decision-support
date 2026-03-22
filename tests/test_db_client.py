"""
Tests for src/database/db_client.py — Phase 1 validation.

Covers:
  - Schema initialisation is idempotent
  - Upsert semantics (insert + replace, no duplicates)
  - Date-range filtering on get_prices
  - proxy_fill flag round-trips correctly
  - log_api_request raises RuntimeError at limit
  - get_api_request_count aggregates across endpoints
  - Dividend, split, and fundamentals upsert / query helpers
  - ingestion_metadata update and retrieval
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

import config
from src.database import db_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    """Provide an in-memory-like (tmp file) connection with fresh schema."""
    db_path = str(tmp_path / "test.db")
    c = db_client.get_connection(db_path)
    db_client.initialize_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestInitializeSchema:
    def test_idempotent(self, conn):
        """Running initialize_schema twice must not raise any error."""
        db_client.initialize_schema(conn)  # second call
        # verify tables exist
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "daily_prices", "daily_dividends", "split_history",
            "pgr_fundamentals_quarterly", "pgr_edgar_monthly",
            "monthly_relative_returns", "api_request_log", "ingestion_metadata",
        }
        assert expected.issubset(tables)

    def test_wal_mode(self, conn):
        result = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert result == "wal"

    def test_foreign_keys_on(self, conn):
        result = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert result == 1


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

class TestUpsertPrices:
    def _sample(self, ticker: str = "TST", date: str = "2020-01-31") -> dict:
        return {
            "ticker": ticker, "date": date,
            "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5,
            "volume": 1000, "source": "av", "proxy_fill": 0,
        }

    def test_insert_returns_count(self, conn):
        n = db_client.upsert_prices(conn, [self._sample()])
        assert n == 1

    def test_empty_list_returns_zero(self, conn):
        assert db_client.upsert_prices(conn, []) == 0

    def test_duplicate_replaces_not_duplicates(self, conn):
        rec = self._sample()
        db_client.upsert_prices(conn, [rec])
        rec2 = {**rec, "close": 99.0}  # same PK, different value
        db_client.upsert_prices(conn, [rec2])
        df = db_client.get_prices(conn, "TST")
        assert len(df) == 1
        assert df["close"].iloc[0] == pytest.approx(99.0)

    def test_proxy_fill_roundtrips(self, conn):
        rec = {**self._sample(), "proxy_fill": 1}
        db_client.upsert_prices(conn, [rec])
        df = db_client.get_prices(conn, "TST")
        assert df["proxy_fill"].iloc[0] == 1

    def test_default_proxy_fill_is_zero(self, conn):
        rec = {k: v for k, v in self._sample().items() if k != "proxy_fill"}
        db_client.upsert_prices(conn, [rec])
        df = db_client.get_prices(conn, "TST")
        assert df["proxy_fill"].iloc[0] == 0


class TestGetPrices:
    def _load_three_rows(self, conn):
        records = [
            {"ticker": "TST", "date": "2020-01-31", "close": 10.0, "proxy_fill": 0},
            {"ticker": "TST", "date": "2020-02-29", "close": 11.0, "proxy_fill": 0},
            {"ticker": "TST", "date": "2020-03-31", "close": 9.0,  "proxy_fill": 1},
        ]
        db_client.upsert_prices(conn, records)

    def test_returns_all_rows_without_filter(self, conn):
        self._load_three_rows(conn)
        df = db_client.get_prices(conn, "TST")
        assert len(df) == 3

    def test_ascending_date_order(self, conn):
        self._load_three_rows(conn)
        df = db_client.get_prices(conn, "TST")
        assert list(df.index) == sorted(df.index)

    def test_start_date_filter(self, conn):
        self._load_three_rows(conn)
        df = db_client.get_prices(conn, "TST", start_date="2020-02-29")
        assert len(df) == 2

    def test_end_date_filter(self, conn):
        self._load_three_rows(conn)
        df = db_client.get_prices(conn, "TST", end_date="2020-01-31")
        assert len(df) == 1

    def test_exclude_proxy_filter(self, conn):
        self._load_three_rows(conn)
        df = db_client.get_prices(conn, "TST", exclude_proxy=True)
        assert len(df) == 2
        assert all(df["proxy_fill"] == 0)

    def test_date_index_is_datetimeindex(self, conn):
        self._load_three_rows(conn)
        df = db_client.get_prices(conn, "TST")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_unknown_ticker_returns_empty(self, conn):
        df = db_client.get_prices(conn, "ZZZZZ")
        assert df.empty


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------

class TestUpsertDividends:
    def test_insert_and_retrieve(self, conn):
        records = [
            {"ticker": "PGR", "ex_date": "2023-06-15", "amount": 0.10, "source": "av"},
            {"ticker": "PGR", "ex_date": "2023-09-15", "amount": 0.12, "source": "av"},
        ]
        n = db_client.upsert_dividends(conn, records)
        assert n == 2
        df = db_client.get_dividends(conn, "PGR")
        assert len(df) == 2

    def test_duplicate_replaces(self, conn):
        db_client.upsert_dividends(conn, [
            {"ticker": "PGR", "ex_date": "2023-06-15", "amount": 0.10}
        ])
        db_client.upsert_dividends(conn, [
            {"ticker": "PGR", "ex_date": "2023-06-15", "amount": 0.99}
        ])
        df = db_client.get_dividends(conn, "PGR")
        assert len(df) == 1
        assert df["amount"].iloc[0] == pytest.approx(0.99)

    def test_returns_datetimeindex(self, conn):
        db_client.upsert_dividends(conn, [
            {"ticker": "PGR", "ex_date": "2023-06-15", "amount": 0.10}
        ])
        df = db_client.get_dividends(conn, "PGR")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_empty_list_returns_zero(self, conn):
        assert db_client.upsert_dividends(conn, []) == 0


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

class TestUpsertSplits:
    def test_insert_and_retrieve(self, conn):
        records = [
            {"ticker": "PGR", "split_date": "2006-05-19",
             "split_ratio": 4.0, "numerator": 4, "denominator": 1},
        ]
        n = db_client.upsert_splits(conn, records)
        assert n == 1
        df = db_client.get_splits(conn, "PGR")
        assert len(df) == 1
        assert df["split_ratio"].iloc[0] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# PGR Fundamentals
# ---------------------------------------------------------------------------

class TestPGRFundamentals:
    def test_upsert_and_retrieve(self, conn):
        records = [
            {"period_end": "2023-03-31", "pe_ratio": 22.5, "pb_ratio": 4.1,
             "roe": 0.18, "eps": 5.2, "revenue": 1500.0, "net_income": 300.0,
             "source": "fmp"},
        ]
        n = db_client.upsert_pgr_fundamentals(conn, records)
        assert n == 1
        df = db_client.get_pgr_fundamentals(conn)
        assert len(df) == 1
        assert df["pe_ratio"].iloc[0] == pytest.approx(22.5)

    def test_datetimeindex(self, conn):
        db_client.upsert_pgr_fundamentals(conn, [
            {"period_end": "2023-03-31", "pe_ratio": 22.5, "source": "fmp"}
        ])
        df = db_client.get_pgr_fundamentals(conn)
        assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# EDGAR Monthly
# ---------------------------------------------------------------------------

class TestPGREdgarMonthly:
    def test_upsert_and_retrieve(self, conn):
        records = [
            {"month_end": "2023-01-31", "combined_ratio": 91.5,
             "pif_total": 30.0e6, "pif_growth_yoy": 0.12, "gainshare_estimate": 1.4},
        ]
        n = db_client.upsert_pgr_edgar_monthly(conn, records)
        assert n == 1
        df = db_client.get_pgr_edgar_monthly(conn)
        assert len(df) == 1
        assert df["combined_ratio"].iloc[0] == pytest.approx(91.5)


# ---------------------------------------------------------------------------
# Relative Returns
# ---------------------------------------------------------------------------

class TestRelativeReturns:
    def _load_sample(self, conn):
        records = [
            {"date": "2023-01-31", "benchmark": "VTI", "target_horizon": 6,
             "pgr_return": 0.08, "benchmark_return": 0.05, "relative_return": 0.03,
             "proxy_fill": 0},
            {"date": "2023-02-28", "benchmark": "VTI", "target_horizon": 6,
             "pgr_return": -0.02, "benchmark_return": 0.01, "relative_return": -0.03,
             "proxy_fill": 0},
        ]
        db_client.upsert_relative_returns(conn, records)

    def test_upsert_and_retrieve_series(self, conn):
        self._load_sample(conn)
        s = db_client.get_relative_returns(conn, "VTI", 6)
        assert len(s) == 2
        assert s.name == "VTI_6m"

    def test_series_is_datetimeindex(self, conn):
        self._load_sample(conn)
        s = db_client.get_relative_returns(conn, "VTI", 6)
        assert isinstance(s.index, pd.DatetimeIndex)

    def test_date_filter(self, conn):
        self._load_sample(conn)
        s = db_client.get_relative_returns(conn, "VTI", 6, end_date="2023-01-31")
        assert len(s) == 1

    def test_missing_benchmark_returns_empty(self, conn):
        self._load_sample(conn)
        s = db_client.get_relative_returns(conn, "BND", 6)
        assert s.empty


# ---------------------------------------------------------------------------
# API Rate-Limit Tracking
# ---------------------------------------------------------------------------

class TestApiRateLimit:
    def test_single_log_increments_count(self, conn):
        db_client.log_api_request(conn, "av", endpoint="/test", utc_date="2025-01-01")
        count = db_client.get_api_request_count(conn, "av", "2025-01-01")
        assert count == 1

    def test_multiple_logs_sum_correctly(self, conn):
        for _ in range(5):
            db_client.log_api_request(conn, "fmp", endpoint="/ep", utc_date="2025-01-01")
        count = db_client.get_api_request_count(conn, "fmp", "2025-01-01")
        assert count == 5

    def test_different_endpoints_aggregate(self, conn):
        db_client.log_api_request(conn, "av", endpoint="/prices", utc_date="2025-01-01")
        db_client.log_api_request(conn, "av", endpoint="/dividends", utc_date="2025-01-01")
        count = db_client.get_api_request_count(conn, "av", "2025-01-01")
        assert count == 2

    def test_different_dates_do_not_aggregate(self, conn):
        db_client.log_api_request(conn, "av", endpoint="/x", utc_date="2025-01-01")
        db_client.log_api_request(conn, "av", endpoint="/x", utc_date="2025-01-02")
        assert db_client.get_api_request_count(conn, "av", "2025-01-01") == 1
        assert db_client.get_api_request_count(conn, "av", "2025-01-02") == 1

    def test_raises_at_av_limit(self, conn):
        limit = config.AV_DAILY_LIMIT
        for i in range(limit):
            db_client.log_api_request(conn, "av", endpoint=f"/e{i}", utc_date="2025-06-01")
        with pytest.raises(RuntimeError, match="Daily API limit reached"):
            db_client.log_api_request(conn, "av", endpoint="/over", utc_date="2025-06-01")

    def test_raises_at_fmp_limit(self, conn):
        limit = config.FMP_DAILY_LIMIT
        for i in range(limit):
            db_client.log_api_request(conn, "fmp", endpoint=f"/e{i}", utc_date="2025-06-01")
        with pytest.raises(RuntimeError, match="Daily API limit reached"):
            db_client.log_api_request(conn, "fmp", endpoint="/over", utc_date="2025-06-01")

    def test_zero_count_for_unknown_date(self, conn):
        count = db_client.get_api_request_count(conn, "av", "1999-01-01")
        assert count == 0


# ---------------------------------------------------------------------------
# Ingestion Metadata
# ---------------------------------------------------------------------------

class TestIngestionMetadata:
    def test_update_and_retrieve(self, conn):
        db_client.update_ingestion_metadata(conn, "PGR", "prices", rows_stored=1000)
        meta = db_client.get_ingestion_metadata(conn, "PGR", "prices")
        assert meta is not None
        assert meta["ticker"] == "PGR"
        assert meta["data_type"] == "prices"
        assert meta["rows_stored"] == 1000
        assert meta["last_fetched"] is not None

    def test_update_overwrites_previous(self, conn):
        db_client.update_ingestion_metadata(conn, "VTI", "prices", rows_stored=500)
        db_client.update_ingestion_metadata(conn, "VTI", "prices", rows_stored=600)
        meta = db_client.get_ingestion_metadata(conn, "VTI", "prices")
        assert meta["rows_stored"] == 600

    def test_missing_entry_returns_none(self, conn):
        meta = db_client.get_ingestion_metadata(conn, "ZZZZZ", "prices")
        assert meta is None
