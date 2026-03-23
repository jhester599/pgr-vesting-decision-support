"""
Tests for src/ingestion/fred_loader.py (v3.0).

Tests mock the HTTP layer so no real FRED API key is required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import config
from src.ingestion.fred_loader import (
    fetch_all_fred_macro,
    fetch_fred_series,
    upsert_fred_to_db,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fred_response(series_id: str, observations: list[dict]) -> MagicMock:
    """Build a mock requests.Response for a FRED observations call."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"observations": observations}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


_SAMPLE_OBSERVATIONS = [
    {"date": "2024-01-31", "value": "0.35"},
    {"date": "2024-02-29", "value": "0.42"},
    {"date": "2024-03-31", "value": "."},  # FRED missing-value sentinel
    {"date": "2024-04-30", "value": "0.51"},
]


# ---------------------------------------------------------------------------
# fetch_fred_series
# ---------------------------------------------------------------------------

class TestFetchFredSeries:
    def test_dry_run_returns_empty(self):
        df = fetch_fred_series("T10Y2Y", dry_run=True)
        assert df.empty
        assert "T10Y2Y" in df.columns

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.setattr(config, "FRED_API_KEY", None)
        with pytest.raises(RuntimeError, match="FRED_API_KEY"):
            fetch_fred_series("T10Y2Y", dry_run=False)

    def test_parses_observations_correctly(self, monkeypatch):
        monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
        mock_resp = _make_fred_response("T10Y2Y", _SAMPLE_OBSERVATIONS)

        with patch("src.ingestion.fred_loader.requests.get", return_value=mock_resp):
            df = fetch_fred_series("T10Y2Y", dry_run=False)

        assert "T10Y2Y" in df.columns
        assert len(df) == 4
        assert df.index.dtype.kind == "M"  # any datetime resolution (ns, us, etc.)

    def test_missing_value_sentinel_becomes_nan(self, monkeypatch):
        monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
        mock_resp = _make_fred_response("T10Y2Y", _SAMPLE_OBSERVATIONS)

        with patch("src.ingestion.fred_loader.requests.get", return_value=mock_resp):
            df = fetch_fred_series("T10Y2Y", dry_run=False)

        # The "." sentinel on 2024-03-31 should be NaN
        march_val = df.loc["2024-03-31", "T10Y2Y"]
        assert pd.isna(march_val)

    def test_empty_observations_returns_empty_df(self, monkeypatch):
        monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
        mock_resp = _make_fred_response("T10Y2Y", [])

        with patch("src.ingestion.fred_loader.requests.get", return_value=mock_resp):
            df = fetch_fred_series("T10Y2Y", dry_run=False)

        assert df.empty

    def test_index_is_sorted_ascending(self, monkeypatch):
        monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
        shuffled = _SAMPLE_OBSERVATIONS[::-1]  # reverse order
        mock_resp = _make_fred_response("T10Y2Y", shuffled)

        with patch("src.ingestion.fred_loader.requests.get", return_value=mock_resp):
            df = fetch_fred_series("T10Y2Y", dry_run=False)

        assert df.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# fetch_all_fred_macro
# ---------------------------------------------------------------------------

class TestFetchAllFredMacro:
    def test_dry_run_returns_empty_with_correct_columns(self):
        series_ids = ["T10Y2Y", "GS10", "NFCI"]
        df = fetch_all_fred_macro(series_ids, dry_run=True)
        assert df.empty
        for sid in series_ids:
            assert sid in df.columns

    def test_monthly_resampling(self, monkeypatch):
        monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
        # Daily observations within a month → should produce 1 month-end row
        daily_obs = [
            {"date": f"2024-01-{d:02d}", "value": "1.0"}
            for d in [3, 10, 17, 24, 31]
        ]
        mock_resp = _make_fred_response("T10Y2Y", daily_obs)

        with patch("src.ingestion.fred_loader.requests.get", return_value=mock_resp):
            df = fetch_all_fred_macro(["T10Y2Y"])

        # All daily obs in Jan 2024 → one month-end row
        assert len(df) == 1
        assert df.index[0].day == 31  # last day of Jan

    def test_multiple_series_joined(self, monkeypatch):
        monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
        obs = [{"date": "2024-01-31", "value": "1.0"}]

        with patch(
            "src.ingestion.fred_loader.requests.get",
            return_value=_make_fred_response("T10Y2Y", obs),
        ):
            df = fetch_all_fred_macro(["T10Y2Y", "GS10"])

        assert "T10Y2Y" in df.columns
        assert "GS10" in df.columns

    def test_failed_series_skipped_gracefully(self, monkeypatch):
        monkeypatch.setattr(config, "FRED_API_KEY", "test-key")

        def side_effect(*args, **kwargs):
            sid = kwargs.get("params", {}).get("series_id", "")
            if sid == "BAD_SERIES":
                raise ValueError("Bad series")
            return _make_fred_response(sid, [{"date": "2024-01-31", "value": "1.0"}])

        with patch("src.ingestion.fred_loader.requests.get", side_effect=side_effect):
            # Should not raise; BAD_SERIES should be skipped
            df = fetch_all_fred_macro(["T10Y2Y", "BAD_SERIES"])

        assert "T10Y2Y" in df.columns
        # BAD_SERIES may be absent or all-NaN — either is acceptable


# ---------------------------------------------------------------------------
# upsert_fred_to_db
# ---------------------------------------------------------------------------

class TestUpsertFredToDb:
    def test_empty_df_returns_zero(self):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        from src.database.db_client import initialize_schema

        initialize_schema(conn)
        n = upsert_fred_to_db(conn, pd.DataFrame())
        assert n == 0
        conn.close()

    def test_upsert_inserts_rows(self):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        from src.database.db_client import initialize_schema

        initialize_schema(conn)

        df = pd.DataFrame(
            {"T10Y2Y": [0.35, 0.42], "GS10": [4.1, 4.2]},
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
        )
        n = upsert_fred_to_db(conn, df)
        assert n == 4  # 2 months × 2 series

        rows = conn.execute("SELECT * FROM fred_macro_monthly").fetchall()
        assert len(rows) == 4
        conn.close()

    def test_upsert_idempotent(self):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        from src.database.db_client import initialize_schema

        initialize_schema(conn)

        df = pd.DataFrame(
            {"T10Y2Y": [0.35]},
            index=pd.to_datetime(["2024-01-31"]),
        )
        upsert_fred_to_db(conn, df)
        upsert_fred_to_db(conn, df)  # second call should not duplicate

        count = conn.execute("SELECT COUNT(*) FROM fred_macro_monthly").fetchone()[0]
        assert count == 1
        conn.close()
