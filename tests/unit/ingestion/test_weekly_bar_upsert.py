"""One price bar per ticker per ISO week (review F22, partial-week duplicates).

Alpha Vantage's TIME_SERIES_WEEKLY labels the in-progress week with its latest
trading day (e.g. Tuesday 2026-03-24).  When the completed bar (Friday
2026-03-27) arrives a week later both rows used to be kept, so PGR and a
benchmark could end a target window on different dates.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config
from src.database import db_client
from src.ingestion.multi_ticker_loader import MultiTickerLoader


@pytest.fixture
def conn(tmp_path):
    c = db_client.get_connection(str(tmp_path / "t.db"))
    db_client.initialize_schema(c)
    yield c
    c.close()


def _dates(conn, ticker: str) -> list[str]:
    return [
        r[0] for r in conn.execute(
            "SELECT date FROM daily_prices WHERE ticker = ? ORDER BY date", (ticker,)
        )
    ]


def test_completed_week_bar_replaces_partial_bar(conn) -> None:
    db_client.upsert_prices(
        conn,
        [{"ticker": "VTI", "date": "2026-03-24", "close": 323.18}],
        one_bar_per_week=True,
    )
    db_client.upsert_prices(
        conn,
        [
            {"ticker": "VTI", "date": "2026-03-27", "close": 313.09},
            {"ticker": "VTI", "date": "2026-04-02", "close": 320.00},
        ],
        one_bar_per_week=True,
    )
    assert _dates(conn, "VTI") == ["2026-03-27", "2026-04-02"]


def test_stale_partial_bar_does_not_replace_completed_bar(conn) -> None:
    db_client.upsert_prices(
        conn, [{"ticker": "VTI", "date": "2026-03-27", "close": 313.09}],
        one_bar_per_week=True,
    )
    db_client.upsert_prices(
        conn, [{"ticker": "VTI", "date": "2026-03-24", "close": 323.18}],
        one_bar_per_week=True,
    )
    rows = conn.execute("SELECT date, close FROM daily_prices").fetchall()
    assert [tuple(r) for r in rows] == [("2026-03-27", 313.09)]


def test_same_payload_duplicates_collapse_to_latest(conn) -> None:
    db_client.upsert_prices(
        conn,
        [
            {"ticker": "VFH", "date": "2026-04-30", "close": 128.20},
            {"ticker": "VFH", "date": "2026-05-01", "close": 127.66},
        ],
        one_bar_per_week=True,
    )
    assert _dates(conn, "VFH") == ["2026-05-01"]


def test_other_tickers_untouched(conn) -> None:
    db_client.upsert_prices(conn, [
        {"ticker": "BND", "date": "2026-03-24", "close": 73.26},
        {"ticker": "BND", "date": "2026-03-27", "close": 73.11},
    ])
    db_client.upsert_prices(
        conn, [{"ticker": "VTI", "date": "2026-03-27", "close": 313.09}],
        one_bar_per_week=True,
    )
    assert _dates(conn, "BND") == ["2026-03-24", "2026-03-27"]


def test_dedupe_weekly_price_bars_removes_existing_duplicates(conn) -> None:
    db_client.upsert_prices(conn, [
        {"ticker": "BND", "date": "2026-03-24", "close": 73.26},
        {"ticker": "BND", "date": "2026-03-27", "close": 73.11},
        {"ticker": "PGR", "date": "2026-03-24", "close": 206.21},
        {"ticker": "PGR", "date": "2026-03-27", "close": 198.84},
        {"ticker": "PGR", "date": "2026-04-02", "close": 199.00},
    ])
    removed = db_client.dedupe_weekly_price_bars(conn)
    assert removed == 2
    assert _dates(conn, "BND") == ["2026-03-27"]
    assert _dates(conn, "PGR") == ["2026-03-27", "2026-04-02"]


def test_iso_week_spans_year_boundary(conn) -> None:
    # 2026-12-31 (Thu) and 2027-01-01 (Fri) are both ISO week 2026-W53.
    db_client.upsert_prices(
        conn,
        [
            {"ticker": "VTI", "date": "2026-12-31", "close": 1.0},
            {"ticker": "VTI", "date": "2027-01-01", "close": 2.0},
        ],
        one_bar_per_week=True,
    )
    assert _dates(conn, "VTI") == ["2027-01-01"]


@patch("src.ingestion.multi_ticker_loader.build_retry_session")
def test_loader_writes_one_bar_per_week(mock_session, conn, monkeypatch) -> None:
    monkeypatch.setattr(config, "AV_API_KEY", "test-key")
    db_client.upsert_prices(conn, [{"ticker": "VTI", "date": "2026-03-24", "close": 323.18}])
    payload = {
        "Weekly Time Series": {
            "2026-03-27": {"1. open": "1", "2. high": "1", "3. low": "1",
                           "4. close": "313.09", "5. volume": "10"},
            "2026-03-20": {"1. open": "1", "2. high": "1", "3. low": "1",
                           "4. close": "320.00", "5. volume": "10"},
        }
    }
    session = MagicMock()
    session.get.return_value = MagicMock(json=lambda: payload)
    mock_session.return_value = session
    MultiTickerLoader(conn).fetch_ticker_prices("VTI", force_refresh=True)
    assert _dates(conn, "VTI") == ["2026-03-20", "2026-03-27"]
