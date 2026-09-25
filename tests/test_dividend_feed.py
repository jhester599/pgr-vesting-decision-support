"""Dividend feed robustness (review F08).

* The first dividend call of a batch must follow a sleep: it used to fire
  straight after the price batch, receive an AV "Information" advisory and be
  skipped silently, so PGR dividends stopped updating after 2026-03-26.
* "Information" advisories are retried with backoff before giving up.
* Per-ticker dividend freshness: max(ex_date) >= last price date minus
  1.5 x the ticker's usual payment interval.
* ETF dividends are refreshed by a budget-aware monthly job.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import config
from src.database import db_client
from src.ingestion.multi_dividend_loader import MultiDividendLoader

_ADVISORY = {"Information": "Please consider spreading out your free API requests."}


@pytest.fixture
def conn(tmp_path):
    c = db_client.get_connection(str(tmp_path / "t.db"))
    db_client.initialize_schema(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def av_api_key(monkeypatch):
    monkeypatch.setattr(config, "AV_API_KEY", "test-key")


def _div_payload(ticker: str) -> dict:
    return {"symbol": ticker, "data": [{"ex_dividend_date": "2026-07-02", "amount": "0.10"}]}


def _response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


class TestDividendLoaderPacing:
    @patch("src.ingestion.multi_dividend_loader.time.sleep")
    @patch("src.ingestion.multi_dividend_loader.build_retry_session")
    def test_first_call_follows_a_sleep(self, mock_session, mock_sleep, conn) -> None:
        events: list[str] = []
        mock_sleep.side_effect = lambda s: events.append(f"sleep:{s}")

        def get(url, params=None, timeout=None):
            events.append(f"get:{params['symbol']}")
            return _response(_div_payload(params["symbol"]))

        mock_session.return_value = MagicMock(get=MagicMock(side_effect=get))
        MultiDividendLoader(conn).fetch_for_tickers(["PGR"], sleep_between=13)
        assert events[0] == "sleep:13"
        assert events[1] == "get:PGR"

    @patch("src.ingestion.multi_dividend_loader.time.sleep")
    @patch("src.ingestion.multi_dividend_loader.build_retry_session")
    def test_advisory_is_retried_with_backoff(self, mock_session, mock_sleep, conn) -> None:
        responses = [_response(_ADVISORY), _response(_div_payload("PGR"))]
        mock_session.return_value = MagicMock(get=MagicMock(side_effect=responses))
        results = MultiDividendLoader(conn).fetch_for_tickers(
            ["PGR"], sleep_between=13, advisory_backoff=30.0,
        )
        assert results["PGR"] == 1
        assert db_client.get_dividends(conn, "PGR").index[-1] == pd.Timestamp("2026-07-02")
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        assert 30.0 in sleeps

    @patch("src.ingestion.multi_dividend_loader.time.sleep")
    @patch("src.ingestion.multi_dividend_loader.build_retry_session")
    def test_persistent_advisory_gives_up_after_retries(
        self, mock_session, mock_sleep, conn
    ) -> None:
        def get(url, params=None, timeout=None):
            if params["symbol"] == "VTI":
                return _response(_ADVISORY)
            return _response(_div_payload(params["symbol"]))

        session = MagicMock(get=MagicMock(side_effect=get))
        mock_session.return_value = session
        results = MultiDividendLoader(conn).fetch_for_tickers(
            ["VTI", "BND"], sleep_between=0, max_advisory_retries=2, advisory_backoff=1.0,
        )
        assert results["VTI"] is None
        assert results["BND"] == 1
        symbols = [c.kwargs["params"]["symbol"] for c in session.get.call_args_list]
        assert symbols == ["VTI", "VTI", "VTI", "BND"]
        backoffs = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] > 0]
        assert backoffs == [1.0, 2.0]  # exponential


class TestDividendFreshness:
    @staticmethod
    def _seed(conn, ticker: str, ex_dates: list[str], last_price: str) -> None:
        db_client.upsert_dividends(conn, [
            {"ticker": ticker, "ex_date": d, "amount": 0.5, "source": "test"}
            for d in ex_dates
        ])
        db_client.upsert_prices(conn, [{"ticker": ticker, "date": last_price, "close": 10.0}])

    def test_quarterly_payer_missing_two_payments_is_stale(self, conn) -> None:
        self._seed(conn, "VOO", ["2025-03-24", "2025-06-24", "2025-09-24", "2025-12-22"],
                   "2026-09-18")
        (row,) = db_client.check_dividend_freshness(conn, ["VOO"])
        assert row["status"] == "STALE"
        assert row["interval_days"] == pytest.approx(91, abs=2)

    def test_quarterly_payer_within_interval_is_ok(self, conn) -> None:
        self._seed(conn, "VOO", ["2025-12-22", "2026-03-24", "2026-06-24"], "2026-09-18")
        (row,) = db_client.check_dividend_freshness(conn, ["VOO"])
        assert row["status"] == "OK"

    def test_annual_payer_uses_its_own_interval(self, conn) -> None:
        self._seed(conn, "DBC", ["2023-12-18", "2024-12-23", "2025-12-22"], "2026-09-18")
        (row,) = db_client.check_dividend_freshness(conn, ["DBC"])
        assert row["status"] == "OK"

    def test_non_payer_is_not_flagged(self, conn) -> None:
        db_client.upsert_prices(conn, [{"ticker": "GLD", "date": "2026-09-18", "close": 1.0}])
        (row,) = db_client.check_dividend_freshness(conn, ["GLD"])
        assert row["status"] == "NO_HISTORY"


class TestDividendRefreshSelection:
    def test_due_tickers_are_capped_by_remaining_budget(self, conn) -> None:
        from scripts.weekly_fetch import select_dividend_refresh_tickers

        for t in ["VTI", "BND", "VOO"]:
            db_client.update_ingestion_metadata(conn, t, "dividends", 1)
        conn.execute(
            "UPDATE ingestion_metadata SET last_fetched = '2026-08-01T00:00:00+00:00' "
            "WHERE ticker IN ('VTI', 'VOO')"
        )
        conn.execute(
            "UPDATE ingestion_metadata SET last_fetched = '2026-09-20T00:00:00+00:00' "
            "WHERE ticker = 'BND'"
        )
        conn.commit()
        due = select_dividend_refresh_tickers(
            conn, ["VTI", "BND", "VOO", "VEA"], today=date(2026, 9, 23), max_calls=2,
        )
        # VEA never fetched (most overdue), then the oldest of VTI/VOO; BND fresh.
        assert due[0] == "VEA"
        assert len(due) == 2
        assert "BND" not in due

    def test_no_calls_when_budget_spent(self, conn) -> None:
        from scripts.weekly_fetch import select_dividend_refresh_tickers

        assert select_dividend_refresh_tickers(
            conn, ["VTI"], today=date(2026, 9, 23), max_calls=0,
        ) == []


@patch("src.ingestion.multi_dividend_loader.build_retry_session")
def test_successful_fetch_without_dividends_records_fetch_time(mock_session, conn) -> None:
    mock_session.return_value = MagicMock(
        get=MagicMock(return_value=_response({"symbol": "GLD", "data": []}))
    )
    assert MultiDividendLoader(conn).fetch_dividends("GLD") == 0
    meta = db_client.get_ingestion_metadata(conn, "GLD", "dividends")
    assert meta is not None and meta["last_fetched"]
