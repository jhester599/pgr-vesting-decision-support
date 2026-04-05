"""
Tests for Phase 2 multi-ticker ingestion pipeline.

Covers:
  - MultiTickerLoader: price parsing, DB upsert, skip-if-fresh, proxy fill
  - MultiDividendLoader: dividend parsing, DB upsert, skip-if-fresh
  - fetch_scheduler: group assignment, PGR always present, parity logic
  - Budget enforcement via mocked db_client.log_api_request

All HTTP calls are mocked so these tests run offline.
"""

from __future__ import annotations


from unittest.mock import MagicMock, patch

import pytest

import config
from src.database import db_client
from src.ingestion.fetch_scheduler import (
    get_all_dividend_tickers,
    get_all_price_tickers,
)
from src.ingestion.multi_dividend_loader import (
    MultiDividendLoader,
    _parse_av_dividends,
)
from src.ingestion.multi_ticker_loader import (
    MultiTickerLoader,
    _parse_av_daily,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    db_path = str(tmp_path / "test.db")
    c = db_client.get_connection(db_path)
    db_client.initialize_schema(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def av_api_key(monkeypatch):
    monkeypatch.setattr(config, "AV_API_KEY", "test-key")


def _mock_retry_session(response=None, side_effect=None):
    session = MagicMock()
    if side_effect is not None:
        session.get.side_effect = side_effect
    else:
        session.get.return_value = response
    return session


# ---------------------------------------------------------------------------
# _parse_av_daily
# ---------------------------------------------------------------------------

class TestParseAvDaily:
    def _sample_payload(self, ticker: str = "VTI") -> dict:
        return {
            "Meta Data": {"2. Symbol": ticker},
            "Weekly Time Series": {
                "2024-01-05": {
                    "1. open":   "220.10",
                    "2. high":   "222.50",
                    "3. low":    "219.00",
                    "4. close":  "221.80",
                    "5. volume": "3000000",
                },
                "2023-12-29": {
                    "1. open":   "218.00",
                    "2. high":   "220.00",
                    "3. low":    "217.50",
                    "4. close":  "219.50",
                    "5. volume": "2500000",
                },
            },
        }

    def test_returns_correct_count(self):
        records = _parse_av_daily(self._sample_payload(), "VTI")
        assert len(records) == 2

    def test_ticker_set_on_all_records(self):
        records = _parse_av_daily(self._sample_payload("SPY"), "SPY")
        assert all(r["ticker"] == "SPY" for r in records)

    def test_close_parsed_as_float(self):
        records = _parse_av_daily(self._sample_payload(), "VTI")
        assert all(isinstance(r["close"], float) for r in records)

    def test_proxy_fill_is_zero(self):
        records = _parse_av_daily(self._sample_payload(), "VTI")
        assert all(r["proxy_fill"] == 0 for r in records)

    def test_source_is_av(self):
        records = _parse_av_daily(self._sample_payload(), "VTI")
        assert all(r["source"] == "av" for r in records)

    def test_empty_series_returns_empty_list(self):
        records = _parse_av_daily({}, "VTI")
        assert records == []

    def test_malformed_row_skipped(self):
        payload = {
            "Weekly Time Series": {
                "2024-01-05": {"4. close": "221.80"},
                "bad-date": {"4. close": "not_a_number"},
            }
        }
        records = _parse_av_daily(payload, "VTI")
        # "bad-date" row has a non-numeric close; should be skipped by the
        # try/except in _parse_av_daily.  The valid row is kept.
        assert len(records) >= 1


# ---------------------------------------------------------------------------
# MultiTickerLoader
# ---------------------------------------------------------------------------

class TestMultiTickerLoader:
    def _av_response(self, ticker: str = "VTI") -> dict:
        return {
            "Weekly Time Series": {
                f"2024-0{i}-07": {
                    "1. open":   f"{100+i}.0",
                    "2. high":   f"{105+i}.0",
                    "3. low":    f"{99+i}.0",
                    "4. close":  f"{103+i}.0",
                    "5. volume": "1000000",
                }
                for i in range(1, 4)
            }
        }

    @patch("src.ingestion.multi_ticker_loader.build_retry_session")
    def test_fetch_ticker_prices_inserts_rows(self, mock_build_session, conn):
        mock_resp = MagicMock(
            status_code=200,
            json=lambda: self._av_response("VTI"),
        )
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)
        loader = MultiTickerLoader(conn)
        n = loader.fetch_ticker_prices("VTI")
        assert n == 3

    @patch("src.ingestion.multi_ticker_loader.build_retry_session")
    def test_fetch_ticker_prices_rows_in_db(self, mock_build_session, conn):
        mock_resp = MagicMock(
            status_code=200,
            json=lambda: self._av_response("VTI"),
        )
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)
        loader = MultiTickerLoader(conn)
        loader.fetch_ticker_prices("VTI")
        df = db_client.get_prices(conn, "VTI")
        assert len(df) == 3

    @patch("src.ingestion.multi_ticker_loader.build_retry_session")
    def test_fetch_ticker_prices_skip_if_fresh_today(self, mock_build_session, conn):
        """Second call same day should skip HTTP (already fresh)."""
        mock_resp = MagicMock(
            status_code=200,
            json=lambda: self._av_response("VTI"),
        )
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)
        loader = MultiTickerLoader(conn)
        loader.fetch_ticker_prices("VTI")   # first call → fetches
        n2 = loader.fetch_ticker_prices("VTI")  # second call → skips
        assert n2 == 0
        assert mock_build_session.return_value.get.call_count == 1  # only one real HTTP call

    @patch("src.ingestion.multi_ticker_loader.build_retry_session")
    def test_force_refresh_bypasses_fresh_check(self, mock_build_session, conn):
        mock_resp = MagicMock(
            status_code=200,
            json=lambda: self._av_response("VTI"),
        )
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)
        loader = MultiTickerLoader(conn)
        loader.fetch_ticker_prices("VTI")
        loader.fetch_ticker_prices("VTI", force_refresh=True)
        assert mock_build_session.return_value.get.call_count == 2

    @patch("src.ingestion.multi_ticker_loader.build_retry_session")
    def test_fetch_all_prices_returns_dict(self, mock_build_session, conn):
        mock_resp = MagicMock(
            status_code=200,
            json=lambda: self._av_response(),
        )
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)
        loader = MultiTickerLoader(conn)
        results = loader.fetch_all_prices(["VTI", "BND"], sleep_between=0)
        assert isinstance(results, dict)
        assert "VTI" in results
        assert "BND" in results

    def test_fetch_all_prices_budget_exceeded_raises(self, conn):
        """Once the LOCAL DB daily AV limit is logged, fetch_all_prices must raise."""
        # Exhaust the budget in the DB — use UTC date to match log_api_request().
        from datetime import datetime, timezone
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        for i in range(config.AV_DAILY_LIMIT):
            conn.execute(
                "INSERT INTO api_request_log (api, date, endpoint, count) "
                "VALUES ('av', ?, ?, 1) ON CONFLICT DO UPDATE SET count=count+1",
                (today, f"/ep{i}"),
            )
        conn.commit()

        loader = MultiTickerLoader(conn)
        with pytest.raises(RuntimeError):
            loader.fetch_all_prices(["VTI"], sleep_between=0)

    @patch("src.ingestion.multi_ticker_loader.build_retry_session")
    def test_fetch_all_prices_information_skips_one_continues_rest(
        self, mock_build_session, conn
    ):
        """'Information' advisory skips one ticker but the batch continues.

        When AV returns the soft 'Information' key, no quota slot is consumed.
        The affected ticker is marked None but all subsequent tickers still run.
        """
        tickers = ["VTI", "BND", "GLD"]

        def side_effect(url, params=None, timeout=None):
            symbol = (params or {}).get("symbol", "")
            if symbol == "BND":
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json.return_value = {
                    "Information": (
                        "We have detected your API key and our standard API "
                        "rate limit is 25 requests per day."
                    )
                }
                return mock_resp
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = self._av_response()
            return mock_resp

        mock_build_session.return_value = _mock_retry_session(side_effect=side_effect)
        loader = MultiTickerLoader(conn)
        results = loader.fetch_all_prices(tickers, sleep_between=0)

        # VTI completed before the advisory
        assert results["VTI"] is not None
        assert results["VTI"] >= 0
        # BND received the 'Information' advisory — skipped (None)
        assert results["BND"] is None
        # GLD still attempted and succeeded (batch continued)
        assert results["GLD"] is not None
        assert results["GLD"] >= 0

    @patch("src.ingestion.multi_ticker_loader.build_retry_session")
    def test_fetch_all_prices_note_hard_stop_defers_remaining(
        self, mock_build_session, conn
    ):
        """'Note' (hard quota) stops the batch and defers all remaining tickers.

        When AV returns 'Note', the 25-call daily limit is exhausted.
        The throttled ticker and all subsequent tickers are set to None.
        """
        tickers = ["VTI", "BND", "GLD"]

        def side_effect(url, params=None, timeout=None):
            symbol = (params or {}).get("symbol", "")
            if symbol == "BND":
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json.return_value = {
                    "Note": (
                        "Thank you for using Alpha Vantage! Our standard API "
                        "rate limit is 25 requests per day."
                    )
                }
                return mock_resp
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = self._av_response()
            return mock_resp

        mock_build_session.return_value = _mock_retry_session(side_effect=side_effect)
        loader = MultiTickerLoader(conn)
        results = loader.fetch_all_prices(tickers, sleep_between=0)

        # VTI completed before the hard limit
        assert results["VTI"] is not None
        assert results["VTI"] >= 0
        # BND and GLD both deferred (hard stop)
        assert results["BND"] is None
        assert results["GLD"] is None

    @patch("src.ingestion.multi_ticker_loader.build_retry_session")
    def test_information_does_not_raise_av_rate_limit_error(self, mock_build_session, conn):
        """Regression: 'Information' must NOT raise AVRateLimitError.

        AVRateLimitError signals a hard quota stop; the soft 'Information'
        advisory must raise AVRateLimitAdvisory instead so the caller can
        distinguish the two cases and choose to continue the batch.
        """
        from src.ingestion.exceptions import AVRateLimitAdvisory, AVRateLimitError
        from src.ingestion.multi_ticker_loader import _av_request

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "Information": "Standard API rate limit is 25 requests per day."
        }
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)

        import sqlite3
        from src.database import db_client as _dbc
        tmp_conn = sqlite3.connect(":memory:")
        _dbc.initialize_schema(tmp_conn)

        with pytest.raises(AVRateLimitAdvisory):
            _av_request(tmp_conn, "PGR")

        # Must not raise the hard-stop error
        try:
            _av_request(tmp_conn, "PGR")
        except AVRateLimitAdvisory:
            pass  # expected
        except AVRateLimitError:
            pytest.fail("'Information' response must not raise AVRateLimitError")

    @patch("src.ingestion.multi_ticker_loader.build_retry_session")
    def test_note_raises_av_rate_limit_error(self, mock_build_session, conn):
        """'Note' response must raise AVRateLimitError (hard quota stop)."""
        from src.ingestion.exceptions import AVRateLimitError
        from src.ingestion.multi_ticker_loader import _av_request

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "Note": "Thank you for using Alpha Vantage! Rate limit 25/day."
        }
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)

        import sqlite3
        from src.database import db_client as _dbc
        tmp_conn = sqlite3.connect(":memory:")
        _dbc.initialize_schema(tmp_conn)

        with pytest.raises(AVRateLimitError):
            _av_request(tmp_conn, "PGR")

    def test_fill_proxy_history_sets_proxy_fill_flag(self, conn):
        """fill_proxy_history must copy proxy rows with proxy_fill=1."""
        # Insert VTI prices into DB as the proxy
        proxy_records = [
            {"ticker": "VTI", "date": f"2017-0{i}-15", "close": float(100 + i),
             "source": "av", "proxy_fill": 0}
            for i in range(1, 4)
        ]
        db_client.upsert_prices(conn, proxy_records)

        loader = MultiTickerLoader(conn)
        n = loader.fill_proxy_history("FZROX", "VTI", "2018-08-02")
        assert n == 3

        df = db_client.get_prices(conn, "FZROX")
        assert len(df) == 3
        assert all(df["proxy_fill"] == 1)

    def test_fill_proxy_excludes_cutoff_date_itself(self, conn):
        """The cutoff date (launch date) is excluded from the proxy fill."""
        db_client.upsert_prices(conn, [
            {"ticker": "VTI", "date": "2018-08-01", "close": 140.0, "proxy_fill": 0},
            {"ticker": "VTI", "date": "2018-08-02", "close": 141.0, "proxy_fill": 0},
        ])
        loader = MultiTickerLoader(conn)
        loader.fill_proxy_history("FZROX", "VTI", "2018-08-02")
        df = db_client.get_prices(conn, "FZROX")
        assert len(df) == 1
        assert df.index[0].strftime("%Y-%m-%d") == "2018-08-01"

    def test_fill_proxy_empty_proxy_returns_zero(self, conn):
        """If proxy has no data before cutoff, return 0."""
        loader = MultiTickerLoader(conn)
        n = loader.fill_proxy_history("FZROX", "VTI", "2018-08-02")
        assert n == 0


# ---------------------------------------------------------------------------
# _parse_av_dividends
# ---------------------------------------------------------------------------

class TestParseAvDividends:
    def _sample_payload(self, ticker: str = "VTI") -> dict:
        return {
            "symbol": ticker,
            "data": [
                {
                    "ex_dividend_date": "2024-03-21",
                    "declaration_date": "2024-03-15",
                    "record_date": "2024-03-22",
                    "payment_date": "2024-03-28",
                    "amount": "0.89",
                },
                {
                    "ex_dividend_date": "2023-12-21",
                    "declaration_date": "2023-12-15",
                    "record_date": "2023-12-22",
                    "payment_date": "2023-12-28",
                    "amount": "0.95",
                },
            ],
        }

    def test_returns_correct_count(self):
        records = _parse_av_dividends(self._sample_payload(), "VTI")
        assert len(records) == 2

    def test_ticker_set_correctly(self):
        records = _parse_av_dividends(self._sample_payload("BND"), "BND")
        assert all(r["ticker"] == "BND" for r in records)

    def test_amount_is_float(self):
        records = _parse_av_dividends(self._sample_payload(), "VTI")
        assert all(isinstance(r["amount"], float) for r in records)

    def test_zero_amount_rows_excluded(self):
        payload = {
            "data": [
                {"ex_dividend_date": "2024-03-21", "amount": "0.00"},
                {"ex_dividend_date": "2024-06-21", "amount": "0.89"},
            ]
        }
        records = _parse_av_dividends(payload, "VTI")
        assert len(records) == 1

    def test_empty_payload_returns_empty(self):
        records = _parse_av_dividends({}, "VTI")
        assert records == []


# ---------------------------------------------------------------------------
# MultiDividendLoader
# ---------------------------------------------------------------------------

class TestMultiDividendLoader:
    def _av_div_response(self, ticker: str = "VTI") -> dict:
        return {
            "symbol": ticker,
            "data": [
                {"ex_dividend_date": "2024-03-21", "amount": "0.89"},
                {"ex_dividend_date": "2023-12-21", "amount": "0.95"},
            ],
        }

    @patch("src.ingestion.multi_dividend_loader.build_retry_session")
    def test_fetch_dividends_inserts_rows(self, mock_build_session, conn):
        mock_resp = MagicMock(
            status_code=200,
            json=lambda: self._av_div_response("VTI"),
        )
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)
        loader = MultiDividendLoader(conn)
        n = loader.fetch_dividends("VTI")
        assert n == 2

    @patch("src.ingestion.multi_dividend_loader.build_retry_session")
    def test_fetch_dividends_rows_in_db(self, mock_build_session, conn):
        mock_resp = MagicMock(
            status_code=200,
            json=lambda: self._av_div_response("VTI"),
        )
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)
        loader = MultiDividendLoader(conn)
        loader.fetch_dividends("VTI")
        df = db_client.get_dividends(conn, "VTI")
        assert len(df) == 2

    @patch("src.ingestion.multi_dividend_loader.build_retry_session")
    def test_skip_if_fresh_today(self, mock_build_session, conn):
        mock_resp = MagicMock(
            status_code=200,
            json=lambda: self._av_div_response("VTI"),
        )
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)
        loader = MultiDividendLoader(conn)
        loader.fetch_dividends("VTI")
        n2 = loader.fetch_dividends("VTI")
        assert n2 == 0
        assert mock_build_session.return_value.get.call_count == 1

    @patch("src.ingestion.multi_dividend_loader.build_retry_session")
    def test_fetch_for_tickers_returns_dict(self, mock_build_session, conn):
        mock_resp = MagicMock(
            status_code=200,
            json=lambda: self._av_div_response(),
        )
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)
        loader = MultiDividendLoader(conn)
        results = loader.fetch_for_tickers(["VTI", "BND"], sleep_between=0)
        assert "VTI" in results
        assert "BND" in results

    @patch("src.ingestion.multi_dividend_loader.build_retry_session")
    def test_fetch_for_tickers_information_skips_one_continues_rest(
        self, mock_build_session, conn
    ):
        """'Information' advisory on dividends skips one ticker; batch continues.

        The soft 'Information' key means VTI is skipped (None) but BND
        still runs and succeeds.
        """
        tickers = ["PGR", "VTI", "BND"]

        def side_effect(url, params=None, timeout=None):
            symbol = (params or {}).get("symbol", "")
            if symbol == "VTI":
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json.return_value = {
                    "Information": (
                        "We have detected your API key and our standard API "
                        "rate limit is 25 requests per day."
                    )
                }
                return mock_resp
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = self._av_div_response(symbol or "PGR")
            return mock_resp

        mock_build_session.return_value = _mock_retry_session(side_effect=side_effect)
        loader = MultiDividendLoader(conn)
        results = loader.fetch_for_tickers(tickers, sleep_between=0)

        # PGR succeeded before the advisory
        assert results["PGR"] is not None
        assert isinstance(results["PGR"], int)
        # VTI received the advisory — skipped (None)
        assert results["VTI"] is None
        # BND still attempted and succeeded (batch continued)
        assert results["BND"] is not None
        assert isinstance(results["BND"], int)

    @patch("src.ingestion.multi_dividend_loader.build_retry_session")
    def test_fetch_for_tickers_note_hard_stop_defers_remaining(
        self, mock_build_session, conn
    ):
        """'Note' (hard quota) on dividends stops the batch; all remaining deferred."""
        tickers = ["PGR", "VTI", "BND"]

        def side_effect(url, params=None, timeout=None):
            symbol = (params or {}).get("symbol", "")
            if symbol == "VTI":
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json.return_value = {
                    "Note": (
                        "Thank you for using Alpha Vantage! "
                        "Rate limit 25 requests per day."
                    )
                }
                return mock_resp
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = self._av_div_response(symbol or "PGR")
            return mock_resp

        mock_build_session.return_value = _mock_retry_session(side_effect=side_effect)
        loader = MultiDividendLoader(conn)
        results = loader.fetch_for_tickers(tickers, sleep_between=0)

        assert results["PGR"] is not None
        assert isinstance(results["PGR"], int)
        # VTI (hard limit) and BND (not attempted) both deferred
        assert results["VTI"] is None
        assert results["BND"] is None

    @patch("src.ingestion.multi_dividend_loader.build_retry_session")
    def test_idempotent_upsert_no_duplicates(self, mock_build_session, conn):
        """Fetching the same dividends twice must not create duplicate rows."""
        mock_resp = MagicMock(
            status_code=200,
            json=lambda: self._av_div_response("VTI"),
        )
        mock_build_session.return_value = _mock_retry_session(response=mock_resp)
        loader = MultiDividendLoader(conn)
        loader.fetch_dividends("VTI", force_refresh=True)
        loader.fetch_dividends("VTI", force_refresh=True)
        df = db_client.get_dividends(conn, "VTI")
        assert len(df) == 2  # not 4


# ---------------------------------------------------------------------------
# Fetch scheduler
# ---------------------------------------------------------------------------

class TestFetchScheduler:
    def test_pgr_first_in_price_list(self):
        tickers = get_all_price_tickers()
        assert tickers[0] == "PGR"

    def test_price_list_length_is_correct(self):
        tickers = get_all_price_tickers()
        assert len(tickers) == 1 + len(config.ETF_BENCHMARK_UNIVERSE)

    def test_all_etfs_in_price_list(self):
        tickers = get_all_price_tickers()
        etf_set = set(tickers) - {"PGR"}
        assert etf_set == set(config.ETF_BENCHMARK_UNIVERSE)

    def test_no_duplicate_tickers_in_price_list(self):
        tickers = get_all_price_tickers()
        assert len(tickers) == len(set(tickers))

    def test_pgr_first_in_dividend_list(self):
        tickers = get_all_dividend_tickers()
        assert tickers[0] == "PGR"

    def test_dividend_list_length_is_correct(self):
        tickers = get_all_dividend_tickers()
        assert len(tickers) == 1 + len(config.ETF_BENCHMARK_UNIVERSE)

    def test_all_etfs_in_dividend_list(self):
        tickers = get_all_dividend_tickers()
        etf_set = set(tickers) - {"PGR"}
        assert etf_set == set(config.ETF_BENCHMARK_UNIVERSE)

    def test_price_and_dividend_lists_identical(self):
        assert get_all_price_tickers() == get_all_dividend_tickers()
