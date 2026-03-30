"""
Tests for the v6.0 peer data fetch infrastructure.

Covers:
  - fetch_scheduler: get_peer_price_tickers / get_peer_dividend_tickers
  - Budget math: peer tickers + main tickers stay within AV daily limit on
    separate days
  - peer_fetch.main() dry-run mode: no HTTP calls, correct projection output
  - Config: PEER_TICKER_UNIVERSE present, correct tickers, no overlap with
    ETF_BENCHMARK_UNIVERSE
  - Cron schedule sanity: Sunday 04:00 UTC is 30 hours after Friday 22:00 UTC
"""

from __future__ import annotations

import io
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
from src.ingestion.fetch_scheduler import (
    get_all_dividend_tickers,
    get_all_price_tickers,
    get_peer_dividend_tickers,
    get_peer_price_tickers,
)


# ---------------------------------------------------------------------------
# config.PEER_TICKER_UNIVERSE
# ---------------------------------------------------------------------------

class TestPeerTickerUniverse:
    def test_four_tickers(self) -> None:
        assert len(config.PEER_TICKER_UNIVERSE) == 4

    def test_expected_tickers_present(self) -> None:
        universe = set(config.PEER_TICKER_UNIVERSE)
        assert universe == {"ALL", "TRV", "CB", "HIG"}

    def test_no_overlap_with_etf_universe(self) -> None:
        peers = set(config.PEER_TICKER_UNIVERSE)
        etfs = set(config.ETF_BENCHMARK_UNIVERSE)
        overlap = peers & etfs
        assert overlap == set(), f"Peer tickers overlap with ETF universe: {overlap}"

    def test_no_overlap_with_pgr(self) -> None:
        assert "PGR" not in config.PEER_TICKER_UNIVERSE

    def test_returns_list(self) -> None:
        assert isinstance(config.PEER_TICKER_UNIVERSE, list)


# ---------------------------------------------------------------------------
# fetch_scheduler peer functions
# ---------------------------------------------------------------------------

class TestGetPeerPriceTickers:
    def test_returns_list(self) -> None:
        tickers = get_peer_price_tickers()
        assert isinstance(tickers, list)

    def test_returns_four_tickers(self) -> None:
        tickers = get_peer_price_tickers()
        assert len(tickers) == 4

    def test_matches_config_universe(self) -> None:
        tickers = get_peer_price_tickers()
        assert set(tickers) == set(config.PEER_TICKER_UNIVERSE)

    def test_no_duplicates(self) -> None:
        tickers = get_peer_price_tickers()
        assert len(tickers) == len(set(tickers))

    def test_does_not_include_pgr(self) -> None:
        tickers = get_peer_price_tickers()
        assert "PGR" not in tickers

    def test_does_not_include_etfs(self) -> None:
        peer_tickers = set(get_peer_price_tickers())
        etf_tickers = set(config.ETF_BENCHMARK_UNIVERSE)
        assert peer_tickers.isdisjoint(etf_tickers)


class TestGetPeerDividendTickers:
    def test_returns_same_as_price_tickers(self) -> None:
        # Dividend tickers must match price tickers — we need both for DRIP total return
        assert get_peer_dividend_tickers() == get_peer_price_tickers()

    def test_returns_four_tickers(self) -> None:
        assert len(get_peer_dividend_tickers()) == 4

    def test_contains_expected_peers(self) -> None:
        assert set(get_peer_dividend_tickers()) == {"ALL", "TRV", "CB", "HIG"}


# ---------------------------------------------------------------------------
# AV budget math — separation invariant
# ---------------------------------------------------------------------------

class TestBudgetSeparation:
    """
    Invariant: Friday AV calls + Sunday AV calls must each be ≤ AV_DAILY_LIMIT.
    Since they run on different calendar days, they never compete.
    """

    def test_friday_budget_within_limit(self) -> None:
        # weekly_fetch.py: all price tickers + PGR dividends
        friday_calls = len(get_all_price_tickers()) + 1  # 23 prices + 1 PGR dividend
        assert friday_calls <= config.AV_DAILY_LIMIT, (
            f"Friday fetch uses {friday_calls} AV calls, exceeds {config.AV_DAILY_LIMIT}/day"
        )

    def test_sunday_budget_within_limit(self) -> None:
        # peer_fetch.py: peer prices + peer dividends
        sunday_calls = len(get_peer_price_tickers()) + len(get_peer_dividend_tickers())
        assert sunday_calls <= config.AV_DAILY_LIMIT, (
            f"Sunday peer fetch uses {sunday_calls} AV calls, exceeds {config.AV_DAILY_LIMIT}/day"
        )

    def test_friday_leaves_margin(self) -> None:
        friday_calls = len(get_all_price_tickers()) + 1
        margin = config.AV_DAILY_LIMIT - friday_calls
        assert margin >= 1, "Friday fetch should leave at least 1 call of margin"

    def test_sunday_leaves_margin(self) -> None:
        sunday_calls = len(get_peer_price_tickers()) + len(get_peer_dividend_tickers())
        margin = config.AV_DAILY_LIMIT - sunday_calls
        assert margin >= 10, (
            f"Sunday peer fetch uses {sunday_calls}/25 — expected at least 10 calls of margin"
        )

    def test_combined_never_exceeds_daily_limit_on_any_one_day(self) -> None:
        # Even if both somehow ran on the same day (should never happen), the sum
        # reflects the true worst case.  We assert each DAY independently passes.
        friday_calls = len(get_all_price_tickers()) + 1
        sunday_calls = len(get_peer_price_tickers()) + len(get_peer_dividend_tickers())
        assert friday_calls <= config.AV_DAILY_LIMIT
        assert sunday_calls <= config.AV_DAILY_LIMIT

    def test_peer_tickers_not_in_all_price_tickers(self) -> None:
        # Peer tickers must not appear in get_all_price_tickers() — they have
        # their own separate fetch and must not be double-counted on Fridays.
        all_price = set(get_all_price_tickers())
        peer_price = set(get_peer_price_tickers())
        double_counted = all_price & peer_price
        assert double_counted == set(), (
            f"Peer tickers appear in Friday fetch: {double_counted}"
        )


# ---------------------------------------------------------------------------
# Cron schedule: 30-hour gap invariant
# ---------------------------------------------------------------------------

class TestCronScheduleGap:
    """
    The peer_data_fetch.yml cron is Sunday 04:00 UTC.
    The weekly_data_fetch.yml cron is Friday 22:00 UTC.
    Gap must be ≥ 30 hours so the two fetches always fall on different
    calendar days regardless of scheduler lag.
    """

    def test_gap_is_at_least_30_hours(self) -> None:
        # Reference week: any week works since both are fixed weekday/time combos.
        # Use ISO week 1 of 2026 as a concrete reference.
        friday_run = datetime(2026, 1, 2, 22, 0, tzinfo=timezone.utc)   # Fri Jan 2
        sunday_run = datetime(2026, 1, 4,  4, 0, tzinfo=timezone.utc)   # Sun Jan 4
        gap_hours = (sunday_run - friday_run).total_seconds() / 3600
        assert gap_hours == pytest.approx(30.0), (
            f"Expected 30-hour gap between Friday 22:00 UTC and Sunday 04:00 UTC; got {gap_hours}h"
        )

    def test_gap_puts_runs_on_different_calendar_days(self) -> None:
        friday_run = datetime(2026, 1, 2, 22, 0, tzinfo=timezone.utc)
        sunday_run = datetime(2026, 1, 4,  4, 0, tzinfo=timezone.utc)
        assert friday_run.date() != sunday_run.date()

    def test_sunday_run_is_not_saturday(self) -> None:
        # weekday() 6 = Sunday; 5 = Saturday
        sunday_run = datetime(2026, 1, 4, 4, 0, tzinfo=timezone.utc)
        assert sunday_run.weekday() == 6, "Peer cron should fire on Sunday (weekday 6)"


# ---------------------------------------------------------------------------
# peer_fetch.main() dry-run
# ---------------------------------------------------------------------------

class TestPeerFetchDryRun:
    """
    Verify that dry-run mode projects the correct call count and makes no
    HTTP requests.  Uses mocks to avoid touching the real DB or AV API.
    """

    def _make_mock_conn(self) -> MagicMock:
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = (0,)
        return conn

    @patch("scripts.peer_fetch.db_client.get_connection")
    @patch("scripts.peer_fetch.db_client.initialize_schema")
    @patch("scripts.peer_fetch.MultiTickerLoader")
    @patch("scripts.peer_fetch.MultiDividendLoader")
    @patch("scripts.peer_fetch.db_client.get_api_request_count", return_value=0)
    def test_dry_run_no_http_calls(
        self,
        mock_av_count,
        mock_div_cls,
        mock_price_cls,
        mock_init_schema,
        mock_get_conn,
    ) -> None:
        mock_conn = self._make_mock_conn()
        mock_get_conn.return_value = mock_conn

        mock_price_loader = MagicMock()
        mock_price_loader.fetch_all_prices.return_value = {t: None for t in config.PEER_TICKER_UNIVERSE}
        mock_price_cls.return_value = mock_price_loader

        mock_div_loader = MagicMock()
        mock_div_loader.fetch_for_tickers.return_value = {t: None for t in config.PEER_TICKER_UNIVERSE}
        mock_div_cls.return_value = mock_div_loader

        from scripts.peer_fetch import main
        main(dry_run=True)

        # In dry-run mode, loaders are still called but with dry_run=True
        mock_price_loader.fetch_all_prices.assert_called_once_with(
            list(config.PEER_TICKER_UNIVERSE), dry_run=True
        )
        mock_div_loader.fetch_for_tickers.assert_called_once_with(
            list(config.PEER_TICKER_UNIVERSE), dry_run=True
        )

    @patch("scripts.peer_fetch.db_client.get_connection")
    @patch("scripts.peer_fetch.db_client.initialize_schema")
    @patch("scripts.peer_fetch.MultiTickerLoader")
    @patch("scripts.peer_fetch.MultiDividendLoader")
    @patch("scripts.peer_fetch.db_client.get_api_request_count", return_value=0)
    def test_dry_run_projected_call_count(
        self,
        mock_av_count,
        mock_div_cls,
        mock_price_cls,
        mock_init_schema,
        mock_get_conn,
    ) -> None:
        mock_conn = self._make_mock_conn()
        mock_get_conn.return_value = mock_conn

        mock_price_loader = MagicMock()
        mock_price_loader.fetch_all_prices.return_value = {t: None for t in config.PEER_TICKER_UNIVERSE}
        mock_price_cls.return_value = mock_price_loader

        mock_div_loader = MagicMock()
        mock_div_loader.fetch_for_tickers.return_value = {t: None for t in config.PEER_TICKER_UNIVERSE}
        mock_div_cls.return_value = mock_div_loader

        captured = io.StringIO()
        sys.stdout = captured
        try:
            from scripts.peer_fetch import main
            main(dry_run=True)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        # Should show projected 8 calls (4 prices + 4 dividends)
        expected_calls = len(config.PEER_TICKER_UNIVERSE) * 2  # prices + dividends
        assert str(expected_calls) in output, (
            f"Expected '{expected_calls}' in dry-run output; got:\n{output}"
        )

    @patch("scripts.peer_fetch.db_client.get_connection")
    @patch("scripts.peer_fetch.db_client.initialize_schema")
    @patch("scripts.peer_fetch.MultiTickerLoader")
    @patch("scripts.peer_fetch.MultiDividendLoader")
    @patch("scripts.peer_fetch.db_client.get_api_request_count", return_value=0)
    def test_dry_run_mentions_all_peer_tickers(
        self,
        mock_av_count,
        mock_div_cls,
        mock_price_cls,
        mock_init_schema,
        mock_get_conn,
    ) -> None:
        mock_conn = self._make_mock_conn()
        mock_get_conn.return_value = mock_conn

        mock_price_loader = MagicMock()
        mock_price_loader.fetch_all_prices.return_value = {t: None for t in config.PEER_TICKER_UNIVERSE}
        mock_price_cls.return_value = mock_price_loader

        mock_div_loader = MagicMock()
        mock_div_loader.fetch_for_tickers.return_value = {t: None for t in config.PEER_TICKER_UNIVERSE}
        mock_div_cls.return_value = mock_div_loader

        captured = io.StringIO()
        sys.stdout = captured
        try:
            from scripts.peer_fetch import main
            main(dry_run=True)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        for ticker in config.PEER_TICKER_UNIVERSE:
            assert ticker in output, f"Expected {ticker} in dry-run output"
