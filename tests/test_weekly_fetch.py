"""Tests for structured logging and dry-run behavior in weekly_fetch.py."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import config


class TestWeeklyFetch:
    def _make_mock_conn(self) -> MagicMock:
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = (0,)
        return conn

    @patch("scripts.weekly_fetch.db_client.get_connection")
    @patch("scripts.weekly_fetch.db_client.initialize_schema")
    @patch("scripts.weekly_fetch.MultiTickerLoader")
    @patch("scripts.weekly_fetch.MultiDividendLoader")
    def test_dry_run_logs_projected_budget(
        self,
        mock_div_cls,
        mock_price_cls,
        mock_init_schema,
        mock_get_conn,
        caplog,
    ) -> None:
        mock_conn = self._make_mock_conn()
        mock_get_conn.return_value = mock_conn

        mock_price_loader = MagicMock()
        mock_price_loader.fetch_all_prices.return_value = {
            ticker: 0 for ticker in ["PGR", *config.ETF_BENCHMARK_UNIVERSE]
        }
        mock_price_cls.return_value = mock_price_loader

        mock_div_loader = MagicMock()
        mock_div_loader.fetch_for_tickers.return_value = {"PGR": 0}
        mock_div_cls.return_value = mock_div_loader

        from scripts.weekly_fetch import main

        with caplog.at_level(logging.INFO):
            main(dry_run=True, skip_fred=True)

        expected_calls = len(config.ETF_BENCHMARK_UNIVERSE) + 2
        assert f"Projected API calls: AV {expected_calls}/{config.AV_DAILY_LIMIT}" in caplog.text
        mock_price_loader.fetch_all_prices.assert_called_once_with(
            ["PGR", *config.ETF_BENCHMARK_UNIVERSE],
            dry_run=True,
        )
        mock_div_loader.fetch_for_tickers.assert_called_once_with(
            ["PGR"],
            dry_run=True,
        )

    @patch("src.ingestion.fred_loader.fetch_all_fred_macro", side_effect=RuntimeError("boom"))
    @patch("src.ingestion.fred_loader.upsert_fred_to_db")
    def test_fetch_fred_step_logs_exception_context(
        self,
        mock_upsert,
        mock_fetch,
        monkeypatch,
        caplog,
    ) -> None:
        from scripts.weekly_fetch import _fetch_fred_step

        monkeypatch.setattr(config, "FRED_API_KEY", "test-key")

        with caplog.at_level(logging.ERROR):
            _fetch_fred_step(MagicMock(), dry_run=False)

        assert "FRED fetch failed. Continuing with cached data." in caplog.text
        assert "RuntimeError: boom" in caplog.text
        mock_upsert.assert_not_called()


class TestRebuildTargetsJumpGuard:
    """Targets are not rebuilt over a split missing from config/splits.py (F03/F05)."""

    @staticmethod
    def _conn(tmp_path, vgt_closes):
        import pandas as pd

        from src.database import db_client

        conn = db_client.get_connection(str(tmp_path / "t.db"))
        db_client.initialize_schema(conn)
        dates = pd.date_range("2026-03-06", periods=len(vgt_closes), freq="W-FRI")
        db_client.upsert_prices(conn, [
            {"ticker": "VGT", "date": d.strftime("%Y-%m-%d"), "close": c}
            for d, c in zip(dates, vgt_closes)
        ])
        return conn

    def test_unexplained_jump_skips_rebuild(self, tmp_path, monkeypatch, caplog) -> None:
        import scripts.weekly_fetch as wf

        monkeypatch.setattr(config, "KNOWN_SPLITS", [])
        conn = self._conn(tmp_path, [800.0, 805.0, 104.0, 105.0])
        rebuilt = MagicMock()
        monkeypatch.setattr(wf, "build_relative_return_targets", rebuilt)
        with caplog.at_level(logging.ERROR):
            assert wf._rebuild_targets(conn, dry_run=False) is False
        rebuilt.assert_not_called()
        assert "Unexplained weekly price jumps" in caplog.text
        conn.close()

    def test_registered_split_allows_rebuild(self, tmp_path, monkeypatch) -> None:
        import pandas as pd

        import scripts.weekly_fetch as wf

        monkeypatch.setattr(config, "KNOWN_SPLITS", [{
            "ticker": "VGT", "split_date": "2026-03-17", "split_ratio": 8.0,
            "numerator": 8.0, "denominator": 1.0,
        }])
        conn = self._conn(tmp_path, [800.0, 805.0, 104.0, 105.0])
        rebuilt = MagicMock(return_value=pd.DataFrame())
        monkeypatch.setattr(wf, "build_relative_return_targets", rebuilt)
        assert wf._rebuild_targets(conn, dry_run=False) is True
        assert rebuilt.call_count == 2
        conn.close()
