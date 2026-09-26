"""Tests for structured logging and fallback behavior in bootstrap.py."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch


def test_run_monthly_decision_logs_exception_context(caplog) -> None:
    from scripts import bootstrap

    with patch("scripts.monthly_decision.main", side_effect=RuntimeError("bootstrap boom")):
        with caplog.at_level(logging.ERROR):
            rc = bootstrap._run_monthly_decision("2026-04-05", dry_run=True)

    assert rc == 1
    assert "monthly_decision raised an unexpected error." in caplog.text
    assert "RuntimeError: bootstrap boom" in caplog.text


class TestBootstrapLogging:
    def _make_mock_conn(self, n_prices: int = 10, n_divs: int = 10) -> MagicMock:
        conn = MagicMock()
        conn.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=(n_prices,))),
            MagicMock(fetchone=MagicMock(return_value=(n_divs,))),
        ]
        return conn

    @patch("scripts.bootstrap.db_client.get_connection")
    @patch("scripts.bootstrap.db_client.initialize_schema")
    @patch("scripts.bootstrap.db_client.warn_if_db_behind")
    @patch("scripts.bootstrap.build_relative_return_targets")
    def test_main_logs_missing_price_table_as_error(
        self,
        mock_build_relative_returns,
        mock_warn_if_db_behind,
        mock_init_schema,
        mock_get_conn,
        caplog,
    ) -> None:
        del mock_build_relative_returns, mock_warn_if_db_behind, mock_init_schema
        mock_get_conn.return_value = self._make_mock_conn(n_prices=0, n_divs=10)

        from scripts import bootstrap

        with caplog.at_level(logging.ERROR):
            rc = bootstrap.main(dry_run=True, skip_decision=True)

        assert rc == 1
        assert "daily_prices is empty" in caplog.text
