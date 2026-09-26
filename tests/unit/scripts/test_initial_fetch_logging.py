"""Tests for structured logging and fallback behavior in initial_fetch.py."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import config


@patch("src.ingestion.fred_loader.fetch_all_fred_macro", side_effect=RuntimeError("boom"))
@patch("src.ingestion.fred_loader.upsert_fred_to_db")
def test_fetch_fred_step_logs_exception_context(
    mock_upsert,
    mock_fetch,
    monkeypatch,
    caplog,
) -> None:
    del mock_fetch
    from scripts.initial_fetch import _fetch_fred_step

    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")

    with caplog.at_level(logging.ERROR):
        _fetch_fred_step(MagicMock(), dry_run=False)

    assert "FRED fetch failed. Continuing without FRED data." in caplog.text
    assert "RuntimeError: boom" in caplog.text
    mock_upsert.assert_not_called()


class TestInitialFetchLogging:
    def _make_mock_conn(self) -> MagicMock:
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = (0,)
        return conn

    @patch("scripts.initial_fetch.db_client.get_connection")
    @patch("scripts.initial_fetch.db_client.initialize_schema")
    @patch("scripts.initial_fetch.MultiTickerLoader")
    def test_price_loader_exception_is_logged_with_context(
        self,
        mock_price_cls,
        mock_init_schema,
        mock_get_conn,
        tmp_path,
        caplog,
    ) -> None:
        del mock_init_schema
        mock_get_conn.return_value = self._make_mock_conn()

        mock_price_loader = MagicMock()
        mock_price_loader.fetch_all_prices.side_effect = RuntimeError("price boom")
        mock_price_cls.return_value = mock_price_loader

        from scripts.initial_fetch import main

        with caplog.at_level(logging.ERROR):
            rc = main(
                do_prices=True,
                dry_run=True,
                status_file=str(tmp_path / "fetch_status.md"),
            )

        assert rc == 1
        assert "MultiTickerLoader raised a fatal error." in caplog.text
        assert "RuntimeError: price boom" in caplog.text
