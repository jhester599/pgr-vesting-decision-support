"""Focused logging tests for src/ingestion/fred_loader.py."""

from __future__ import annotations

import logging

import config
import pandas as pd
from src.ingestion import fred_loader


def test_failed_series_logs_exception_context_and_continues(monkeypatch, caplog) -> None:
    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")

    def _mock_fetch(
        series_id: str,
        observation_start: str = "2008-01-01",
        dry_run: bool = False,
    ) -> pd.DataFrame:
        del observation_start, dry_run
        if series_id == "BAD_SERIES":
            raise RuntimeError("fred boom")
        return pd.DataFrame(
            {series_id: [1.0]},
            index=pd.to_datetime(["2024-01-31"]),
        )

    monkeypatch.setattr(fred_loader, "fetch_fred_series", _mock_fetch)

    with caplog.at_level(logging.ERROR):
        df = fred_loader.fetch_all_fred_macro(["GOOD_SERIES", "BAD_SERIES"])

    assert "GOOD_SERIES" in df.columns
    assert "Failed to fetch FRED series BAD_SERIES" in caplog.text
    assert "RuntimeError: fred boom" in caplog.text
