"""Tests for structured logging in migrate_v1_to_v2.py."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch


def test_migrate_edgar_monthly_logs_loader_exception(caplog) -> None:
    from scripts import migrate_v1_to_v2

    with patch(
        "pgr_vds.ingestion.edgar_monthly.load.load_from_csv",
        side_effect=RuntimeError("csv loader boom"),
    ):
        with caplog.at_level(logging.ERROR):
            n = migrate_v1_to_v2._migrate_edgar_monthly(MagicMock())

    assert n == 0
    assert "Could not load the EDGAR cache CSV; skipping." in caplog.text
    assert "RuntimeError: csv loader boom" in caplog.text
