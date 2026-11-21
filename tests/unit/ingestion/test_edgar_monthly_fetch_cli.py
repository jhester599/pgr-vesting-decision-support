"""``cli/edgar_monthly_fetch.py --export-csv`` keeps the committed CSV in step with the DB."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from pgr_vds.ingestion.edgar_monthly import load
from src.database import db_client

CLI_PATH = Path(__file__).resolve().parents[3] / "cli" / "edgar_monthly_fetch.py"


def _cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("edgar_monthly_fetch_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_fetch(conn: sqlite3.Connection, **_: object) -> int:
    return db_client.upsert_pgr_edgar_monthly(
        conn,
        [
            {"month_end": "2026-09-30", "filing_date": "2026-10-14", "combined_ratio": 90.0},
            {"month_end": "2026-10-31", "filing_date": "2026-11-18", "combined_ratio": 91.2},
        ],
    )


def test_export_csv_writes_the_table_after_the_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(load, "fetch_and_upsert", _fake_fetch)
    out = tmp_path / "pgr_edgar_cache.csv"

    _cli().main(["--export-csv", str(out)])

    written = pd.read_csv(out)
    assert tuple(written.columns) == load.EDGAR_CACHE_CSV_COLUMNS
    assert written["report_period"].tolist() == ["2026-09", "2026-10"]
    assert written["combined_ratio"].tolist() == [90.0, 91.2]


def test_no_csv_is_written_without_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(load, "fetch_and_upsert", _fake_fetch)
    exported: list[str] = []
    monkeypatch.setattr(
        load, "export_edgar_cache_csv", lambda _conn, path: exported.append(path) or 0
    )

    _cli().main([])

    assert exported == []


def test_export_csv_is_refused_on_a_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "pgr_edgar_cache.csv"

    with pytest.raises(SystemExit) as stop:
        _cli().main(["--dry-run", "--export-csv", str(out)])

    assert stop.value.code == 2
    assert not out.exists()
