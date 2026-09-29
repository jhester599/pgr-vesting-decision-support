"""Independent immutable checks of the exact approved input blobs."""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from datetime import date

import pandas as pd
import pytest

from src.database import db_client
from src.processing import pgr_edgar_validation as validation
from src.processing.price_integrity import (
    find_duplicate_week_bars,
    find_unexplained_price_jumps,
)


SEED = "aae0be883309bdc068ee9afbb78b0aa9a4b52b8d"
REPAIR = "ed7997f6f540f664e59dd44a4079616d74a8e8cb"
HASHES = {
    SEED: "7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d",
    REPAIR: "38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6",
}
pytestmark = pytest.mark.artifact


@pytest.fixture
def snapshot(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Extract Git bytes outside the tracked DB path; never create sidecars."""
    commit = os.environ.get("V200_PREFLIGHT_COMMIT", REPAIR)
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:data/pgr_financials.db"]
    )
    assert hashlib.sha256(payload).hexdigest() == HASHES[commit]
    path = tmp_path / "input.db"
    path.write_bytes(payload)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES[commit]


def test_required_dividends_are_fresh(snapshot: sqlite3.Connection) -> None:
    """This assertion is deliberately red on the seed, green on D7."""
    rows = db_client.check_dividend_freshness(
        snapshot, as_of=date(2026, 9, 26)
    )
    stale = [row["ticker"] for row in rows if row["status"] == "STALE"]
    print("STALE_TICKERS=" + json.dumps(stale))
    required = {
        "PGR",
        "VOO",
        "VXUS",
        "VWO",
        "VMBS",
        "BND",
        "GLD",
        "DBC",
        "VDE",
    }
    assert not required.intersection(stale), stale
    assert all(
        row["ticker"] == "GLD" for row in rows if row["status"] == "NO_HISTORY"
    )


def test_repair_accounting_calendar_and_provenance(
    snapshot: sqlite3.Connection,
) -> None:
    assert snapshot.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert find_unexplained_price_jumps(snapshot).empty
    assert find_duplicate_week_bars(snapshot).empty
    assert not snapshot.execute(
        "SELECT series_id, substr(month_end,1,7), COUNT(*) "
        "FROM fred_macro_monthly GROUP BY 1,2 HAVING COUNT(*) > 1"
    ).fetchall()
    monthly = db_client.get_pgr_edgar_monthly(snapshot)
    quarterly = db_client.get_pgr_fundamentals(snapshot)
    assert len(monthly) == 265
    assert len(quarterly) == 73
    assert not validation.missing_months(monthly)
    for check in (
        validation.income_identity_violations,
        validation.combined_ratio_violations,
        validation.equity_violations,
        validation.pif_jump_violations,
    ):
        assert check(monthly).empty
    assert validation.quarterly_net_income_violations(monthly, quarterly).empty
    for frame in (monthly, quarterly):
        filed = pd.to_datetime(frame["filing_date"])
        assert filed.notna().all()
        assert (filed >= frame.index).all()
    assert (
        monthly.iloc[-1][
            [
                "combined_ratio",
                "pif_growth_yoy",
                "npw_growth_yoy",
                "investment_income",
                "book_value_per_share",
                "investment_book_yield",
            ]
        ]
        .notna()
        .all()
    )
