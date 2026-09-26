"""DB-integrity guards for the committed price, split and target tables.

Review 2026-09-25 findings F03, F05 and F22:

* every weekly close ratio outside [0.6, 1.7] must be explained by a
  ``split_history`` row within +/-7 days (or by a reviewed entry in
  ``config.KNOWN_PRICE_JUMPS``);
* ``daily_prices`` holds at most one bar per ticker per ISO week;
* ``monthly_relative_returns`` equals a fresh rebuild from the stored prices,
  dividends and splits (so a newly added split row cannot leave stale targets).

The committed DB is opened read-only; these tests never write to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.database import db_client
from src.processing.price_integrity import (
    find_duplicate_week_bars,
    find_unexplained_price_jumps,
)

_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "pgr_financials.db"

pytestmark = pytest.mark.skipif(
    not _DB_PATH.exists(), reason="committed DB not present"
)


@pytest.fixture(scope="module")
def ro_conn():
    conn = db_client.get_connection(str(_DB_PATH), read_only=True)
    yield conn
    conn.close()


@pytest.mark.artifact
def test_no_unexplained_weekly_price_jumps(ro_conn) -> None:
    jumps = find_unexplained_price_jumps(ro_conn)
    assert jumps.empty, (
        "Weekly close ratios outside [0.6, 1.7] with no split_history row "
        "within 7 days:\n" + jumps.to_string()
    )


@pytest.mark.artifact
def test_split_history_contains_every_canonical_split(ro_conn) -> None:
    stored = {
        (r[0], r[1], round(float(r[2]), 6))
        for r in ro_conn.execute(
            "SELECT ticker, split_date, split_ratio FROM split_history"
        )
    }
    for split in config.KNOWN_SPLITS:
        key = (split["ticker"], split["split_date"], round(split["split_ratio"], 6))
        assert key in stored, f"split_history is missing {key}"


@pytest.mark.artifact
def test_one_price_bar_per_ticker_per_iso_week(ro_conn) -> None:
    dupes = find_duplicate_week_bars(ro_conn)
    assert dupes.empty, "Duplicate ticker-ISO-week bars:\n" + dupes.to_string()


@pytest.mark.slow
@pytest.mark.artifact
def test_stored_targets_match_fresh_rebuild(ro_conn) -> None:
    from src.processing.multi_total_return import build_relative_return_targets

    for horizon in (6, 12):
        fresh = build_relative_return_targets(ro_conn, horizon, upsert=False)
        stored = pd.read_sql_query(
            "SELECT date, benchmark, relative_return FROM monthly_relative_returns "
            "WHERE target_horizon = ?",
            ro_conn,
            params=[horizon],
            parse_dates=["date"],
        ).pivot(index="date", columns="benchmark", values="relative_return")
        stored = stored[[c for c in fresh.columns if c in stored.columns]]
        assert set(stored.columns) == set(fresh.columns)
        fresh_long = fresh.stack().dropna().rename("fresh")
        stored_long = stored.stack().dropna().rename("stored")
        joined = pd.concat([fresh_long, stored_long], axis=1)
        missing = joined[joined["stored"].isna()]
        extra = joined[joined["fresh"].isna()]
        assert missing.empty, f"{horizon}M targets missing from DB:\n{missing.head()}"
        assert extra.empty, f"{horizon}M stale targets in DB:\n{extra.head()}"
        diff = (joined["fresh"] - joined["stored"]).abs()
        assert np.nanmax(diff.to_numpy()) < 1e-9


class TestGuardFunctionsOnSyntheticData:
    """The guard helpers themselves must flag the defects they target."""

    @pytest.fixture
    def conn(self, tmp_path):
        c = db_client.get_connection(str(tmp_path / "t.db"))
        db_client.initialize_schema(c)
        yield c
        c.close()

    @staticmethod
    def _bars(ticker: str, closes: list[float], start: str = "2020-01-03") -> list[dict]:
        dates = pd.date_range(start=start, periods=len(closes), freq="W-FRI")
        return [
            {"ticker": ticker, "date": d.strftime("%Y-%m-%d"), "close": c}
            for d, c in zip(dates, closes)
        ]

    def test_split_like_drop_without_split_row_is_flagged(self, conn) -> None:
        db_client.upsert_prices(conn, self._bars("AAA", [100, 101, 25.5, 25.7]))
        jumps = find_unexplained_price_jumps(conn, allowlist=[])
        assert jumps["ticker"].tolist() == ["AAA"]
        assert jumps["date"].tolist() == ["2020-01-17"]

    def test_split_row_within_seven_days_explains_jump(self, conn) -> None:
        db_client.upsert_prices(conn, self._bars("AAA", [100, 101, 25.5, 25.7]))
        db_client.upsert_splits(conn, [{
            "ticker": "AAA", "split_date": "2020-01-13", "split_ratio": 4.0,
        }])
        assert find_unexplained_price_jumps(conn, allowlist=[]).empty

    def test_split_row_too_far_away_does_not_explain_jump(self, conn) -> None:
        db_client.upsert_prices(conn, self._bars("AAA", [100, 101, 25.5, 25.7]))
        db_client.upsert_splits(conn, [{
            "ticker": "AAA", "split_date": "2020-01-01", "split_ratio": 4.0,
        }])
        assert len(find_unexplained_price_jumps(conn, allowlist=[])) == 1

    def test_reviewed_allowlist_entry_explains_jump(self, conn) -> None:
        db_client.upsert_prices(conn, self._bars("AAA", [100, 45, 46]))
        allow = [{"ticker": "AAA", "date": "2020-01-10", "reason": "test"}]
        assert find_unexplained_price_jumps(conn, allowlist=allow).empty

    def test_duplicate_week_bars_are_flagged(self, conn) -> None:
        db_client.upsert_prices(conn, [
            {"ticker": "AAA", "date": "2026-03-24", "close": 10.0},
            {"ticker": "AAA", "date": "2026-03-27", "close": 10.5},
            {"ticker": "AAA", "date": "2026-04-03", "close": 10.6},
        ])
        dupes = find_duplicate_week_bars(conn)
        assert dupes["date"].tolist() == ["2026-03-24", "2026-03-27"]


def test_integrity_script_flags_missing_split(tmp_path, monkeypatch, capsys) -> None:
    from scripts.check_data_integrity import main

    path = str(tmp_path / "t.db")
    conn = db_client.get_connection(path)
    db_client.initialize_schema(conn)
    dates = pd.date_range("2026-03-06", periods=4, freq="W-FRI")
    db_client.upsert_prices(conn, [
        {"ticker": "VGT", "date": d.strftime("%Y-%m-%d"), "close": c}
        for d, c in zip(dates, [800.0, 805.0, 104.0, 105.0])
    ])
    conn.close()
    db_client.finalize_for_commit(path)
    assert main(["--db", path]) == 1
    assert "VGT" in capsys.readouterr().out
    conn = db_client.get_connection(path)
    db_client.upsert_splits(conn, [{"ticker": "VGT", "split_date": "2026-03-17",
                                    "split_ratio": 8.0}])
    conn.close()
    db_client.finalize_for_commit(path)
    assert main(["--db", path]) == 0
