"""Review 2026-09-25 step 3a (WP3): FRED pipeline, findings F06, F07 and F27.

- A mocked series fetched, stored and built gives feature(M) = raw(M - lag):
  the lag is applied once, by calendar month.
- No (series, month) pair is ever stored twice.
- Freshness is checked per FRED series behind a live feature and per ticker;
  a stale series (or ticker) returns WARNING.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import config
from src.database import db_client, migration_runner
from src.ingestion.fred_loader import fetch_all_fred_macro, upsert_fred_to_db
from src.processing import feature_engineering
from src.processing.feature_engineering import (
    _apply_fred_lags,
    build_feature_matrix,
    build_feature_matrix_from_db,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_value(series_id: str, period: pd.Period) -> float:
    """A distinct value per series and month, so any lag error is visible."""
    base = {"VIXCLS": 1000.0, "NFCI": 2000.0, "GS10": 3000.0}[series_id]
    return base + period.year * 12 + period.month


def _daily_observations(series_id: str, start: str, end: str) -> list[dict[str, str]]:
    """Business-day observations; only the month's last one carries the month value."""
    days = pd.bdate_range(start, end)
    obs = []
    for i, day in enumerate(days):
        is_last = i == len(days) - 1 or days[i + 1].month != day.month
        value = _raw_value(series_id, day.to_period("M")) if is_last else -1.0
        obs.append({"date": day.strftime("%Y-%m-%d"), "value": str(value)})
    return obs


def _weekly_observations(series_id: str, start: str, end: str) -> list[dict[str, str]]:
    """Friday observations (NFCI's cadence), month value on the last Friday."""
    days = pd.date_range(start, end, freq="W-FRI")
    obs = []
    for i, day in enumerate(days):
        is_last = i == len(days) - 1 or days[i + 1].month != day.month
        value = _raw_value(series_id, day.to_period("M")) if is_last else -1.0
        obs.append({"date": day.strftime("%Y-%m-%d"), "value": str(value)})
    return obs


def _mock_fred_session(observations: dict[str, list[dict[str, str]]]) -> MagicMock:
    def get(url, params=None, timeout=None):  # noqa: ARG001
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"observations": observations[params["series_id"]]}
        return response

    session = MagicMock()
    session.get.side_effect = get
    return session


def _memory_db() -> sqlite3.Connection:
    conn = db_client.get_connection(":memory:")
    db_client.initialize_schema(conn)
    return conn


def _weekly_prices(ticker: str, start: str, end: str) -> list[dict]:
    fridays = pd.date_range(start, end, freq="W-FRI")
    return [
        {"ticker": ticker, "date": d.strftime("%Y-%m-%d"), "close": 100.0 + i * 0.1}
        for i, d in enumerate(fridays)
    ]


# ---------------------------------------------------------------------------
# Lag applied once, by calendar month (F06)
# ---------------------------------------------------------------------------

def test_fetch_store_build_gives_feature_equal_raw_minus_configured_lag(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
    monkeypatch.setattr(feature_engineering, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
    observations = {
        "VIXCLS": _daily_observations("VIXCLS", "2018-01-01", "2020-12-31"),
        "NFCI": _weekly_observations("NFCI", "2018-01-01", "2020-12-31"),
    }
    conn = _memory_db()
    with patch(
        "src.ingestion.fred_loader.build_retry_session",
        return_value=_mock_fred_session(observations),
    ):
        fetched = fetch_all_fred_macro(["VIXCLS", "NFCI"], observation_start="2018-01-01")
    upsert_fred_to_db(conn, fetched)
    db_client.upsert_prices(conn, _weekly_prices("PGR", "2017-01-06", "2020-12-31"))

    features = build_feature_matrix_from_db(conn)

    lags = {"vix": ("VIXCLS", 1), "nfci": ("NFCI", 2)}
    assert config.FRED_SERIES_LAGS["VIXCLS"] == 1
    assert config.FRED_SERIES_LAGS["NFCI"] == 2
    checked = 0
    for row_date in features.index:
        month = row_date.to_period("M")
        for feature, (series_id, lag) in lags.items():
            source_month = month - lag
            if source_month < pd.Period("2018-01", "M"):
                continue
            assert features.loc[row_date, feature] == pytest.approx(
                _raw_value(series_id, source_month)
            ), f"{feature} at {month} should be raw {series_id} for {source_month}"
            checked += 1
    assert checked > 60


def test_fetch_all_fred_macro_returns_raw_last_observation_per_month(monkeypatch) -> None:
    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
    observations = {"VIXCLS": _daily_observations("VIXCLS", "2020-01-01", "2020-06-30")}
    with patch(
        "src.ingestion.fred_loader.build_retry_session",
        return_value=_mock_fred_session(observations),
    ):
        fetched = fetch_all_fred_macro(["VIXCLS"])

    assert list(fetched.index.strftime("%Y-%m-%d")) == [
        "2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30", "2020-05-29", "2020-06-30",
    ]
    for ts, value in fetched["VIXCLS"].items():
        assert value == pytest.approx(_raw_value("VIXCLS", ts.to_period("M")))


def test_lag_counts_calendar_months_not_rows() -> None:
    """A month stored under two labels must not cost an extra month of lag."""
    frame = pd.DataFrame(
        {"GS10": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0]},
        index=pd.to_datetime(
            ["2020-03-31", "2020-04-30", "2020-05-29", "2020-05-31", "2020-06-30", "2020-07-31"]
        ),
    )
    lagged = _apply_fred_lags(frame)["GS10"]

    assert lagged.loc["2020-06-30"] == 3.0  # June row uses May
    assert lagged.loc["2020-07-31"] == 4.0  # July row uses June
    assert lagged.loc["2020-08-31"] == 5.0  # August row uses July
    assert lagged.index.is_unique


def test_lag_never_extends_a_stale_series() -> None:
    """A series that stopped updating is NaN in rows it cannot cover."""
    frame = pd.DataFrame(
        {
            "VIXCLS": [10.0, 11.0, 12.0, 13.0, 14.0],
            "PCU5241265241261": [1.0, 2.0, np.nan, np.nan, np.nan],
        },
        index=pd.to_datetime(
            ["2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30", "2026-05-29"]
        ),
    )
    lagged = _apply_fred_lags(frame)

    assert lagged.loc["2026-03-31", "PCU5241265241261"] == 2.0
    assert np.isnan(lagged.loc["2026-04-30", "PCU5241265241261"])
    assert np.isnan(lagged.loc["2026-06-30", "PCU5241265241261"])
    assert lagged.loc["2026-06-30", "VIXCLS"] == 14.0


def test_duration_rate_shock_lags_gs10_once(monkeypatch, tmp_path: Path) -> None:
    """duration_rate_shock_3m uses the already lagged frame as given."""
    monkeypatch.setattr(feature_engineering, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
    dates = pd.bdate_range("2012-01-01", "2020-12-31", freq="BME")  # >= 60 obs kept
    steps = np.arange(len(dates), dtype=float)
    # Weekly price bars, as stored in daily_prices (the builder rejects coarser bars).
    weeks = pd.date_range(dates[0], dates[-1], freq="W-FRI")
    prices = pd.DataFrame({"close": np.linspace(50, 80, len(weeks))}, index=weeks)
    # GS10 = i**2 at row i, so a 3-month change identifies the row it came from.
    gs10 = pd.DataFrame({"GS10": steps**2}, index=dates)
    pgr_monthly = pd.DataFrame({"fixed_income_duration": 2.0}, index=dates)
    empty_divs = pd.DataFrame(columns=["dividend"], index=pd.DatetimeIndex([]))
    empty_splits = pd.DataFrame(
        columns=["split_ratio", "numerator", "denominator"], index=pd.DatetimeIndex([])
    )
    df = build_feature_matrix(
        prices, empty_divs, empty_splits, pgr_monthly=pgr_monthly, fred_macro=gs10
    )

    expected = pd.Series(2.0 * (steps**2 - (steps - 3) ** 2), index=dates)
    got = df["duration_rate_shock_3m"].dropna()
    assert len(got) >= 60
    pd.testing.assert_series_equal(got, expected.loc[got.index], check_names=False)


# ---------------------------------------------------------------------------
# One row per (series, month) (F06/F27)
# ---------------------------------------------------------------------------

def test_upsert_keeps_one_row_per_series_month() -> None:
    conn = _memory_db()
    db_client.upsert_fred_macro(
        conn,
        [
            {"series_id": "GS10", "month_end": "2020-05-29", "value": 0.66},
            {"series_id": "GS10", "month_end": "2020-05-31", "value": 0.67},
            {"series_id": "GS10", "month_end": "2020-06-30", "value": 0.73},
        ],
    )
    rows = [
        tuple(r) for r in conn.execute(
            "SELECT month_end, value FROM fred_macro_monthly WHERE series_id = 'GS10' "
            "ORDER BY month_end"
        ).fetchall()
    ]
    assert rows == [("2020-05-29", 0.67), ("2020-06-30", 0.73)]
    dup = conn.execute(
        "SELECT COUNT(*) FROM (SELECT series_id, substr(month_end, 1, 7) m, COUNT(*) n "
        "FROM fred_macro_monthly GROUP BY 1, 2 HAVING n > 1)"
    ).fetchone()[0]
    assert dup == 0


def test_unique_index_rejects_a_second_label_for_a_month() -> None:
    conn = _memory_db()
    conn.execute(
        "INSERT INTO fred_macro_monthly (series_id, month_end, value) "
        "VALUES ('NFCI', '2020-05-29', 0.267)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO fred_macro_monthly (series_id, month_end, value) "
            "VALUES ('NFCI', '2020-05-31', 0.27064)"
        )


def test_migration_005_collapses_legacy_duplicate_months() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE fred_macro_monthly (series_id TEXT NOT NULL, month_end TEXT NOT NULL, "
        "value REAL, PRIMARY KEY (series_id, month_end))"
    )
    conn.executemany(
        "INSERT INTO fred_macro_monthly VALUES (?, ?, ?)",
        [
            ("NFCI", "2020-05-29", 0.267),       # production (business) label
            ("NFCI", "2020-05-31", 0.27064),     # legacy v19 calendar label
            ("CUSR0000SETE", "2017-12-31", 5.0),  # calendar label only (Sunday)
            ("GS10", "2020-06-30", 0.73),
        ],
    )
    path = migration_runner.migrations_dir() / "005_fred_one_row_per_month.py"
    migration_runner._run_python_migration(conn, path)

    rows = [
        tuple(r) for r in conn.execute(
            "SELECT series_id, month_end, value FROM fred_macro_monthly ORDER BY 1, 2"
        ).fetchall()
    ]
    assert rows == [
        ("CUSR0000SETE", "2017-12-29", 5.0),
        ("GS10", "2020-06-30", 0.73),
        ("NFCI", "2020-05-29", 0.267),
    ]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO fred_macro_monthly VALUES ('GS10', '2020-06-29', 1.0)")


def test_v19_fredgraph_rows_use_business_month_end_labels() -> None:
    from src.research.v19 import fetch_fredgraph_series

    csv_text = "observation_date,DTWEXBGS\n2020-05-01,1.0\n2020-05-29,2.0\n2020-06-01,3.0\n"
    response = MagicMock()
    response.text = csv_text
    response.raise_for_status = MagicMock()
    with patch("src.research.v19.requests.get", return_value=response):
        df = fetch_fredgraph_series("DTWEXBGS", observation_start="2020-01-01")

    assert list(df.index.strftime("%Y-%m-%d")) == ["2020-05-29", "2020-06-30"]
    assert list(df["DTWEXBGS"]) == [2.0, 3.0]


# ---------------------------------------------------------------------------
# Freshness per series and per ticker (F07)
# ---------------------------------------------------------------------------

_REFERENCE = date(2026, 9, 25)  # decision row 2026-08-31
# FRED series behind the live Ridge/GBT features (config.MODEL_FEATURE_OVERRIDES).
_LIVE_FRED_SERIES = {
    "T10Y2Y", "GS5", "GS2", "GS10", "T10YIE", "BAMLH0A0HYM2", "NFCI", "VIXCLS",
    "PCU5241265241261", "CUSR0000SETA02", "CUSR0000SAM2",
}


def _fresh_db() -> sqlite3.Connection:
    """Every live FRED series, PGR and the 8 benchmarks fresh on _REFERENCE."""
    conn = _memory_db()
    for ticker in ["PGR", *config.PRIMARY_FORECAST_UNIVERSE]:
        db_client.upsert_prices(
            conn, [{"ticker": ticker, "date": "2026-09-18", "close": 100.0}]
        )
    records = []
    for series_id in sorted(_LIVE_FRED_SERIES):
        lag = config.FRED_SERIES_LAGS.get(series_id, config.FRED_DEFAULT_LAG_MONTHS)
        needed = pd.Period("2026-08", "M") - lag
        records.append(
            {"series_id": series_id, "month_end": str(needed.end_time.date()), "value": 1.0}
        )
    db_client.upsert_fred_macro(conn, records)
    db_client.upsert_pgr_edgar_monthly(conn, [{"month_end": "2026-08-31", "combined_ratio": 90.0}])
    return conn


def test_live_feature_fred_series_covers_every_live_fred_feature() -> None:
    series = db_client.live_feature_fred_series()
    assert set(series) == _LIVE_FRED_SERIES
    assert "rate_adequacy_gap_yoy" in series["PCU5241265241261"]


def test_all_fresh_returns_ok() -> None:
    report = db_client.check_data_freshness(_fresh_db(), _REFERENCE)
    assert report["overall_status"] == "OK", report["warnings"]
    feeds = {row["feed"] for row in report["checks"]}
    assert {"Prices PGR", "Prices VOO", "FRED VIXCLS", "FRED PCU5241265241261"} <= feeds


def test_one_stale_fred_series_returns_warning() -> None:
    conn = _fresh_db()
    conn.execute("DELETE FROM fred_macro_monthly WHERE series_id = 'PCU5241265241261'")
    db_client.upsert_fred_macro(
        conn, [{"series_id": "PCU5241265241261", "month_end": "2026-02-27", "value": 1.0}]
    )
    # A fresher, future-labelled month for another series must not hide it.
    db_client.upsert_fred_macro(
        conn, [{"series_id": "VIXCLS", "month_end": "2026-09-30", "value": 15.0}]
    )

    report = db_client.check_data_freshness(conn, _REFERENCE)

    assert report["overall_status"] == "WARNING"
    stale = {row["feed"]: row for row in report["checks"] if row["status"] != "OK"}
    assert list(stale) == ["FRED PCU5241265241261"]
    assert stale["FRED PCU5241265241261"]["months_behind"] == 5
    assert any(
        "PCU5241265241261" in w and "rate_adequacy_gap_yoy" in w for w in report["warnings"]
    )


def test_one_stale_ticker_returns_warning() -> None:
    conn = _fresh_db()
    conn.execute("DELETE FROM daily_prices WHERE ticker = 'VOO'")
    db_client.upsert_prices(conn, [{"ticker": "VOO", "date": "2026-06-26", "close": 1.0}])

    report = db_client.check_data_freshness(conn, _REFERENCE)

    assert report["overall_status"] == "WARNING"
    stale = [row["feed"] for row in report["checks"] if row["status"] != "OK"]
    assert stale == ["Prices VOO"]


def test_future_month_label_is_not_zero_days_old() -> None:
    """Only a future-labelled month for a series: judged by month, not by age."""
    conn = _memory_db()
    db_client.upsert_fred_macro(
        conn, [{"series_id": "NFCI", "month_end": "2026-10-30", "value": -0.5}]
    )
    report = db_client.check_data_freshness(
        conn, _REFERENCE, price_tickers=["PGR"], fred_series=["NFCI"]
    )
    nfci = next(row for row in report["checks"] if row["feed"] == "FRED NFCI")
    assert nfci["status"] == "MISSING"  # months after the reference month are ignored
    assert nfci["age_days"] is None


# ---------------------------------------------------------------------------
# FRED_FEATURE_SOURCES matches the feature builder
# ---------------------------------------------------------------------------

def test_fred_feature_sources_match_the_builder(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(feature_engineering, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
    dates = pd.bdate_range("2012-01-01", "2020-12-31", freq="BME")
    rng = np.random.default_rng(0)
    weeks = pd.date_range(dates[0], dates[-1], freq="W-FRI")
    prices = pd.DataFrame(
        {"close": 50 + np.cumsum(rng.normal(0, 0.5, len(weeks)))}, index=weeks
    )
    all_series = sorted({s for srcs in config.FRED_FEATURE_SOURCES.values() for s in srcs})
    fred = pd.DataFrame(
        {sid: 50 + np.cumsum(rng.normal(0, 1, len(dates))) for sid in all_series}, index=dates
    )
    pgr_monthly = pd.DataFrame({"fixed_income_duration": 2.0}, index=dates)
    fundamentals = pd.DataFrame({"pe_ratio": 15.0, "pb_ratio": 3.0}, index=dates)
    empty_divs = pd.DataFrame(columns=["dividend"], index=pd.DatetimeIndex([]))
    empty_splits = pd.DataFrame(
        columns=["split_ratio", "numerator", "denominator"], index=pd.DatetimeIndex([])
    )

    def build(frame: pd.DataFrame | None) -> pd.DataFrame:
        return build_feature_matrix(
            prices, empty_divs, empty_splits, fundamentals=fundamentals,
            pgr_monthly=pgr_monthly, fred_macro=frame,
        )

    full = build(fred)
    without_fred = build(None)
    fred_features = {
        c for c in full.columns
        if c not in without_fred.columns or not full[c].equals(without_fred[c])
    }
    assert fred_features == set(config.FRED_FEATURE_SOURCES) & set(full.columns)
    for series_id in all_series:
        reduced = build(fred.drop(columns=[series_id]))
        changed = {
            c for c in fred_features
            if c not in reduced.columns or not full[c].equals(reduced[c])
        }
        expected = {
            f for f, srcs in config.FRED_FEATURE_SOURCES.items()
            if series_id in srcs and f in full.columns
        }
        assert changed == expected, series_id


# ---------------------------------------------------------------------------
# scripts/rebuild_fred_macro.py
# ---------------------------------------------------------------------------

def test_rebuild_replaces_lagged_duplicated_rows_with_raw_months(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import rebuild_fred_macro

    db_path = tmp_path / "copy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE fred_macro_monthly (series_id TEXT NOT NULL, month_end TEXT NOT NULL, "
        "value REAL, PRIMARY KEY (series_id, month_end))"
    )
    # Legacy state: lagged values (April holds March) and a calendar-label copy.
    conn.executemany(
        "INSERT INTO fred_macro_monthly VALUES (?, ?, ?)",
        [
            ("VIXCLS", "2020-04-30", 53.54),
            ("VIXCLS", "2020-05-29", 34.15),
            ("VIXCLS", "2020-05-31", 34.15),
            ("SP500_PE_RATIO_MULTPL", "2020-05-31", 22.0),
        ],
    )
    conn.commit()
    conn.close()

    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame(
        {"VIXCLS": [53.54, 34.15, 27.51, 30.43]},
        index=pd.DatetimeIndex(
            ["2020-03-31", "2020-04-30", "2020-05-28", "2020-05-29"], name="date"
        ),
    ).to_csv(cache / "VIXCLS.csv")
    monkeypatch.setattr(rebuild_fred_macro, "rebuild_plan", lambda: {"VIXCLS": "1990-01-01"})

    diff = rebuild_fred_macro.main(str(db_path), cache_dir=str(cache), offline=True)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT series_id, month_end, value FROM fred_macro_monthly ORDER BY 1, 2"
    ).fetchall()
    conn.close()
    assert rows == [
        ("SP500_PE_RATIO_MULTPL", "2020-05-29", 22.0),  # relabelled, not refetched
        ("VIXCLS", "2020-03-31", 53.54),
        ("VIXCLS", "2020-04-30", 34.15),
        ("VIXCLS", "2020-05-29", 30.43),
    ]
    vix = diff.set_index("series_id").loc["VIXCLS"]
    assert vix["duplicate_months_before"] == 1
    assert vix["rows_after"] == 3


def test_rebuild_unlags_stored_months_before_a_truncated_fetch() -> None:
    """FRED serves 3 years of ICE BofA data; older stored months are un-lagged."""
    from scripts.rebuild_fred_macro import recover_truncated_history

    months = pd.period_range("2019-01", "2026-08", freq="M")
    raw_full = pd.Series(np.arange(len(months), dtype=float) + 100.0, index=months)
    stored = raw_full.copy()
    stored.index = stored.index + 1  # legacy loader stored raw(M - 1) at M
    fetched = raw_full[raw_full.index >= pd.Period("2023-09", "M")]

    combined, note = recover_truncated_history("BAMLH0A0HYM2", stored, fetched)

    pd.testing.assert_series_equal(combined, raw_full)
    assert "un-lagged by 1" in note

    with pytest.raises(RuntimeError):
        recover_truncated_history("BAMLH0A0HYM2", stored * 2.0, fetched)
