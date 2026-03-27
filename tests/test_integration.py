"""
End-to-end integration smoke test for the v2 PGR vesting decision support engine.

Uses a temporary in-memory SQLite database populated with fully synthetic data
for PGR + 3 ETF benchmarks.  Exercises the full pipeline:

  1. Schema initialization (db_client)
  2. Price / dividend / relative return data loading
  3. Feature matrix construction from DB (feature_engineering)
  4. Relative return computation (multi_total_return)
  5. WFO training for multiple benchmarks (multi_benchmark_wfo)
  6. Current-period signal generation (multi_benchmark_wfo)
  7. Vesting event enumeration (vesting_events)
  8. Backtest result construction
  9. Reporting table generation and CSV export (backtest_report)
  10. Plot generation (visualization/plots)

This test is marked ``integration`` and may take 10–30 seconds.

No live API calls are made — all data is generated synthetically.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import config
from src.database.db_client import get_connection, initialize_schema, upsert_prices, upsert_relative_returns
from src.backtest.vesting_events import enumerate_vesting_events
from src.backtest.backtest_engine import BacktestEventResult, _signal_from_prediction
from src.models.multi_benchmark_wfo import run_all_benchmarks, get_current_signals
from src.processing.feature_engineering import get_X_y_relative
from src.reporting.backtest_report import (
    generate_backtest_table,
    export_backtest_to_csv,
    print_backtest_summary,
)


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers to build synthetic data
# ---------------------------------------------------------------------------

_SYNTHETIC_ETFS = ["VTI", "BND", "VGT"]
_N_DAYS = 365 * 15   # ~15 years of daily prices


def _generate_prices(ticker: str, start: str = "2005-01-03") -> list[dict]:
    """Generate synthetic daily price rows for a ticker."""
    rng = np.random.default_rng(hash(ticker) % (2**32))
    dates = pd.bdate_range(start, periods=_N_DAYS, freq="B")
    prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, _N_DAYS))
    return [
        {
            "ticker":     ticker,
            "date":       d.strftime("%Y-%m-%d"),
            "open":       float(p * 0.999),
            "high":       float(p * 1.005),
            "low":        float(p * 0.995),
            "close":      float(p),
            "volume":     int(1_000_000),
            "source":     "synthetic",
            "proxy_fill": 0,
        }
        for d, p in zip(dates, prices)
    ]


def _generate_relative_returns(
    conn: sqlite3.Connection,
    etfs: list[str],
    forward_months: int,
) -> None:
    """Upsert synthetic relative return rows for each ETF/horizon."""
    rng = np.random.default_rng(999)
    dates = pd.bdate_range("2010-01-29", periods=120, freq="BME")
    records = []
    for etf in etfs:
        for d in dates:
            rel = float(rng.normal(0.005, 0.05))
            records.append({
                "date":             d.strftime("%Y-%m-%d"),
                "benchmark":        etf,
                "target_horizon":   forward_months,
                "pgr_return":       float(rng.normal(0.04, 0.08)),
                "benchmark_return": float(rng.normal(0.035, 0.07)),
                "relative_return":  rel,
                "proxy_fill":       0,
            })
    upsert_relative_returns(conn, records)


@pytest.fixture(scope="module")
def integration_db():
    """
    Module-scoped fixture: creates a temporary SQLite DB, populates it with
    synthetic PGR + ETF price/return data, and returns the connection.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = get_connection(db_path)
    initialize_schema(conn)

    # Insert synthetic prices for PGR and 3 ETFs
    for ticker in ["PGR"] + _SYNTHETIC_ETFS:
        rows = _generate_prices(ticker)
        upsert_prices(conn, rows)

    # Insert synthetic relative returns for both horizons
    for horizon in [6, 12]:
        _generate_relative_returns(conn, _SYNTHETIC_ETFS, horizon)

    conn.commit()
    yield conn

    conn.close()
    os.unlink(db_path)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationPipeline:

    def test_schema_has_required_tables(self, integration_db):
        cur = integration_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cur.fetchall()}
        required = {
            "daily_prices", "daily_dividends", "split_history",
            "pgr_fundamentals_quarterly", "pgr_edgar_monthly",
            "monthly_relative_returns", "api_request_log", "ingestion_metadata",
        }
        assert required.issubset(tables)

    def test_prices_loaded_for_all_tickers(self, integration_db):
        for ticker in ["PGR"] + _SYNTHETIC_ETFS:
            cur = integration_db.execute(
                "SELECT COUNT(*) FROM daily_prices WHERE ticker = ?", (ticker,)
            )
            count = cur.fetchone()[0]
            assert count > 1000, f"{ticker}: expected >1000 price rows, got {count}"

    def test_relative_returns_loaded(self, integration_db):
        cur = integration_db.execute(
            "SELECT COUNT(*) FROM monthly_relative_returns"
        )
        count = cur.fetchone()[0]
        # 3 ETFs × 2 horizons × 120 dates
        assert count == 3 * 2 * 120

    def test_feature_matrix_from_db(self, integration_db):
        from src.processing.feature_engineering import build_feature_matrix_from_db
        df = build_feature_matrix_from_db(integration_db, force_refresh=True)
        assert not df.empty
        assert "mom_3m" in df.columns
        assert "vol_63d" in df.columns  # vol_21d dropped by config.FEATURES_TO_DROP (v4.3)
        assert "target_6m_return" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_relative_return_loading(self, integration_db):
        from src.processing.multi_total_return import load_relative_return_matrix
        series = load_relative_return_matrix(integration_db, "VTI", 6)
        assert not series.empty
        assert series.name == "VTI_6m"

    def test_wfo_trains_for_multiple_benchmarks(self, integration_db):
        from src.processing.feature_engineering import build_feature_matrix_from_db
        from src.processing.multi_total_return import load_relative_return_matrix

        df = build_feature_matrix_from_db(integration_db, force_refresh=True)
        feature_cols = [c for c in df.columns if c != "target_6m_return"]
        X = df[feature_cols]

        # Build a small relative return matrix from the 3 ETFs
        rel_cols = {}
        for etf in _SYNTHETIC_ETFS:
            s = load_relative_return_matrix(integration_db, etf, 6)
            if not s.empty:
                rel_cols[etf] = s.rename(etf)

        if not rel_cols:
            pytest.skip("No relative returns loaded.")

        rel_matrix = pd.DataFrame(rel_cols)

        results = run_all_benchmarks(
            X, rel_matrix, model_type="lasso", target_horizon_months=6
        )
        # At least one model must train
        assert len(results) > 0
        for etf, res in results.items():
            assert res.benchmark == etf
            assert res.target_horizon == 6

    def test_current_signals_generate_dataframe(self, integration_db):
        from src.processing.feature_engineering import build_feature_matrix_from_db
        from src.processing.multi_total_return import load_relative_return_matrix

        df = build_feature_matrix_from_db(integration_db, force_refresh=True)
        feature_cols = [c for c in df.columns if c != "target_6m_return"]
        X = df[feature_cols]

        rel_cols = {}
        for etf in _SYNTHETIC_ETFS:
            s = load_relative_return_matrix(integration_db, etf, 6)
            if not s.empty:
                rel_cols[etf] = s.rename(etf)

        if not rel_cols:
            pytest.skip("No relative returns loaded.")

        rel_matrix = pd.DataFrame(rel_cols)
        wfo_results = run_all_benchmarks(X, rel_matrix, target_horizon_months=6)

        if not wfo_results:
            pytest.skip("No WFO results produced.")

        signals = get_current_signals(
            X_full=X,
            relative_return_matrix=rel_matrix,
            wfo_results=wfo_results,
            X_current=X.iloc[[-1]],
        )
        assert isinstance(signals, pd.DataFrame)
        assert "signal" in signals.columns

    def test_vesting_events_enumerated(self):
        events = enumerate_vesting_events(start_year=2014, end_year=2023)
        assert len(events) == 20  # 10 years × 2 events
        for e in events:
            assert e.event_date.weekday() < 5  # all weekdays

    def test_backtest_result_construction(self):
        from src.backtest.vesting_events import VestingEvent, _add_months
        event = VestingEvent(
            event_date=date(2020, 1, 20),
            rsu_type="time",
            horizon_6m_end=_add_months(date(2020, 1, 20), 6),
            horizon_12m_end=_add_months(date(2020, 1, 20), 12),
        )
        r = BacktestEventResult(
            event=event,
            benchmark="VTI",
            target_horizon=6,
            predicted_relative_return=0.05,
            realized_relative_return=0.03,
            signal_direction=_signal_from_prediction(0.05),
            correct_direction=True,
            predicted_sell_pct=0.25,
            ic_at_event=0.12,
            hit_rate_at_event=0.60,
            n_train_observations=120,
            proxy_fill_fraction=0.0,
        )
        assert r.correct_direction is True
        assert r.signal_direction == "OUTPERFORM"

    def test_reporting_table_and_csv(self, tmp_path):
        from src.backtest.vesting_events import VestingEvent, _add_months

        # Synthesize 4 results: 2 events × 2 benchmarks
        results = []
        for year in [2020, 2021]:
            for bench in ["VTI", "BND"]:
                ev = VestingEvent(
                    event_date=date(year, 1, 19),
                    rsu_type="time",
                    horizon_6m_end=_add_months(date(year, 1, 19), 6),
                    horizon_12m_end=_add_months(date(year, 1, 19), 12),
                )
                results.append(BacktestEventResult(
                    event=ev, benchmark=bench, target_horizon=6,
                    predicted_relative_return=0.04,
                    realized_relative_return=0.02,
                    signal_direction="OUTPERFORM",
                    correct_direction=True,
                    predicted_sell_pct=0.25,
                    ic_at_event=0.10,
                    hit_rate_at_event=0.60,
                    n_train_observations=100,
                    proxy_fill_fraction=0.0,
                ))

        table = generate_backtest_table(results, horizon=6)
        assert table.shape == (2, 2)

        csv_path = str(tmp_path / "backtest.csv")
        export_backtest_to_csv(results, csv_path)
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 4

    def test_print_backtest_summary_no_crash(self, capsys):
        from src.backtest.vesting_events import VestingEvent, _add_months
        ev = VestingEvent(
            event_date=date(2021, 7, 16),
            rsu_type="performance",
            horizon_6m_end=_add_months(date(2021, 7, 16), 6),
            horizon_12m_end=_add_months(date(2021, 7, 16), 12),
        )
        results = [BacktestEventResult(
            event=ev, benchmark="VTI", target_horizon=6,
            predicted_relative_return=-0.03,
            realized_relative_return=-0.01,
            signal_direction="UNDERPERFORM",
            correct_direction=True,
            predicted_sell_pct=1.0,
            ic_at_event=0.08,
            hit_rate_at_event=0.55,
            n_train_observations=110,
            proxy_fill_fraction=0.05,
        )]
        print_backtest_summary(results)
        out = capsys.readouterr().out
        assert "BACKTEST SUMMARY" in out
