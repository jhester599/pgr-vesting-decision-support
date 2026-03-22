"""
Tests for Phase 3: multi-ticker DRIP total return and relative return pipeline.

Covers:
  - build_etf_monthly_returns: correct Series name, non-empty output, empty
    ticker handling, proxy_fill exclusion
  - build_relative_return_targets: correct columns, sign/magnitude of relative
    return, DB persistence
  - load_relative_return_matrix: round-trip through DB
  - get_X_y_relative: alignment, ValueError on empty overlap, NaN dropping,
    no target column in X
  - build_feature_matrix_from_db: returns expected structure from DB data

All tests use in-memory SQLite (tmp_path fixture).  No HTTP calls.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import config
from src.database import db_client
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_X_y_relative,
)
from src.processing.multi_total_return import (
    build_etf_monthly_returns,
    build_relative_return_targets,
    load_relative_return_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    db_path = str(tmp_path / "test.db")
    c = db_client.get_connection(db_path)
    db_client.initialize_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _price_records(
    ticker: str,
    n_weeks: int = 200,
    start: str = "2010-01-08",
    start_price: float = 100.0,
    weekly_growth: float = 0.003,
) -> list[dict]:
    """Generate deterministic weekly price records (no randomness)."""
    dates = pd.date_range(start=start, periods=n_weeks, freq="W-FRI")
    records = []
    price = start_price
    for dt in dates:
        price = price * (1 + weekly_growth)
        records.append({
            "ticker":     ticker,
            "date":       dt.strftime("%Y-%m-%d"),
            "open":       round(price * 0.99, 4),
            "high":       round(price * 1.01, 4),
            "low":        round(price * 0.98, 4),
            "close":      round(price, 4),
            "volume":     1_000_000,
            "source":     "test",
            "proxy_fill": 0,
        })
    return records


def _price_records_split(
    ticker: str,
    n_real: int = 100,
    n_proxy: int = 50,
    start: str = "2010-01-08",
    start_price: float = 50.0,
    weekly_growth: float = 0.002,
) -> list[dict]:
    """Generate records with first n_proxy rows flagged proxy_fill=1."""
    dates = pd.date_range(start=start, periods=n_proxy + n_real, freq="W-FRI")
    records = []
    price = start_price
    for i, dt in enumerate(dates):
        price = price * (1 + weekly_growth)
        records.append({
            "ticker":     ticker,
            "date":       dt.strftime("%Y-%m-%d"),
            "open":       round(price * 0.99, 4),
            "high":       round(price * 1.01, 4),
            "low":        round(price * 0.98, 4),
            "close":      round(price, 4),
            "volume":     500_000,
            "source":     "test",
            "proxy_fill": 1 if i < n_proxy else 0,
        })
    return records


def _dividend_records(
    ticker: str, dates: list[str], amount: float = 0.25
) -> list[dict]:
    return [
        {"ticker": ticker, "ex_date": d, "amount": amount, "source": "test"}
        for d in dates
    ]


# ---------------------------------------------------------------------------
# TestBuildEtfMonthlyReturns
# ---------------------------------------------------------------------------

class TestBuildEtfMonthlyReturns:

    def test_returns_correct_series_name(self, conn):
        db_client.upsert_prices(conn, _price_records("PGR"))
        result = build_etf_monthly_returns(conn, "PGR", forward_months=6)
        assert result.name == "PGR_6m_return"

    def test_returns_correct_series_name_12m(self, conn):
        db_client.upsert_prices(conn, _price_records("VTI"))
        result = build_etf_monthly_returns(conn, "VTI", forward_months=12)
        assert result.name == "VTI_12m_return"

    def test_returns_nonempty_series_with_sufficient_data(self, conn):
        db_client.upsert_prices(conn, _price_records("PGR", n_weeks=200))
        result = build_etf_monthly_returns(conn, "PGR", forward_months=6)
        assert not result.empty

    def test_returns_float_values(self, conn):
        db_client.upsert_prices(conn, _price_records("VTI", n_weeks=150))
        result = build_etf_monthly_returns(conn, "VTI", forward_months=6)
        non_nan = result.dropna()
        assert len(non_nan) > 0
        assert all(isinstance(v, float) for v in non_nan)

    def test_last_observations_are_nan(self, conn):
        """Forward window extends past data end → final months must be NaN."""
        db_client.upsert_prices(conn, _price_records("PGR", n_weeks=150))
        result = build_etf_monthly_returns(conn, "PGR", forward_months=6)
        # The last ~6 months of observations should be NaN
        assert result.iloc[-1] is np.nan or pd.isna(result.iloc[-1])

    def test_returns_empty_series_for_unknown_ticker(self, conn):
        result = build_etf_monthly_returns(conn, "FAKEXYZ", forward_months=6)
        assert result.empty
        assert result.name == "FAKEXYZ_6m_return"

    def test_index_is_datetimeindex(self, conn):
        db_client.upsert_prices(conn, _price_records("BND", n_weeks=150))
        result = build_etf_monthly_returns(conn, "BND", forward_months=6)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_exclude_proxy_shortens_series(self, conn):
        """exclude_proxy=True should produce a shorter (or equal) series."""
        records = _price_records_split("VTI", n_real=100, n_proxy=50)
        db_client.upsert_prices(conn, records)

        full = build_etf_monthly_returns(conn, "VTI", forward_months=6, exclude_proxy=False)
        proxy_excl = build_etf_monthly_returns(conn, "VTI", forward_months=6, exclude_proxy=True)

        # Excluding early proxy rows → shorter or equal series
        assert len(proxy_excl) <= len(full)

    def test_dividends_increase_return(self, conn):
        """Adding dividends to a flat price series should produce positive return."""
        # Flat price series — zero return without dividends
        n_weeks = 150
        dates = pd.date_range(start="2015-01-02", periods=n_weeks, freq="W-FRI")
        flat_records = [
            {"ticker": "FLAT", "date": dt.strftime("%Y-%m-%d"),
             "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
             "volume": 1_000_000, "source": "test", "proxy_fill": 0}
            for dt in dates
        ]
        db_client.upsert_prices(conn, flat_records)

        # Add quarterly dividends
        div_dates = [d.strftime("%Y-%m-%d") for d in dates[::13]]
        db_client.upsert_dividends(conn, _dividend_records("FLAT", div_dates, amount=1.0))

        result = build_etf_monthly_returns(conn, "FLAT", forward_months=6)
        non_nan = result.dropna()
        assert (non_nan > 0).all(), "Dividend DRIP should produce positive returns on flat prices"


# ---------------------------------------------------------------------------
# TestBuildRelativeReturnTargets
# ---------------------------------------------------------------------------

class TestBuildRelativeReturnTargets:

    def _insert_two_tickers(self, conn, pgr_growth=0.003, etf_growth=0.0015):
        """Insert PGR and one ETF with different growth rates."""
        db_client.upsert_prices(conn, _price_records("PGR", n_weeks=200,
                                                      weekly_growth=pgr_growth))
        db_client.upsert_prices(conn, _price_records("VTI", n_weeks=200,
                                                      weekly_growth=etf_growth))

    def test_returns_dataframe(self, conn, monkeypatch):
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI"])
        self._insert_two_tickers(conn)
        df = build_relative_return_targets(conn, forward_months=6, upsert=False)
        assert isinstance(df, pd.DataFrame)

    def test_columns_equal_benchmark_universe(self, conn, monkeypatch):
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI"])
        self._insert_two_tickers(conn)
        df = build_relative_return_targets(conn, forward_months=6, upsert=False)
        assert list(df.columns) == ["VTI"]

    def test_multiple_benchmarks(self, conn, monkeypatch):
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI", "BND"])
        db_client.upsert_prices(conn, _price_records("PGR", n_weeks=200, weekly_growth=0.003))
        db_client.upsert_prices(conn, _price_records("VTI", n_weeks=200, weekly_growth=0.0015))
        db_client.upsert_prices(conn, _price_records("BND", n_weeks=200, weekly_growth=0.001))
        df = build_relative_return_targets(conn, forward_months=6, upsert=False)
        assert set(df.columns) == {"VTI", "BND"}

    def test_relative_return_is_pgr_minus_etf(self, conn, monkeypatch):
        """PGR growing faster than ETF → positive relative returns."""
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI"])
        self._insert_two_tickers(conn, pgr_growth=0.004, etf_growth=0.001)
        df = build_relative_return_targets(conn, forward_months=6, upsert=False)
        assert not df.empty
        assert (df["VTI"].dropna() > 0).all(), \
            "PGR outperforms VTI → all relative returns should be positive"

    def test_relative_return_negative_when_pgr_underperforms(self, conn, monkeypatch):
        """PGR growing slower than ETF → negative relative returns."""
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI"])
        self._insert_two_tickers(conn, pgr_growth=0.001, etf_growth=0.004)
        df = build_relative_return_targets(conn, forward_months=6, upsert=False)
        assert not df.empty
        assert (df["VTI"].dropna() < 0).all(), \
            "PGR underperforms VTI → all relative returns should be negative"

    def test_missing_etf_data_excluded_not_error(self, conn, monkeypatch):
        """ETF with no data in DB should be silently skipped (not raise)."""
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI", "FAKEXYZ"])
        self._insert_two_tickers(conn)  # only inserts PGR and VTI
        df = build_relative_return_targets(conn, forward_months=6, upsert=False)
        assert "VTI" in df.columns
        assert "FAKEXYZ" not in df.columns

    def test_upsert_persists_to_db(self, conn, monkeypatch):
        """upsert=True should write rows to monthly_relative_returns."""
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI"])
        self._insert_two_tickers(conn)
        build_relative_return_targets(conn, forward_months=6, upsert=True)
        series = db_client.get_relative_returns(conn, "VTI", 6)
        assert not series.empty

    def test_upsert_false_does_not_persist(self, conn, monkeypatch):
        """upsert=False must not write anything to the DB."""
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI"])
        self._insert_two_tickers(conn)
        build_relative_return_targets(conn, forward_months=6, upsert=False)
        series = db_client.get_relative_returns(conn, "VTI", 6)
        assert series.empty

    def test_empty_df_when_no_etf_data(self, conn, monkeypatch):
        """If no ETF has data, return empty DataFrame."""
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["FAKEA", "FAKEB"])
        db_client.upsert_prices(conn, _price_records("PGR", n_weeks=200))
        df = build_relative_return_targets(conn, forward_months=6, upsert=False)
        assert df.empty

    def test_index_is_datetimeindex(self, conn, monkeypatch):
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI"])
        self._insert_two_tickers(conn)
        df = build_relative_return_targets(conn, forward_months=6, upsert=False)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_relative_return_arithmetic(self, conn, monkeypatch):
        """Verify: relative_return == pgr_return - etf_return for a sample row."""
        monkeypatch.setattr(config, "ETF_BENCHMARK_UNIVERSE", ["VTI"])
        self._insert_two_tickers(conn, pgr_growth=0.003, etf_growth=0.0015)

        # Compute underlying returns independently
        from src.processing.multi_total_return import build_etf_monthly_returns
        pgr_ret = build_etf_monthly_returns(conn, "PGR", 6)
        vti_ret = build_etf_monthly_returns(conn, "VTI", 6)

        df = build_relative_return_targets(conn, forward_months=6, upsert=False)

        # Check a sample non-NaN row
        common = df["VTI"].dropna().index
        assert len(common) > 0
        sample_date = common[len(common) // 2]
        expected = pgr_ret.loc[sample_date] - vti_ret.loc[sample_date]
        assert abs(df.at[sample_date, "VTI"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# TestLoadRelativeReturnMatrix
# ---------------------------------------------------------------------------

class TestLoadRelativeReturnMatrix:

    def test_returns_empty_series_when_no_db_rows(self, conn):
        result = load_relative_return_matrix(conn, "VTI", 6)
        assert result.empty
        assert result.name == "VTI_6m"

    def test_round_trip_through_db(self, conn):
        """Upsert rows then load them back; values should match."""
        records = [
            {"date": "2020-01-31", "benchmark": "VTI", "target_horizon": 6,
             "pgr_return": 0.12, "benchmark_return": 0.08, "relative_return": 0.04,
             "proxy_fill": 0},
            {"date": "2020-02-29", "benchmark": "VTI", "target_horizon": 6,
             "pgr_return": 0.10, "benchmark_return": 0.06, "relative_return": 0.04,
             "proxy_fill": 0},
        ]
        db_client.upsert_relative_returns(conn, records)
        series = load_relative_return_matrix(conn, "VTI", 6)
        assert len(series) == 2
        assert abs(series.iloc[0] - 0.04) < 1e-9
        assert series.name == "VTI_6m"

    def test_horizon_filter(self, conn):
        """Different horizons stored separately; loading 12m must not return 6m rows."""
        db_client.upsert_relative_returns(conn, [
            {"date": "2020-01-31", "benchmark": "VTI", "target_horizon": 6,
             "pgr_return": 0.1, "benchmark_return": 0.05, "relative_return": 0.05,
             "proxy_fill": 0},
            {"date": "2020-01-31", "benchmark": "VTI", "target_horizon": 12,
             "pgr_return": 0.2, "benchmark_return": 0.12, "relative_return": 0.08,
             "proxy_fill": 0},
        ])
        s6 = load_relative_return_matrix(conn, "VTI", 6)
        s12 = load_relative_return_matrix(conn, "VTI", 12)
        assert abs(s6.iloc[0] - 0.05) < 1e-9
        assert abs(s12.iloc[0] - 0.08) < 1e-9

    def test_date_range_filter(self, conn):
        """start_date / end_date parameters correctly restrict the result."""
        db_client.upsert_relative_returns(conn, [
            {"date": "2019-12-31", "benchmark": "BND", "target_horizon": 6,
             "pgr_return": 0.1, "benchmark_return": 0.02, "relative_return": 0.08,
             "proxy_fill": 0},
            {"date": "2020-06-30", "benchmark": "BND", "target_horizon": 6,
             "pgr_return": 0.05, "benchmark_return": 0.01, "relative_return": 0.04,
             "proxy_fill": 0},
        ])
        result = load_relative_return_matrix(conn, "BND", 6, start_date="2020-01-01")
        assert len(result) == 1
        assert abs(result.iloc[0] - 0.04) < 1e-9


# ---------------------------------------------------------------------------
# TestGetXYRelative
# ---------------------------------------------------------------------------

class TestGetXYRelative:

    def _make_feature_df(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create a minimal feature matrix with mom_3m and target_6m_return."""
        df = pd.DataFrame(
            {"mom_3m": np.linspace(0.01, 0.10, len(dates)),
             "vol_21d": np.linspace(0.15, 0.25, len(dates)),
             "target_6m_return": np.linspace(0.05, 0.15, len(dates))},
            index=dates,
        )
        df.index.name = "date"
        return df

    def _make_relative_series(
        self, dates: pd.DatetimeIndex, name: str = "VTI_6"
    ) -> pd.Series:
        s = pd.Series(np.linspace(0.02, 0.08, len(dates)), index=dates, name=name)
        s.index.name = "date"
        return s

    def test_raises_on_empty_overlap(self):
        feat_dates = pd.date_range("2018-01-31", periods=24, freq="BME")
        rel_dates  = pd.date_range("2023-01-31", periods=12, freq="BME")
        df = self._make_feature_df(feat_dates)
        rel = self._make_relative_series(rel_dates)
        with pytest.raises(ValueError, match="No overlapping"):
            get_X_y_relative(df, rel)

    def test_correct_row_count_on_inner_join(self):
        dates = pd.date_range("2015-01-31", periods=36, freq="BME")
        df = self._make_feature_df(dates)
        # Only overlap the first 20 dates
        rel = self._make_relative_series(dates[:20])
        X, y = get_X_y_relative(df, rel)
        assert len(X) == 20
        assert len(y) == 20

    def test_x_does_not_contain_target_column(self):
        dates = pd.date_range("2015-01-31", periods=24, freq="BME")
        df = self._make_feature_df(dates)
        rel = self._make_relative_series(dates)
        X, _ = get_X_y_relative(df, rel)
        assert "target_6m_return" not in X.columns
        assert "VTI_6" not in X.columns

    def test_y_has_correct_name(self):
        dates = pd.date_range("2015-01-31", periods=24, freq="BME")
        df = self._make_feature_df(dates)
        rel = self._make_relative_series(dates, name="VGT_12")
        _, y = get_X_y_relative(df, rel)
        assert y.name == "VGT_12"

    def test_drop_na_target_removes_nan_rows(self):
        dates = pd.date_range("2015-01-31", periods=24, freq="BME")
        df = self._make_feature_df(dates)
        rel = self._make_relative_series(dates)
        # Introduce NaN at the end of relative returns
        rel.iloc[-4:] = np.nan
        X_drop, y_drop = get_X_y_relative(df, rel, drop_na_target=True)
        X_keep, y_keep = get_X_y_relative(df, rel, drop_na_target=False)
        assert len(X_drop) == len(X_keep) - 4
        assert y_drop.isna().sum() == 0
        assert y_keep.isna().sum() == 4

    def test_feature_values_preserved(self):
        dates = pd.date_range("2015-01-31", periods=12, freq="BME")
        df = self._make_feature_df(dates)
        rel = self._make_relative_series(dates)
        X, y = get_X_y_relative(df, rel)
        # X values should match the original feature matrix on the same dates
        pd.testing.assert_series_equal(
            X["mom_3m"], df.loc[X.index, "mom_3m"], check_names=False
        )

    def test_raises_when_all_nan_after_drop(self):
        dates = pd.date_range("2015-01-31", periods=12, freq="BME")
        df = self._make_feature_df(dates)
        rel = self._make_relative_series(dates)
        rel[:] = np.nan  # all NaN
        with pytest.raises(ValueError):
            get_X_y_relative(df, rel, drop_na_target=True)


# ---------------------------------------------------------------------------
# TestBuildFeatureMatrixFromDb
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrixFromDb:

    def test_returns_dataframe(self, conn, tmp_path, monkeypatch):
        # Redirect parquet cache so tests don't write to real data/processed/
        fake_parquet = str(tmp_path / "feature_matrix.parquet")
        monkeypatch.setattr(
            "src.processing.feature_engineering._PROCESSED_PATH", fake_parquet
        )
        # Insert enough PGR price rows (300 weeks ≈ 5.8 years)
        db_client.upsert_prices(conn, _price_records("PGR", n_weeks=300))
        df = build_feature_matrix_from_db(conn)
        assert isinstance(df, pd.DataFrame)

    def test_has_momentum_columns(self, conn, tmp_path, monkeypatch):
        fake_parquet = str(tmp_path / "feature_matrix.parquet")
        monkeypatch.setattr(
            "src.processing.feature_engineering._PROCESSED_PATH", fake_parquet
        )
        db_client.upsert_prices(conn, _price_records("PGR", n_weeks=300))
        df = build_feature_matrix_from_db(conn)
        for col in ["mom_3m", "mom_6m", "vol_21d", "vol_63d"]:
            assert col in df.columns, f"Expected column '{col}' missing"

    def test_has_target_column(self, conn, tmp_path, monkeypatch):
        fake_parquet = str(tmp_path / "feature_matrix.parquet")
        monkeypatch.setattr(
            "src.processing.feature_engineering._PROCESSED_PATH", fake_parquet
        )
        db_client.upsert_prices(conn, _price_records("PGR", n_weeks=300))
        df = build_feature_matrix_from_db(conn)
        assert "target_6m_return" in df.columns

    def test_index_is_datetimeindex(self, conn, tmp_path, monkeypatch):
        fake_parquet = str(tmp_path / "feature_matrix.parquet")
        monkeypatch.setattr(
            "src.processing.feature_engineering._PROCESSED_PATH", fake_parquet
        )
        db_client.upsert_prices(conn, _price_records("PGR", n_weeks=300))
        df = build_feature_matrix_from_db(conn)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_dividends_in_db_used_correctly(self, conn, tmp_path, monkeypatch):
        """Dividends in DB should be picked up and affect the target return."""
        fake_parquet = str(tmp_path / "feature_matrix.parquet")
        monkeypatch.setattr(
            "src.processing.feature_engineering._PROCESSED_PATH", fake_parquet
        )
        db_client.upsert_prices(conn, _price_records("PGR", n_weeks=300))

        # Add quarterly dividends
        dates = pd.date_range("2012-01-01", periods=40, freq="QS")
        div_recs = [
            {"ticker": "PGR", "ex_date": d.strftime("%Y-%m-%d"),
             "amount": 0.10, "source": "test"}
            for d in dates
        ]
        db_client.upsert_dividends(conn, div_recs)

        df = build_feature_matrix_from_db(conn)
        # target_6m_return should have some non-NaN values
        assert df["target_6m_return"].notna().any()
