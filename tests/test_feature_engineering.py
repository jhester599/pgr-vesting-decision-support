"""
Tests for src/processing/feature_engineering.py

Critical tests:
  1. No feature uses data from t+1 or later (temporal no-leakage guarantee).
  2. Price-derived feature columns have no NaN after the burn-in period.
  3. target_6m_return is NaN for the final 6 months of the dataset.
  4. Gainshare columns are dropped when insufficient observations exist.
  5. get_X_y correctly splits features from the target.
"""

import pytest
import numpy as np
import pandas as pd

from src.processing.feature_engineering import (
    build_feature_matrix,
    get_feature_columns,
    get_X_y,
)


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def _make_prices(n_years: int = 5) -> pd.DataFrame:
    """Linearly trending prices for n_years of business days."""
    dates = pd.bdate_range("2010-01-01", periods=n_years * 252)
    close = np.linspace(50.0, 150.0, len(dates))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1_000_000},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _make_dividends() -> pd.DataFrame:
    """Quarterly $0.10 dividends."""
    div_dates = pd.bdate_range("2010-01-04", periods=20, freq="QS")
    return pd.DataFrame(
        {"dividend": 0.10},
        index=pd.DatetimeIndex(div_dates, name="date"),
    )


def _make_splits() -> pd.DataFrame:
    """No splits."""
    return pd.DataFrame(
        {"numerator": [], "denominator": [], "split_ratio": []},
        index=pd.DatetimeIndex([], name="date"),
    )


@pytest.fixture()
def price_history():
    return _make_prices()


@pytest.fixture()
def dividend_history():
    return _make_dividends()


@pytest.fixture()
def split_history():
    return _make_splits()


@pytest.fixture()
def feature_matrix(price_history, dividend_history, split_history, tmp_path, monkeypatch):
    """Build a feature matrix without writing to the real data dir."""
    import config
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
    import src.processing.feature_engineering as fe
    monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "feature_matrix.parquet"))
    return build_feature_matrix(
        price_history, dividend_history, split_history, force_refresh=True
    )


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_no_future_leakage_in_momentum(
        self, price_history, dividend_history, split_history, tmp_path, monkeypatch
    ):
        """
        For each monthly observation t, mom_12m[t] must equal
        (close[t] / close[t - 252 days]) - 1 using only past data.

        We test this by checking that the momentum series is strictly
        computed from prices on or before t, not after t.
        """
        import config, src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        df = build_feature_matrix(
            price_history, dividend_history, split_history, force_refresh=True
        )
        # Verify mom_12m at a specific known date
        daily_close = price_history["close"]
        monthly_dates = daily_close.resample("BME").last().index
        # Take an observation well into the dataset (avoid burn-in NaN)
        test_date = monthly_dates[15]
        expected_shifted = daily_close.shift(252).reindex([test_date], method="ffill").iloc[0]
        expected_mom = (daily_close.asof(test_date) / expected_shifted) - 1.0
        actual_mom = df.at[test_date, "mom_12m"]
        assert np.isclose(actual_mom, expected_mom, rtol=1e-6), (
            f"mom_12m mismatch at {test_date}: expected {expected_mom:.6f}, got {actual_mom:.6f}"
        )

    def test_target_nan_in_final_6_months(
        self, price_history, dividend_history, split_history, tmp_path, monkeypatch
    ):
        """target_6m_return MUST be NaN for the last 6 months — no look-ahead leakage."""
        import config, src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm2.parquet"))
        df = build_feature_matrix(
            price_history, dividend_history, split_history, force_refresh=True
        )
        data_end = price_history.index.max()
        cutoff = data_end - pd.DateOffset(months=6)
        tail = df.loc[df.index > cutoff, "target_6m_return"]
        assert tail.isna().all(), (
            f"target_6m_return must be NaN after {cutoff.date()}, "
            f"but got non-NaN values: {tail.dropna()}"
        )

    def test_price_features_no_nan_after_burnin(
        self, price_history, dividend_history, split_history, tmp_path, monkeypatch
    ):
        """Price-derived features should be non-NaN after the 252-day burn-in period."""
        import config, src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm3.parquet"))
        df = build_feature_matrix(
            price_history, dividend_history, split_history, force_refresh=True
        )
        burn_in_end = price_history.index.min() + pd.DateOffset(months=14)
        after_burnin = df.loc[df.index > burn_in_end]
        for col in ["mom_12m", "vol_21d", "vol_63d"]:
            assert after_burnin[col].notna().all(), (
                f"Column {col} has NaN values after burn-in period."
            )

    def test_gainshare_dropped_when_insufficient(
        self, price_history, dividend_history, split_history, tmp_path, monkeypatch
    ):
        """Gainshare columns must be dropped if EDGAR cache provides < 60 rows."""
        import config, src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 60)
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm5.parquet"))

        # Provide sparse pgr_monthly data (only 5 rows — below threshold)
        sparse_monthly = pd.DataFrame(
            {"combined_ratio": [90.0] * 5, "pif_total": [1000.0] * 5,
             "pif_growth_yoy": [0.05] * 5, "gainshare_estimate": [1.0] * 5},
            index=pd.DatetimeIndex(
                pd.bdate_range("2010-01-01", periods=5, freq="MS"), name="date"
            ),
        )
        df = build_feature_matrix(
            price_history, dividend_history, split_history,
            pgr_monthly=sparse_monthly, force_refresh=True
        )
        for col in ["combined_ratio_ttm", "pif_growth_yoy", "gainshare_est"]:
            assert col not in df.columns, (
                f"Column {col} should be dropped when fewer than 60 observations."
            )


# ---------------------------------------------------------------------------
# get_X_y
# ---------------------------------------------------------------------------

class TestGetXY:
    def test_target_not_in_X(self, feature_matrix):
        X, y = get_X_y(feature_matrix)
        assert "target_6m_return" not in X.columns

    def test_y_name(self, feature_matrix):
        _, y = get_X_y(feature_matrix)
        assert y.name == "target_6m_return"

    def test_no_nan_target_when_dropped(self, feature_matrix):
        _, y = get_X_y(feature_matrix, drop_na_target=True)
        assert y.notna().all(), "After drop_na_target=True, y must have zero NaN."

    def test_nan_target_preserved_when_not_dropped(self, feature_matrix):
        _, y = get_X_y(feature_matrix, drop_na_target=False)
        # The final 6 months should remain NaN
        assert y.isna().any(), "Expected NaN values in y when drop_na_target=False."

    def test_X_y_same_index(self, feature_matrix):
        X, y = get_X_y(feature_matrix)
        assert X.index.equals(y.index)

    def test_feature_columns_helper(self, feature_matrix):
        feature_cols = get_feature_columns(feature_matrix)
        assert "target_6m_return" not in feature_cols
        assert len(feature_cols) == len(feature_matrix.columns) - 1
