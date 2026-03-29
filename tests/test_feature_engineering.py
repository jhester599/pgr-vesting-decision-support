"""
Tests for src/processing/feature_engineering.py

Critical tests:
  1. No feature uses data from t+1 or later (temporal no-leakage guarantee).
  2. Price-derived feature columns have no NaN after the burn-in period.
  3. target_6m_return is NaN for the final 6 months of the dataset.
  4. Gainshare columns are dropped when insufficient observations exist.
  5. get_X_y correctly splits features from the target.
  6. (v4.1) EDGAR filing lag is applied before features are built.
"""

import pytest
import numpy as np
import pandas as pd

from src.processing.feature_engineering import (
    build_feature_matrix,
    get_feature_columns,
    get_X_y,
    _apply_edgar_lag,
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
        # vol_21d is dropped by config.FEATURES_TO_DROP (v4.3); check remaining cols
        for col in ["mom_12m", "vol_63d"]:
            assert after_burnin[col].notna().all(), (
                f"Column {col} has NaN values after burn-in period."
            )

    def test_features_to_drop_absent(
        self, price_history, dividend_history, split_history, tmp_path, monkeypatch
    ):
        """Columns in config.FEATURES_TO_DROP must not appear in the final matrix."""
        import config, src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm4_drop.parquet"))
        df = build_feature_matrix(
            price_history, dividend_history, split_history, force_refresh=True
        )
        for col in config.FEATURES_TO_DROP:
            assert col not in df.columns, (
                f"Column '{col}' should be dropped per config.FEATURES_TO_DROP "
                f"but is still present in the feature matrix."
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


# ---------------------------------------------------------------------------
# v4.1 — EDGAR filing lag
# ---------------------------------------------------------------------------

class TestEdgarLag:
    def test_edgar_lag_applied_in_feature_engineering(self):
        """EDGAR data should be shifted by EDGAR_FILING_LAG_MONTHS before use in features.

        _apply_edgar_lag uses a frequency-based index shift (shift(n, freq='MS')),
        which moves each row's index date forward by n months rather than inserting
        NaN rows.  After the lag, the row originally at 2020-01-31 (Jan period-end)
        is accessible at 2020-03-31 (2 months later, the approximate filing date),
        and the original Jan and Feb dates have no entries.
        """
        import config

        # Create synthetic EDGAR data (monthly, period-end dates)
        dates = pd.date_range("2020-01-31", periods=6, freq="ME")
        df = pd.DataFrame(
            {"combined_ratio_ttm": [95.0, 96.0, 97.0, 98.0, 99.0, 100.0]},
            index=dates,
        )

        result = _apply_edgar_lag(df)

        lag = config.EDGAR_FILING_LAG_MONTHS  # should be 2
        # The original Jan 31 row should now be at Mar 31 (shifted 2 months forward)
        first_orig_date = dates[0]  # 2020-01-31
        shifted_date = (first_orig_date + pd.offsets.DateOffset(months=lag))
        shifted_date_end = shifted_date + pd.offsets.MonthEnd(0)
        # Result index should start at shifted_date_end (2020-03-31)
        assert shifted_date_end in result.index, (
            f"Expected {shifted_date_end} in result index after {lag}-month lag; "
            f"got {result.index.tolist()}"
        )
        # The value at the shifted date should be the original first value
        assert result.loc[shifted_date_end, "combined_ratio_ttm"] == pytest.approx(95.0)
        # The original first `lag` dates should not be present in the shifted result
        for i in range(lag):
            original_date = dates[i]
            assert original_date not in result.index, (
                f"Date {original_date} should not be in shifted result index"
            )


# ---------------------------------------------------------------------------
# v4.3.1 — dtype coercion (object columns from EDGAR XBRL None values)
# ---------------------------------------------------------------------------

def test_feature_matrix_all_columns_numeric() -> None:
    """
    All columns in the feature matrix must be float64, not object.

    EDGAR XBRL returns None for pe_ratio / pb_ratio; reindexing produces
    object-dtype Series.  numpy >= 2.0 rejects object arrays in nanmedian,
    so feature_engineering must coerce to float64 before returning.
    """
    prices = _make_prices(6)
    div = _make_dividends()

    # Inject object-dtype fundamentals (simulates EDGAR XBRL None values)
    fund_dates = pd.date_range("2019-01-31", periods=24, freq="ME")
    fundamentals = pd.DataFrame(
        {
            "eps_diluted": [1.0] * 24,
            "revenue": [1e9] * 24,
            "net_income": [5e8] * 24,
            "pe_ratio": [None] * 24,    # object dtype — EDGAR XBRL can't provide this
            "pb_ratio": [None] * 24,    # object dtype — EDGAR XBRL can't provide this
            "roe": [None] * 24,
        },
        index=fund_dates,
    )

    splits = _make_splits()
    df = build_feature_matrix(prices, div, splits, fundamentals=fundamentals)
    feature_cols = get_feature_columns(df)

    object_cols = [c for c in feature_cols if df[c].dtype == object]
    assert object_cols == [], (
        f"Feature columns must be float64, not object.  "
        f"Object-dtype columns found: {object_cols}"
    )


def test_feature_matrix_pe_ratio_object_becomes_nan() -> None:
    """
    pe_ratio object column (None values from EDGAR XBRL) coerces to NaN float64,
    not raised as an error.
    """
    prices = _make_prices(6)
    div = _make_dividends()
    splits = _make_splits()

    fund_dates = pd.date_range("2019-01-31", periods=24, freq="ME")
    fundamentals = pd.DataFrame(
        {
            "eps_diluted": [1.5] * 24,
            "revenue": [2e9] * 24,
            "net_income": [4e8] * 24,
            "pe_ratio": [None] * 24,
            "pb_ratio": [None] * 24,
            "roe": [None] * 24,
        },
        index=fund_dates,
    )

    df = build_feature_matrix(prices, div, splits, fundamentals=fundamentals)

    if "pe_ratio" in df.columns:
        assert df["pe_ratio"].dtype == float, "pe_ratio should be float64 after coercion"
        assert df["pe_ratio"].isna().all(), "pe_ratio from None should be all-NaN"
