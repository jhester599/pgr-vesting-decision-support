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
import config

from src.processing.feature_engineering import (
    build_feature_matrix,
    build_feature_matrix_from_db,
    get_feature_columns,
    get_model_feature_columns,
    get_X_y,
    _apply_edgar_lag,
    _normalize_pgr_monthly_columns,
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

    def test_v15_derived_features_built_from_existing_inputs(
        self, price_history, dividend_history, split_history, tmp_path, monkeypatch
    ):
        """Key v15 derived features should be emitted when their source inputs exist."""
        import config, src.processing.feature_engineering as fe

        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 1)
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_v15.parquet"))

        month_ends = pd.date_range("2010-01-31", periods=96, freq="ME")
        pgr_monthly = pd.DataFrame(
            {
                "combined_ratio": np.linspace(96.0, 90.0, len(month_ends)),
                "pif_total": np.linspace(1_000_000, 1_600_000, len(month_ends)),
                "pif_growth_yoy": np.linspace(0.01, 0.08, len(month_ends)),
                "net_premiums_written": np.linspace(800.0, 1400.0, len(month_ends)),
                "net_premiums_earned": np.linspace(780.0, 1360.0, len(month_ends)),
                "underwriting_income": np.linspace(20.0, 140.0, len(month_ends)),
                "unearned_premium_growth_yoy": np.linspace(0.01, 0.10, len(month_ends)),
                "roe_net_income_ttm": np.linspace(0.08, 0.22, len(month_ends)),
                "book_value_per_share": np.linspace(20.0, 75.0, len(month_ends)),
                "investment_income": np.linspace(40.0, 120.0, len(month_ends)),
                "investment_book_yield": np.linspace(0.025, 0.055, len(month_ends)),
                "fixed_income_duration": np.linspace(3.2, 4.4, len(month_ends)),
                "shareholders_equity": np.linspace(5000.0, 9500.0, len(month_ends)),
                "loss_lae_reserves": np.linspace(3200.0, 4800.0, len(month_ends)),
                "pif_direct_auto": np.linspace(450000.0, 900000.0, len(month_ends)),
                "pif_total_personal_lines": np.linspace(700000.0, 1200000.0, len(month_ends)),
                "total_net_realized_gains": np.linspace(10.0, 45.0, len(month_ends)),
                "net_income": np.linspace(60.0, 180.0, len(month_ends)),
                "net_unrealized_gains_fixed": np.linspace(-40.0, 120.0, len(month_ends)),
                "losses_lae": np.linspace(450.0, 910.0, len(month_ends)),
                "policy_acquisition_costs": np.linspace(90.0, 170.0, len(month_ends)),
                "other_underwriting_expenses": np.linspace(55.0, 95.0, len(month_ends)),
            },
            index=month_ends,
        )

        fred_macro = pd.DataFrame(
            {
                "T10Y2Y": np.linspace(2.0, 0.5, len(month_ends)),
                "GS5": np.linspace(2.5, 3.0, len(month_ends)),
                "GS2": np.linspace(1.0, 2.0, len(month_ends)),
                "GS10": np.linspace(2.2, 4.0, len(month_ends)),
                "T10YIE": np.linspace(1.5, 2.4, len(month_ends)),
                "BAA10Y": np.linspace(1.2, 2.1, len(month_ends)),
                "BAMLH0A0HYM2": np.linspace(3.5, 5.0, len(month_ends)),
                "NFCI": np.linspace(-0.5, 0.2, len(month_ends)),
                "VIXCLS": np.linspace(15.0, 22.0, len(month_ends)),
                "CUSR0000SETA02": np.linspace(100.0, 140.0, len(month_ends)),
                "CUSR0000SAM2": np.linspace(100.0, 130.0, len(month_ends)),
                "CUSR0000SETE": np.linspace(100.0, 135.0, len(month_ends)),
                "PCU5241265241261": np.linspace(100.0, 150.0, len(month_ends)),
                "DTWEXBGS": np.linspace(100.0, 112.0, len(month_ends)),
                "DCOILWTICO": np.linspace(45.0, 78.0, len(month_ends)),
                "MORTGAGE30US": np.linspace(3.5, 6.5, len(month_ends)),
                "WPU45110101": np.linspace(100.0, 132.0, len(month_ends)),
                "PPIACO": np.linspace(100.0, 122.0, len(month_ends)),
                "MRTSSM447USN": np.linspace(12000.0, 15500.0, len(month_ends)),
                "THREEFYTP10": np.linspace(1.5, 0.4, len(month_ends)),
                "SP500_PE_RATIO_MULTPL": np.linspace(18.0, 25.0, len(month_ends)),
                "SP500_EARNINGS_YIELD_MULTPL": np.linspace(5.5, 4.0, len(month_ends)),
                "SP500_PRICE_TO_BOOK_MULTPL": np.linspace(2.8, 4.5, len(month_ends)),
            },
            index=month_ends,
        )

        fundamentals = pd.DataFrame(
            {
                "pe_ratio": np.linspace(12.0, 22.0, len(month_ends)),
                "pb_ratio": np.linspace(1.8, 3.6, len(month_ends)),
                "roe": np.linspace(0.08, 0.16, len(month_ends)),
            },
            index=month_ends,
        )

        df = build_feature_matrix(
            price_history,
            dividend_history,
            split_history,
            fundamentals=fundamentals,
            pgr_monthly=pgr_monthly,
            fred_macro=fred_macro,
            force_refresh=True,
        )

        expected_cols = {
            "monthly_combined_ratio_delta",
            "pif_growth_acceleration",
            "npw_per_pif_yoy",
            "npw_vs_npe_spread_pct",
            "underwriting_margin_ttm",
            "book_value_per_share_growth_yoy",
            "duration_rate_shock_3m",
            "severity_index_yoy",
            "rate_adequacy_gap_yoy",
            "auto_pricing_power_spread",
            "reserve_to_npe_ratio",
            "direct_channel_pif_share_ttm",
            "channel_mix_direct_pct_yoy",
            "realized_gain_to_net_income_ratio",
            "unrealized_gain_pct_equity",
            "loss_ratio_ttm",
            "expense_ratio_ttm",
            "pgr_premium_to_surplus",
            "breakeven_inflation_10y",
            "breakeven_momentum_3m",
            "real_yield_change_6m",
            "baa10y_spread",
            "usd_broad_return_3m",
            "usd_momentum_6m",
            "wti_return_3m",
            "mortgage_spread_30y_10y",
            "credit_spread_ratio",
            "term_premium_10y",
            "legal_services_ppi_relative",
            "gasoline_retail_sales_delta",
            "pgr_pe_vs_market_pe",
            "pgr_price_to_book_relative",
            "equity_risk_premium",
            "excess_bond_premium_proxy",
        }
        missing = expected_cols.difference(df.columns)
        assert not missing, f"Expected v15-derived columns missing from matrix: {sorted(missing)}"

    def test_v18_price_relative_features_built_from_existing_prices(
        self, monkeypatch, price_history
    ) -> None:
        """Benchmark-side v18 relative features should be derivable from existing prices."""
        from src.database import db_client

        def _make_scaled_prices(scale: float) -> pd.DataFrame:
            scaled = price_history.copy()
            scaled["open"] *= scale
            scaled["high"] *= scale
            scaled["low"] *= scale
            scaled["close"] *= scale
            return scaled

        empty_dividends = pd.DataFrame(
            columns=["amount", "source"],
            index=pd.DatetimeIndex([], name="ex_date"),
        )
        empty_splits = pd.DataFrame(
            columns=["split_ratio", "numerator", "denominator"],
            index=pd.DatetimeIndex([], name="split_date"),
        )

        price_map = {
            "PGR": _make_scaled_prices(1.00),
            "VWO": _make_scaled_prices(0.85),
            "VXUS": _make_scaled_prices(0.80),
            "GLD": _make_scaled_prices(1.20),
            "BND": _make_scaled_prices(0.75),
            "DBC": _make_scaled_prices(1.05),
            "VOO": _make_scaled_prices(0.95),
            "VFH": _make_scaled_prices(0.90),
            "ALL": _make_scaled_prices(0.88),
            "TRV": _make_scaled_prices(0.92),
            "CB": _make_scaled_prices(0.94),
            "HIG": _make_scaled_prices(0.89),
        }

        monkeypatch.setattr(db_client, "get_prices", lambda conn, ticker, *args, **kwargs: price_map.get(ticker, pd.DataFrame()))
        monkeypatch.setattr(db_client, "get_dividends", lambda conn, ticker: empty_dividends)
        monkeypatch.setattr(db_client, "get_splits", lambda conn, ticker: empty_splits)
        monkeypatch.setattr(db_client, "get_pgr_fundamentals", lambda conn: pd.DataFrame())
        monkeypatch.setattr(db_client, "get_pgr_edgar_monthly", lambda conn: pd.DataFrame())
        monkeypatch.setattr(db_client, "get_fred_macro", lambda conn, series: pd.DataFrame())

        df = build_feature_matrix_from_db(conn=None, force_refresh=True)

        expected_cols = {
            "vwo_vxus_spread_6m",
            "gold_vs_treasury_6m",
            "commodity_equity_momentum",
            "pgr_vs_peers_6m",
        }
        missing = expected_cols.difference(df.columns)
        assert not missing, f"Expected v18 price-relative columns missing from matrix: {sorted(missing)}"


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

    def test_model_feature_columns_applies_override(self):
        df = pd.DataFrame(
            columns=[
                "mom_3m",
                "mom_6m",
                "mom_12m",
                "vol_63d",
                "yield_slope",
                "yield_curvature",
                "real_rate_10y",
                "credit_spread_hy",
                "nfci",
                "vix",
                "vmt_yoy",
                "investment_income_growth_yoy",
                "roe_net_income_ttm",
                "underwriting_income",
                "target_6m_return",
            ]
        )

        feature_cols = get_model_feature_columns(df, model_type="elasticnet")

        assert feature_cols == config.MODEL_FEATURE_OVERRIDES["elasticnet"]


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

    def test_edgar_lag_snaps_weekend_month_end_to_business_month_end(self):
        dates = pd.DatetimeIndex([pd.Timestamp("2020-02-29")], name="date")
        df = pd.DataFrame({"combined_ratio_ttm": [95.0]}, index=dates)

        result = _apply_edgar_lag(df)

        assert result.index[0].dayofweek < 5
        assert result.index[0] == pd.Timestamp("2020-04-30")


class TestPgrMonthlyNormalization:
    def test_normalizes_legacy_roe_column_name(self):
        raw = pd.DataFrame(
            {"roe_net_income_trailing_12m": [0.12]},
            index=pd.DatetimeIndex([pd.Timestamp("2020-01-31")], name="date"),
        )

        normalized = _normalize_pgr_monthly_columns(raw)

        assert "roe_net_income_ttm" in normalized.columns
        assert normalized.iloc[0]["roe_net_income_ttm"] == pytest.approx(0.12)


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


def test_build_feature_matrix_from_db_logs_optional_synthetic_feature_failure(
    monkeypatch, price_history, caplog
) -> None:
    from src.database import db_client

    empty_dividends = pd.DataFrame(
        columns=["amount", "source"],
        index=pd.DatetimeIndex([], name="ex_date"),
    )
    empty_splits = pd.DataFrame(
        columns=["split_ratio", "numerator", "denominator"],
        index=pd.DatetimeIndex([], name="split_date"),
    )

    def fake_get_prices(conn, ticker, *args, **kwargs):
        if ticker == "KIE":
            raise RuntimeError("synthetic KIE price failure")
        return price_history.copy()

    monkeypatch.setattr(db_client, "get_prices", fake_get_prices)
    monkeypatch.setattr(db_client, "get_dividends", lambda conn, ticker: empty_dividends)
    monkeypatch.setattr(db_client, "get_splits", lambda conn, ticker: empty_splits)
    monkeypatch.setattr(db_client, "get_pgr_fundamentals", lambda conn: pd.DataFrame())
    monkeypatch.setattr(db_client, "get_pgr_edgar_monthly", lambda conn: pd.DataFrame())
    monkeypatch.setattr(db_client, "get_fred_macro", lambda conn, series: pd.DataFrame())

    with caplog.at_level("ERROR"):
        df = build_feature_matrix_from_db(conn=None, force_refresh=True)

    assert isinstance(df, pd.DataFrame)
    assert "Could not build synthetic feature pgr_vs_kie_6m" in caplog.text
    assert "synthetic KIE price failure" in caplog.text
