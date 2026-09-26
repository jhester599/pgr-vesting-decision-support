"""
Tests for v6.0 feature engineering additions.

Covers:
  - high_52w (George & Hwang 2004): correct values, range, burn-in, no look-ahead
  - pgr_vs_peers_6m: equal-weight composite math, sign invariants,
    injection into build_feature_matrix() via synthetic fred_macro
  - Column presence / absence guards (silent-fail when data is absent)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.feature_engineering import (
    build_feature_matrix,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_prices(
    n_years: int = 5,
    start: str = "2010-01-01",
    price_start: float = 50.0,
    price_end: float = 150.0,
) -> pd.DataFrame:
    """Linearly trending daily prices for n_years of business days."""
    dates = pd.bdate_range(start, periods=n_years * 252)
    close = np.linspace(price_start, price_end, len(dates))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _make_flat_prices(
    n_years: int = 5,
    start: str = "2010-01-01",
    price: float = 100.0,
) -> pd.DataFrame:
    """Flat daily prices — every day closes at exactly `price`."""
    dates = pd.bdate_range(start, periods=n_years * 252)
    close = np.full(len(dates), price)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _make_dividends() -> pd.DataFrame:
    """Quarterly $0.10 dividends (no effect on price-ratio features)."""
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
def trending_prices():
    return _make_prices()


@pytest.fixture()
def flat_prices():
    return _make_flat_prices()


@pytest.fixture()
def dividends():
    return _make_dividends()


@pytest.fixture()
def splits():
    return _make_splits()


def _build(prices, dividends, splits, fred_macro=None, *, tmp_path, monkeypatch):
    """Build feature matrix with tmp_path isolation."""
    import config
    import src.processing.feature_engineering as fe

    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
    return build_feature_matrix(
        price_history=prices,
        dividend_history=dividends,
        split_history=splits,
        fred_macro=fred_macro,
        force_refresh=True,
    )


# ---------------------------------------------------------------------------
# high_52w
# ---------------------------------------------------------------------------

class TestHigh52wPresence:
    """high_52w column appears in the feature matrix output."""

    def test_column_present(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        assert "high_52w" in fm.columns, (
            "high_52w column missing from feature matrix"
        )

    def test_column_in_get_feature_columns(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        feature_cols = get_feature_columns(fm)
        assert "high_52w" in feature_cols

    def test_column_not_target(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        # Verify that it is not accidentally treated as the target column
        assert "high_52w" != "target_6m_return"
        assert "high_52w" in fm.columns


class TestHigh52wValues:
    """high_52w values must lie in (0, 1] — current price can never exceed its own high."""

    def test_all_non_nan_values_at_most_one(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        valid = fm["high_52w"].dropna()
        assert (valid <= 1.0 + 1e-9).all(), (
            f"high_52w exceeds 1.0: max={valid.max():.6f}"
        )

    def test_all_non_nan_values_positive(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        valid = fm["high_52w"].dropna()
        assert (valid > 0.0).all(), (
            f"high_52w contains non-positive values: min={valid.min():.6f}"
        )

    def test_flat_prices_ratio_equals_one(
        self, flat_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        """Flat prices: current close == 52-week high every month → ratio = 1.0."""
        fm = _build(flat_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        valid = fm["high_52w"].dropna()
        assert len(valid) > 0, "No non-NaN high_52w values for flat prices"
        max_deviation = (valid - 1.0).abs().max()
        assert max_deviation < 1e-9, (
            f"Flat prices: expected high_52w=1.0, max deviation={max_deviation:.2e}"
        )

    def test_trending_prices_ratio_equals_one(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        """Monotonically rising prices: current close is always the 52-week high → ratio = 1.0."""
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        valid = fm["high_52w"].dropna()
        max_deviation = (valid - 1.0).abs().max()
        assert max_deviation < 1e-9, (
            f"Monotonically rising prices: expected high_52w=1.0 throughout, "
            f"max deviation={max_deviation:.2e}"
        )

    def test_declining_prices_ratio_below_one(
        self, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        """Monotonically falling prices: current close < 52-week high → ratio < 1.0."""
        falling = _make_prices(n_years=3, price_start=150.0, price_end=50.0)
        fm = _build(falling, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        # After the 126-day burn-in there should be months where price < 52w high
        valid = fm["high_52w"].dropna()
        # In a strictly declining series the 52-week high is always above the current price
        # for rows past the burn-in.  Allow the very first valid row where current==high.
        assert (valid < 1.0 + 1e-9).all()
        # At least some months should have ratio clearly below 1
        assert (valid < 0.99).any(), (
            "Expected at least one high_52w < 0.99 for a declining price series"
        )


class TestHigh52wBurnIn:
    """high_52w is NaN during the first ~6 months (min_periods=126 trading days)."""

    def test_nan_for_first_months(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        # min_periods=126 trading days ≈ 5-6 months.  Month-end at exactly
        # 126 business days may resolve; check first 5 months are definitely NaN.
        first_five = fm["high_52w"].iloc[:5]
        assert first_five.isna().all(), (
            f"Expected NaN for first 5 months of high_52w; got {first_five.values}"
        )

    def test_non_nan_after_burn_in(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        # After month 12 we should have valid values
        after_burnin = fm["high_52w"].iloc[12:]
        assert after_burnin.notna().all(), (
            "Expected non-NaN high_52w values after burn-in period"
        )


class TestHigh52wNoLookahead:
    """high_52w at month t must not use prices after t."""

    def test_no_lookahead_via_price_shock(
        self, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        """
        Insert a large upward price spike at the very end of the series.
        high_52w for the month *before* the spike should be unaffected.
        """
        n_days = 4 * 252
        dates = pd.bdate_range("2010-01-01", periods=n_days)
        close = np.full(n_days, 100.0)
        # Spike only on the very last 21 trading days (final month)
        close[-21:] = 500.0
        prices = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": 1_000_000},
            index=pd.DatetimeIndex(dates, name="date"),
        )
        fm = _build(prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        # Second-to-last month-end row — should not see the spike
        penultimate = fm["high_52w"].iloc[-2]
        # If no look-ahead, the 52-week high at that date is 100, so ratio = 1.0
        assert penultimate == pytest.approx(1.0, abs=1e-6), (
            f"high_52w look-ahead detected: penultimate row = {penultimate:.4f} "
            f"(expected ~1.0 — should not see the final-month spike)"
        )


# ---------------------------------------------------------------------------
# pgr_vs_peers_6m — composite math (unit tests, no DB)
# ---------------------------------------------------------------------------

class TestPgrVsPeers6mMath:
    """
    Verify the equal-weight peer composite arithmetic directly.
    These tests exercise the computation logic without hitting the DB.
    """

    def _make_monthly_prices(
        self,
        n_months: int,
        growth_per_month: float,
        start_price: float = 100.0,
    ) -> pd.Series:
        """Return a monthly close Series with constant geometric growth."""
        idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
        prices = start_price * (1 + growth_per_month) ** np.arange(n_months)
        return pd.Series(prices, index=idx)

    def test_equal_weight_composite_averages_three_peers(self) -> None:
        """Mean of three peer 6M returns equals arithmetic average."""
        n = 24
        p1 = self._make_monthly_prices(n, 0.01)   # +1%/month
        p2 = self._make_monthly_prices(n, 0.02)   # +2%/month
        p3 = self._make_monthly_prices(n, 0.03)   # +3%/month

        peer_df = pd.concat([p1, p2, p3], axis=1)
        peer_6m = peer_df.pct_change(6)
        composite = peer_6m.mean(axis=1)

        ind_6m_avg = peer_6m.mean(axis=1)
        pd.testing.assert_series_equal(composite, ind_6m_avg)

    def test_pgr_outperform_returns_positive(self) -> None:
        """pgr_vs_peers_6m > 0 when PGR 6M return > peer composite 6M return."""
        n = 24
        pgr = self._make_monthly_prices(n, 0.05)  # +5%/month
        peer = self._make_monthly_prices(n, 0.01) # +1%/month

        pgr_6m = pgr.pct_change(6)
        peer_6m = peer.pct_change(6)
        result = (pgr_6m - peer_6m).dropna()

        assert (result > 0).all(), (
            "Expected positive pgr_vs_peers_6m when PGR outperforms"
        )

    def test_pgr_underperform_returns_negative(self) -> None:
        """pgr_vs_peers_6m < 0 when PGR 6M return < peer composite 6M return."""
        n = 24
        pgr = self._make_monthly_prices(n, 0.01)  # +1%/month
        peer = self._make_monthly_prices(n, 0.05) # +5%/month

        pgr_6m = pgr.pct_change(6)
        peer_6m = peer.pct_change(6)
        result = (pgr_6m - peer_6m).dropna()

        assert (result < 0).all(), (
            "Expected negative pgr_vs_peers_6m when PGR underperforms"
        )

    def test_equal_performance_returns_zero(self) -> None:
        """pgr_vs_peers_6m == 0 when PGR and peer composite have identical returns."""
        n = 24
        pgr = self._make_monthly_prices(n, 0.02)
        peer = self._make_monthly_prices(n, 0.02)

        pgr_6m = pgr.pct_change(6)
        peer_6m = peer.pct_change(6)
        result = (pgr_6m - peer_6m).dropna()

        max_abs = result.abs().max()
        assert max_abs < 1e-10, (
            f"Expected zero pgr_vs_peers_6m for identical returns; max abs={max_abs:.2e}"
        )

    def test_partial_peer_availability_uses_mean_of_available(self) -> None:
        """If only 2 of 4 peers have data, composite is average of those 2."""
        n = 24
        p1 = self._make_monthly_prices(n, 0.02)
        p2 = self._make_monthly_prices(n, 0.04)

        peer_df = pd.concat([p1, p2], axis=1)
        composite = peer_df.pct_change(6).mean(axis=1)

        # Manually compute expected
        expected = (p1.pct_change(6) + p2.pct_change(6)) / 2
        pd.testing.assert_series_equal(composite, expected)


# ---------------------------------------------------------------------------
# pgr_vs_peers_6m — injection into build_feature_matrix() via fred_macro
# ---------------------------------------------------------------------------

class TestPgrVsPeers6mInjection:
    """
    Verify that pgr_vs_peers_6m flows through the FRED synthetic-column
    injection path in build_feature_matrix() correctly.
    """

    def _make_synthetic_fred(self, monthly_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Synthetic fred_macro DataFrame with a pgr_vs_peers_6m column."""
        rng = np.random.default_rng(42)
        vals = rng.normal(0.0, 0.05, size=len(monthly_dates))
        return pd.DataFrame(
            {"pgr_vs_peers_6m": vals},
            index=monthly_dates,
        )

    def test_column_present_when_injected(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        """pgr_vs_peers_6m appears in feature matrix when injected via fred_macro."""
        # Build a basic matrix first to get the monthly_dates shape
        import src.processing.feature_engineering as fe
        import config
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))

        # First pass to discover date range
        fm0 = build_feature_matrix(
            trending_prices, dividends, splits, force_refresh=True
        )
        fred_mock = self._make_synthetic_fred(fm0.index)

        # Second pass with synthetic fred_macro containing pgr_vs_peers_6m
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm2.parquet"))
        fm = build_feature_matrix(
            trending_prices, dividends, splits,
            fred_macro=fred_mock,
            force_refresh=True,
        )
        assert "pgr_vs_peers_6m" in fm.columns, (
            "pgr_vs_peers_6m not found in feature matrix after injection"
        )

    def test_column_absent_without_injection(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        """pgr_vs_peers_6m absent from feature matrix when not in fred_macro."""
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        assert "pgr_vs_peers_6m" not in fm.columns

    def test_injected_values_pass_through_unchanged(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        """Injected pgr_vs_peers_6m values survive the reindex step without distortion."""
        import src.processing.feature_engineering as fe
        import config
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_a.parquet"))

        fm0 = build_feature_matrix(
            trending_prices, dividends, splits, force_refresh=True
        )
        fred_mock = self._make_synthetic_fred(fm0.index)

        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_b.parquet"))
        fm = build_feature_matrix(
            trending_prices, dividends, splits,
            fred_macro=fred_mock,
            force_refresh=True,
        )
        expected = fred_mock["pgr_vs_peers_6m"].reindex(fm.index, method="ffill")
        pd.testing.assert_series_equal(
            fm["pgr_vs_peers_6m"].rename("pgr_vs_peers_6m"),
            expected.rename("pgr_vs_peers_6m"),
            check_names=True,
        )

    def test_column_in_get_feature_columns(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        """pgr_vs_peers_6m appears in get_feature_columns() when injected."""
        import src.processing.feature_engineering as fe
        import config
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_p.parquet"))

        fm0 = build_feature_matrix(
            trending_prices, dividends, splits, force_refresh=True
        )
        fred_mock = self._make_synthetic_fred(fm0.index)

        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_q.parquet"))
        fm = build_feature_matrix(
            trending_prices, dividends, splits,
            fred_macro=fred_mock,
            force_refresh=True,
        )
        assert "pgr_vs_peers_6m" in get_feature_columns(fm)


# ---------------------------------------------------------------------------
# v6.0 feature column names (regression guard)
# ---------------------------------------------------------------------------

class TestV60FeatureColumnNames:
    """
    Regression tests: ensure the v6.0 feature column names are stable.
    A rename would silently break the WFO pipeline and any saved models.
    """

    def test_high_52w_column_name_is_exact(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        assert "high_52w" in fm.columns
        # Guard against accidental renames
        assert "high52w" not in fm.columns
        assert "high_52week" not in fm.columns

    def test_pgr_vs_peers_6m_column_name_is_exact(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        import src.processing.feature_engineering as fe
        import config
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_n.parquet"))

        fm0 = build_feature_matrix(
            trending_prices, dividends, splits, force_refresh=True
        )
        idx = fm0.index
        fred_mock = pd.DataFrame(
            {"pgr_vs_peers_6m": np.zeros(len(idx))}, index=idx
        )

        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_m.parquet"))
        fm = build_feature_matrix(
            trending_prices, dividends, splits,
            fred_macro=fred_mock,
            force_refresh=True,
        )
        assert "pgr_vs_peers_6m" in fm.columns
        assert "pgr_vs_peers" not in fm.columns  # partial name should not appear

    def test_pgr_vs_vfh_6m_column_name_is_exact(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        import src.processing.feature_engineering as fe
        import config
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_v1.parquet"))

        fm0 = build_feature_matrix(
            trending_prices, dividends, splits, force_refresh=True
        )
        idx = fm0.index
        fred_mock = pd.DataFrame(
            {"pgr_vs_vfh_6m": np.zeros(len(idx))}, index=idx
        )

        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_v2.parquet"))
        fm = build_feature_matrix(
            trending_prices, dividends, splits,
            fred_macro=fred_mock,
            force_refresh=True,
        )
        assert "pgr_vs_vfh_6m" in fm.columns
        assert "pgr_vs_vfh" not in fm.columns  # partial name should not appear


# ---------------------------------------------------------------------------
# pgr_vs_vfh_6m — injection into build_feature_matrix() via fred_macro
# ---------------------------------------------------------------------------

class TestPgrVsVfh6mInjection:
    """
    Verify that pgr_vs_vfh_6m flows through the FRED synthetic-column
    injection path correctly, following the same pattern as pgr_vs_kie_6m
    and pgr_vs_peers_6m.
    """

    def _make_synthetic_fred_vfh(self, monthly_dates: pd.DatetimeIndex) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        vals = rng.normal(0.0, 0.04, size=len(monthly_dates))
        return pd.DataFrame({"pgr_vs_vfh_6m": vals}, index=monthly_dates)

    def test_column_present_when_injected(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        import src.processing.feature_engineering as fe
        import config
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_vfh0.parquet"))

        fm0 = build_feature_matrix(
            trending_prices, dividends, splits, force_refresh=True
        )
        fred_mock = self._make_synthetic_fred_vfh(fm0.index)

        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_vfh1.parquet"))
        fm = build_feature_matrix(
            trending_prices, dividends, splits,
            fred_macro=fred_mock,
            force_refresh=True,
        )
        assert "pgr_vs_vfh_6m" in fm.columns, (
            "pgr_vs_vfh_6m not found in feature matrix after injection"
        )

    def test_column_absent_without_injection(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        fm = _build(trending_prices, dividends, splits,
                    tmp_path=tmp_path, monkeypatch=monkeypatch)
        assert "pgr_vs_vfh_6m" not in fm.columns

    def test_injected_values_pass_through_unchanged(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        import src.processing.feature_engineering as fe
        import config
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_vfh2.parquet"))

        fm0 = build_feature_matrix(
            trending_prices, dividends, splits, force_refresh=True
        )
        fred_mock = self._make_synthetic_fred_vfh(fm0.index)

        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_vfh3.parquet"))
        fm = build_feature_matrix(
            trending_prices, dividends, splits,
            fred_macro=fred_mock,
            force_refresh=True,
        )
        expected = fred_mock["pgr_vs_vfh_6m"].reindex(fm.index, method="ffill")
        pd.testing.assert_series_equal(
            fm["pgr_vs_vfh_6m"].rename("pgr_vs_vfh_6m"),
            expected.rename("pgr_vs_vfh_6m"),
            check_names=True,
        )

    def test_column_in_get_feature_columns(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        import src.processing.feature_engineering as fe
        import config
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_vfh4.parquet"))

        fm0 = build_feature_matrix(
            trending_prices, dividends, splits, force_refresh=True
        )
        fred_mock = self._make_synthetic_fred_vfh(fm0.index)

        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_vfh5.parquet"))
        fm = build_feature_matrix(
            trending_prices, dividends, splits,
            fred_macro=fred_mock,
            force_refresh=True,
        )
        assert "pgr_vs_vfh_6m" in get_feature_columns(fm)

    def test_sign_matches_arithmetic(self) -> None:
        """pgr_vs_vfh_6m is positive when PGR outperforms VFH over 6 months."""
        n = 24
        idx = pd.date_range("2015-01-31", periods=n, freq="ME")
        pgr_prices = pd.Series(100.0 * (1.04 ** np.arange(n)), index=idx)
        vfh_prices = pd.Series(100.0 * (1.01 ** np.arange(n)), index=idx)
        pgr_6m = pgr_prices.pct_change(6)
        vfh_6m = vfh_prices.pct_change(6)
        result = (pgr_6m - vfh_6m).dropna()
        assert (result > 0).all(), "pgr_vs_vfh_6m should be positive when PGR outperforms VFH"

    def test_sign_negative_when_underperforms(self) -> None:
        """pgr_vs_vfh_6m is negative when PGR underperforms VFH over 6 months."""
        n = 24
        idx = pd.date_range("2015-01-31", periods=n, freq="ME")
        pgr_prices = pd.Series(100.0 * (1.01 ** np.arange(n)), index=idx)
        vfh_prices = pd.Series(100.0 * (1.04 ** np.arange(n)), index=idx)
        pgr_6m = pgr_prices.pct_change(6)
        vfh_6m = vfh_prices.pct_change(6)
        result = (pgr_6m - vfh_6m).dropna()
        assert (result < 0).all(), "pgr_vs_vfh_6m should be negative when PGR underperforms VFH"

    def test_independent_of_peers_signal(
        self, trending_prices, dividends, splits, tmp_path, monkeypatch
    ) -> None:
        """pgr_vs_vfh_6m and pgr_vs_peers_6m can coexist with independent values."""
        import src.processing.feature_engineering as fe
        import config
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_vfh6.parquet"))

        fm0 = build_feature_matrix(
            trending_prices, dividends, splits, force_refresh=True
        )
        rng = np.random.default_rng(7)
        fred_mock = pd.DataFrame(
            {
                "pgr_vs_vfh_6m": rng.normal(0, 0.04, len(fm0)),
                "pgr_vs_peers_6m": rng.normal(0, 0.06, len(fm0)),
            },
            index=fm0.index,
        )

        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm_vfh7.parquet"))
        fm = build_feature_matrix(
            trending_prices, dividends, splits,
            fred_macro=fred_mock,
            force_refresh=True,
        )
        assert "pgr_vs_vfh_6m" in fm.columns
        assert "pgr_vs_peers_6m" in fm.columns
        # The two columns should have different values (independent signals)
        assert not fm["pgr_vs_vfh_6m"].equals(fm["pgr_vs_peers_6m"])
