"""
v6.4 tests — P2.x features in build_feature_matrix.

Coverage:
  P2.2 — Underwriting income features
   1. underwriting_income present when data is available
   2. underwriting_income_3m is a 3-month rolling mean of underwriting_income
   3. underwriting_income_growth_yoy is a 12M pct_change
   4. All three absent when underwriting_income column is missing from pgr_monthly
   5. All three absent when underwriting_income column is all-NaN
  P2.3 — Unearned premium pipeline
   6. unearned_premium_growth_yoy present when data is available
   7. unearned_premium_to_npw_ratio math correctness
   8. unearned_premium_to_npw_ratio absent when npw is zero (guard)
   9. Both absent when columns are missing
  P2.4 — ROE trend
  10. roe_net_income_ttm present when data is available
  11. roe_trend = current ROE − trailing 12M mean
  12. Both absent when roe_net_income_ttm column is missing
  P2.1 — Investment portfolio
  13. investment_income_growth_yoy present when data is available
  14. investment_book_yield present when data is available
  15. Both absent when columns are missing
  P2.5 — Share repurchase signal
  16. buyback_acceleration present when shares_repurchased and avg_cost_per_share are available
  17. buyback_acceleration > 1 when current month is above trailing mean
  18. buyback_yield present when equity and bvps are also available
  19. buyback_acceleration absent when column is missing from pgr_monthly
  Global
  20. All P2.x features absent when pgr_monthly=None
  21. All P2.x features dropped when below WFO_MIN_GAINSHARE_OBS threshold
  22. All P2.x features coexist with v6.3 channel-mix features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.feature_engineering import build_feature_matrix


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_prices(n_years: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2008-01-01", periods=n_years * 252)
    close = np.linspace(50.0, 200.0, len(dates))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1_000_000},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _make_dividends() -> pd.DataFrame:
    div_dates = pd.bdate_range("2008-01-07", periods=24, freq="QS")
    return pd.DataFrame(
        {"dividend": 0.10},
        index=pd.DatetimeIndex(div_dates, name="date"),
    )


def _make_splits() -> pd.DataFrame:
    return pd.DataFrame(
        {"numerator": [], "denominator": [], "split_ratio": []},
        index=pd.DatetimeIndex([], name="date"),
    )


def _make_pgr_monthly(
    n_months: int = 72,
    include_uw_income: bool = True,
    include_unearned: bool = True,
    include_roe_ttm: bool = True,
    include_inv_income: bool = True,
    include_inv_book_yield: bool = True,
    include_buyback: bool = True,
    all_nan_uw_income: bool = False,
    uw_income_base: float = 500.0,
    unearned_premiums_base: float = 8_000.0,
    npw_base: float = 15_000.0,
    roe_base: float = 0.18,
    inv_income_base: float = 300.0,
    inv_book_yield_base: float = 0.035,
    shares_repurchased_base: float = 1.0,   # millions of shares/month
    avg_cost_base: float = 100.0,           # $/share
    shareholders_equity_base: float = 10_000.0,  # $M
    bvps_base: float = 50.0,               # $/share
) -> pd.DataFrame:
    """Build a synthetic pgr_monthly with all v6.4 columns."""
    idx = pd.date_range("2008-01-31", periods=n_months, freq="ME")
    data: dict = {
        # Core columns always present
        "combined_ratio": [90.0] * n_months,
        "pif_total": [30_000.0 + i * 10 for i in range(n_months)],
        "pif_growth_yoy": [0.08] * n_months,
        "gainshare_estimate": [1.2] * n_months,
        "channel_mix_agency_pct": [0.45] * n_months,
        "npw_growth_yoy": [0.12] * n_months,
    }

    # P2.2 — underwriting income
    if include_uw_income:
        if all_nan_uw_income:
            data["underwriting_income"] = [float("nan")] * n_months
        else:
            data["underwriting_income"] = [uw_income_base] * n_months

    # P2.3 — unearned premiums
    if include_unearned:
        data["unearned_premiums"] = [unearned_premiums_base] * n_months
        data["net_premiums_written"] = [npw_base] * n_months
        # pre-computed YoY growth (constant → 0.0 after 12 months)
        data["unearned_premium_growth_yoy"] = (
            [float("nan")] * 12 + [0.0] * (n_months - 12)
        )

    # P2.4 — ROE TTM
    if include_roe_ttm:
        data["roe_net_income_ttm"] = [roe_base] * n_months

    # P2.1 — investment portfolio
    if include_inv_income:
        data["investment_income"] = [inv_income_base] * n_months
    if include_inv_book_yield:
        data["investment_book_yield"] = [inv_book_yield_base] * n_months

    # P2.5 — buyback
    if include_buyback:
        data["shares_repurchased"] = [shares_repurchased_base] * n_months
        data["avg_cost_per_share"] = [avg_cost_base] * n_months
        data["shareholders_equity"] = [shareholders_equity_base] * n_months
        data["book_value_per_share"] = [bvps_base] * n_months

    return pd.DataFrame(data, index=idx)


def _build(tmp_path, monkeypatch, pgr_monthly, *, min_obs: int = 12):
    """Helper: monkeypatch config and paths, then call build_feature_matrix."""
    import config
    import src.processing.feature_engineering as fe
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
    monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", min_obs)
    return build_feature_matrix(
        _make_prices(), _make_dividends(), _make_splits(),
        pgr_monthly=pgr_monthly, force_refresh=True,
    )


# ===========================================================================
# P2.2 — Underwriting income
# ===========================================================================

class TestUnderwritingIncome:
    def test_underwriting_income_present(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly())
        assert "underwriting_income" in df.columns

    def test_underwriting_income_3m_is_rolling_mean(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly(uw_income_base=500.0))
        non_nan = df[["underwriting_income", "underwriting_income_3m"]].dropna()
        assert len(non_nan) > 0
        # 3m rolling mean of a constant must equal the constant
        assert np.allclose(non_nan["underwriting_income_3m"].values, 500.0, atol=1e-6)

    def test_underwriting_income_growth_yoy_zero_for_constant(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly(uw_income_base=500.0))
        non_nan = df["underwriting_income_growth_yoy"].dropna()
        assert len(non_nan) > 0
        # pct_change of a constant series = 0.0
        assert np.allclose(non_nan.values, 0.0, atol=1e-9)

    def test_underwriting_income_absent_when_column_missing(self, tmp_path, monkeypatch):
        pgr = _make_pgr_monthly(include_uw_income=False)
        df = _build(tmp_path, monkeypatch, pgr)
        assert "underwriting_income" not in df.columns
        assert "underwriting_income_3m" not in df.columns
        assert "underwriting_income_growth_yoy" not in df.columns

    def test_underwriting_income_absent_when_all_nan(self, tmp_path, monkeypatch):
        pgr = _make_pgr_monthly(all_nan_uw_income=True)
        df = _build(tmp_path, monkeypatch, pgr)
        assert "underwriting_income" not in df.columns
        assert "underwriting_income_3m" not in df.columns
        assert "underwriting_income_growth_yoy" not in df.columns

    def test_underwriting_income_growth_yoy_non_zero_when_changing(
        self, tmp_path, monkeypatch
    ):
        """If underwriting income doubles at month 37, yoy peak = 1.0 (100%)."""
        n = 72
        idx = pd.date_range("2008-01-31", periods=n, freq="ME")
        uw_vals = [500.0] * 36 + [1000.0] * 36
        data = {
            "combined_ratio": [90.0] * n,
            "pif_total": [30_000.0] * n,
            "pif_growth_yoy": [0.08] * n,
            "gainshare_estimate": [1.2] * n,
            "underwriting_income": uw_vals,
        }
        pgr = pd.DataFrame(data, index=idx)
        df = _build(tmp_path, monkeypatch, pgr)
        # pct_change(12) first hits 1.0 at the 12th month after the step (month 48).
        # By the last row both numerator and denominator are 1000 → growth = 0.
        # Check that the maximum observed growth across all rows is ≈ 1.0.
        max_growth = df["underwriting_income_growth_yoy"].dropna().max()
        assert max_growth == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# P2.3 — Unearned premium pipeline
# ===========================================================================

class TestUnearnedPremium:
    def test_unearned_premium_growth_yoy_present(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly())
        assert "unearned_premium_growth_yoy" in df.columns

    def test_unearned_premium_to_npw_ratio_math(self, tmp_path, monkeypatch):
        # ratio = 8000 / 15000 ≈ 0.5333…
        df = _build(
            tmp_path, monkeypatch,
            _make_pgr_monthly(unearned_premiums_base=8_000.0, npw_base=15_000.0),
        )
        non_nan = df["unearned_premium_to_npw_ratio"].dropna()
        assert len(non_nan) > 0
        assert non_nan.iloc[-1] == pytest.approx(8_000.0 / 15_000.0, abs=1e-9)

    def test_unearned_premium_to_npw_ratio_nan_when_npw_zero(
        self, tmp_path, monkeypatch
    ):
        """Zero NPW → ratio should be NaN (no division by zero)."""
        n = 72
        idx = pd.date_range("2008-01-31", periods=n, freq="ME")
        data = {
            "combined_ratio": [90.0] * n,
            "pif_total": [30_000.0] * n,
            "pif_growth_yoy": [0.08] * n,
            "gainshare_estimate": [1.2] * n,
            "unearned_premiums": [8_000.0] * n,
            "net_premiums_written": [0.0] * n,  # all zero → denominator always invalid
            "unearned_premium_growth_yoy": [0.0] * n,
        }
        pgr = pd.DataFrame(data, index=idx)
        df = _build(tmp_path, monkeypatch, pgr)
        # Column may still appear but all values should be NaN
        if "unearned_premium_to_npw_ratio" in df.columns:
            assert df["unearned_premium_to_npw_ratio"].isna().all()

    def test_unearned_columns_absent_when_missing(self, tmp_path, monkeypatch):
        pgr = _make_pgr_monthly(include_unearned=False)
        df = _build(tmp_path, monkeypatch, pgr)
        assert "unearned_premium_growth_yoy" not in df.columns
        assert "unearned_premium_to_npw_ratio" not in df.columns


# ===========================================================================
# P2.4 — ROE trend
# ===========================================================================

class TestROETrend:
    def test_roe_net_income_ttm_present(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly())
        assert "roe_net_income_ttm" in df.columns

    def test_roe_trend_zero_for_constant_roe(self, tmp_path, monkeypatch):
        """Constant ROE → rolling mean = constant → trend = 0."""
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly(roe_base=0.18))
        non_nan = df["roe_trend"].dropna()
        assert len(non_nan) > 0
        assert np.allclose(non_nan.values, 0.0, atol=1e-9)

    def test_roe_trend_positive_when_roe_rising(self, tmp_path, monkeypatch):
        """Step up in ROE should produce positive roe_trend in the transition window."""
        n = 72
        idx = pd.date_range("2008-01-31", periods=n, freq="ME")
        roe_vals = [0.10] * 36 + [0.25] * 36
        data = {
            "combined_ratio": [90.0] * n,
            "pif_total": [30_000.0] * n,
            "pif_growth_yoy": [0.08] * n,
            "gainshare_estimate": [1.2] * n,
            "roe_net_income_ttm": roe_vals,
        }
        pgr = pd.DataFrame(data, index=idx)
        df = _build(tmp_path, monkeypatch, pgr)
        # In the 12 months immediately after the step, rolling mean still includes
        # some 0.10-period values, so trend (current − mean) must be > 0.
        # Once the trailing window is fully at 0.25 the trend decays to 0.
        # The maximum trend observed must be positive.
        trend_max = df["roe_trend"].dropna().max()
        assert trend_max > 0

    def test_roe_columns_absent_when_missing(self, tmp_path, monkeypatch):
        pgr = _make_pgr_monthly(include_roe_ttm=False)
        df = _build(tmp_path, monkeypatch, pgr)
        assert "roe_net_income_ttm" not in df.columns
        assert "roe_trend" not in df.columns


# ===========================================================================
# P2.1 — Investment portfolio features
# ===========================================================================

class TestInvestmentPortfolio:
    def test_investment_income_growth_yoy_present(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly())
        assert "investment_income_growth_yoy" in df.columns

    def test_investment_income_growth_yoy_zero_for_constant(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly(inv_income_base=300.0))
        non_nan = df["investment_income_growth_yoy"].dropna()
        assert len(non_nan) > 0
        assert np.allclose(non_nan.values, 0.0, atol=1e-9)

    def test_investment_book_yield_present(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly())
        assert "investment_book_yield" in df.columns

    def test_investment_book_yield_value(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly(inv_book_yield_base=0.042))
        non_nan = df["investment_book_yield"].dropna()
        assert len(non_nan) > 0
        assert non_nan.iloc[-1] == pytest.approx(0.042, abs=1e-9)

    def test_investment_features_absent_when_missing(self, tmp_path, monkeypatch):
        pgr = _make_pgr_monthly(include_inv_income=False, include_inv_book_yield=False)
        df = _build(tmp_path, monkeypatch, pgr)
        assert "investment_income_growth_yoy" not in df.columns
        assert "investment_book_yield" not in df.columns


# ===========================================================================
# P2.5 — Share repurchase signal
# ===========================================================================

class TestBuybackSignal:
    def test_buyback_acceleration_present(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly())
        assert "buyback_acceleration" in df.columns

    def test_buyback_acceleration_equals_one_for_constant(self, tmp_path, monkeypatch):
        """Constant buyback amount → rolling mean = same → acceleration = 1.0."""
        df = _build(
            tmp_path, monkeypatch,
            _make_pgr_monthly(shares_repurchased_base=1.0, avg_cost_base=100.0),
        )
        non_nan = df["buyback_acceleration"].dropna()
        assert len(non_nan) > 0
        assert np.allclose(non_nan.values, 1.0, atol=1e-9)

    def test_buyback_acceleration_above_one_when_accelerating(
        self, tmp_path, monkeypatch
    ):
        """If buyback triples in second half, max acceleration > 1 in transition."""
        n = 72
        idx = pd.date_range("2008-01-31", periods=n, freq="ME")
        shares = [1.0] * 36 + [3.0] * 36
        data = {
            "combined_ratio": [90.0] * n,
            "pif_total": [30_000.0] * n,
            "pif_growth_yoy": [0.08] * n,
            "gainshare_estimate": [1.2] * n,
            "shares_repurchased": shares,
            "avg_cost_per_share": [100.0] * n,
            "shareholders_equity": [10_000.0] * n,
            "book_value_per_share": [50.0] * n,
        }
        pgr = pd.DataFrame(data, index=idx)
        df = _build(tmp_path, monkeypatch, pgr)
        # In the transition window (months 37–48), the rolling mean still mixes
        # 1× and 3× months so acceleration > 1.  After 12 months at 3× it
        # decays back to 1.  The maximum observed must be > 1.
        accel_max = df["buyback_acceleration"].dropna().max()
        assert accel_max > 1.0

    def test_buyback_yield_present(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly())
        assert "buyback_yield" in df.columns

    def test_buyback_yield_positive_when_repurchasing(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly())
        non_nan = df["buyback_yield"].dropna()
        assert len(non_nan) > 0
        assert (non_nan > 0).all()

    def test_buyback_acceleration_absent_when_columns_missing(
        self, tmp_path, monkeypatch
    ):
        pgr = _make_pgr_monthly(include_buyback=False)
        df = _build(tmp_path, monkeypatch, pgr)
        assert "buyback_acceleration" not in df.columns
        assert "buyback_yield" not in df.columns


# ===========================================================================
# Global / cross-cutting tests
# ===========================================================================

class TestP2xGlobal:
    _P2X_COLS = [
        "underwriting_income", "underwriting_income_3m",
        "underwriting_income_growth_yoy",
        "unearned_premium_growth_yoy", "unearned_premium_to_npw_ratio",
        "roe_net_income_ttm", "roe_trend",
        "investment_income_growth_yoy", "investment_book_yield",
        "buyback_yield", "buyback_acceleration",
    ]

    def test_all_p2x_absent_when_pgr_monthly_none(self, tmp_path, monkeypatch):
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            pgr_monthly=None, force_refresh=True,
        )
        for col in self._P2X_COLS:
            assert col not in df.columns, f"{col} should be absent when pgr_monthly=None"

    def test_all_p2x_dropped_below_min_obs(self, tmp_path, monkeypatch):
        # Impossible threshold → all optional columns dropped
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly(), min_obs=9999)
        for col in self._P2X_COLS:
            assert col not in df.columns, (
                f"{col} should be dropped when below WFO_MIN_GAINSHARE_OBS"
            )

    def test_p2x_coexists_with_v63_channel_mix(self, tmp_path, monkeypatch):
        df = _build(tmp_path, monkeypatch, _make_pgr_monthly())
        expected = [
            "channel_mix_agency_pct", "npw_growth_yoy",  # v6.3
            "underwriting_income", "roe_net_income_ttm",  # v6.4
            "investment_book_yield", "buyback_acceleration",
        ]
        for col in expected:
            assert col in df.columns, f"Expected '{col}' in feature matrix"
