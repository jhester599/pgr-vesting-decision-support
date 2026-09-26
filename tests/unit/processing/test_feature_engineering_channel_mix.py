"""
v6.3 tests — channel-mix features in build_feature_matrix (P1.4).

Coverage:
  1. channel_mix_agency_pct column appears in feature matrix when data is present
  2. npw_growth_yoy column appears in feature matrix when data is present
  3. channel_mix_agency_pct values are forward-filled from pgr_monthly to monthly dates
  4. npw_growth_yoy values are forward-filled from pgr_monthly to monthly dates
  5. channel_mix_agency_pct is absent when pgr_monthly=None
  6. npw_growth_yoy is absent when pgr_monthly=None
  7. channel_mix_agency_pct dropped when fewer than WFO_MIN_GAINSHARE_OBS non-NaN rows
  8. npw_growth_yoy dropped when fewer than WFO_MIN_GAINSHARE_OBS non-NaN rows
  9. channel_mix_agency_pct absent when column is all-NaN in pgr_monthly
 10. npw_growth_yoy absent when column is all-NaN in pgr_monthly
 11. No look-ahead: channel_mix features never use data from t+1 (lag applied first)
 12. Both features present alongside existing Gainshare features
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
    channel_mix: float = 0.45,
    npw_growth: float = 0.12,
    include_channel_mix: bool = True,
    include_npw_growth: bool = True,
    all_nan_channel_mix: bool = False,
    all_nan_npw_growth: bool = False,
) -> pd.DataFrame:
    """Build a synthetic pgr_monthly DataFrame with v6.3 channel-mix columns."""
    idx = pd.date_range("2008-01-31", periods=n_months, freq="ME")
    data: dict = {
        "combined_ratio": [85.0 + i * 0.05 for i in range(n_months)],
        "pif_total": [30000.0 + i * 10 for i in range(n_months)],
        "pif_growth_yoy": [0.08] * n_months,
        "gainshare_estimate": [1.2] * n_months,
    }
    if include_channel_mix:
        if all_nan_channel_mix:
            data["channel_mix_agency_pct"] = [float("nan")] * n_months
        else:
            data["channel_mix_agency_pct"] = [channel_mix] * n_months
    if include_npw_growth:
        if all_nan_npw_growth:
            data["npw_growth_yoy"] = [float("nan")] * n_months
        else:
            data["npw_growth_yoy"] = [npw_growth] * n_months
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# 1. channel_mix_agency_pct present when data is available
# ---------------------------------------------------------------------------

class TestChannelMixPresent:
    def test_channel_mix_in_feature_matrix(self, tmp_path, monkeypatch):
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 12)

        prices = _make_prices()
        pgr_monthly = _make_pgr_monthly()
        df = build_feature_matrix(
            prices, _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )
        assert "channel_mix_agency_pct" in df.columns, (
            "channel_mix_agency_pct should appear in feature matrix when pgr_monthly has the column"
        )

    def test_npw_growth_yoy_in_feature_matrix(self, tmp_path, monkeypatch):
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 12)

        prices = _make_prices()
        pgr_monthly = _make_pgr_monthly()
        df = build_feature_matrix(
            prices, _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )
        assert "npw_growth_yoy" in df.columns, (
            "npw_growth_yoy should appear in feature matrix when pgr_monthly has the column"
        )


# ---------------------------------------------------------------------------
# 2–4. Values are correctly forward-filled from pgr_monthly
# ---------------------------------------------------------------------------

class TestChannelMixValues:
    def test_channel_mix_value_matches_pgr_monthly(self, tmp_path, monkeypatch):
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 12)

        prices = _make_prices()
        pgr_monthly = _make_pgr_monthly(channel_mix=0.43)
        df = build_feature_matrix(
            prices, _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )
        # After burn-in, channel_mix_agency_pct should be ~0.43
        non_nan = df["channel_mix_agency_pct"].dropna()
        assert len(non_nan) > 0
        assert non_nan.iloc[-1] == pytest.approx(0.43, abs=1e-9)

    def test_npw_growth_yoy_value_matches_pgr_monthly(self, tmp_path, monkeypatch):
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 12)

        prices = _make_prices()
        pgr_monthly = _make_pgr_monthly(npw_growth=0.15)
        df = build_feature_matrix(
            prices, _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )
        non_nan = df["npw_growth_yoy"].dropna()
        assert len(non_nan) > 0
        assert non_nan.iloc[-1] == pytest.approx(0.15, abs=1e-9)

    def test_channel_mix_does_not_use_future_pgr_monthly(self, tmp_path, monkeypatch):
        """channel_mix_agency_pct must never reference data from t+1 or later."""
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 12)

        prices = _make_prices()
        # Create pgr_monthly with a distinct value change at a specific date
        n = 72
        idx = pd.date_range("2008-01-31", periods=n, freq="ME")
        channel_mix_vals = [0.40] * 36 + [0.60] * 36  # step up at month 37
        pgr_monthly = pd.DataFrame({
            "combined_ratio": [85.0] * n,
            "pif_total": [30000.0] * n,
            "pif_growth_yoy": [0.08] * n,
            "gainshare_estimate": [1.2] * n,
            "channel_mix_agency_pct": channel_mix_vals,
            "npw_growth_yoy": [0.12] * n,
        }, index=idx)

        df = build_feature_matrix(
            prices, _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )

        # The feature matrix must not have the 0.60 value on or before the step date.
        # The step happens at month 37 (idx[36]); feature matrix rows before that date
        # should show 0.40, not 0.60.
        step_date = idx[36]
        rows_before_step = df.loc[df.index < step_date, "channel_mix_agency_pct"]
        rows_with_0_60 = rows_before_step[rows_before_step >= 0.55]
        assert len(rows_with_0_60) == 0, (
            f"channel_mix_agency_pct shows future value (0.60) before step date {step_date}"
        )


# ---------------------------------------------------------------------------
# 5–6. Features absent when pgr_monthly=None
# ---------------------------------------------------------------------------

class TestChannelMixAbsent:
    def test_channel_mix_absent_when_no_pgr_monthly(self, tmp_path, monkeypatch):
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))

        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            pgr_monthly=None, force_refresh=True
        )
        assert "channel_mix_agency_pct" not in df.columns
        assert "npw_growth_yoy" not in df.columns

    def test_channel_mix_absent_when_column_missing_from_pgr_monthly(
        self, tmp_path, monkeypatch
    ):
        """pgr_monthly without the v6.3 columns → features absent (backward compat)."""
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 12)

        # Old-style pgr_monthly without v6.3 columns
        pgr_monthly = _make_pgr_monthly(
            include_channel_mix=False,
            include_npw_growth=False,
        )
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )
        assert "channel_mix_agency_pct" not in df.columns
        assert "npw_growth_yoy" not in df.columns


# ---------------------------------------------------------------------------
# 7–8. Dropped when insufficient observations
# ---------------------------------------------------------------------------

class TestChannelMixDropped:
    def test_channel_mix_dropped_when_below_min_obs(self, tmp_path, monkeypatch):
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 200)  # impossible threshold

        pgr_monthly = _make_pgr_monthly(n_months=72)
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )
        assert "channel_mix_agency_pct" not in df.columns
        assert "npw_growth_yoy" not in df.columns

    def test_channel_mix_present_when_above_min_obs(self, tmp_path, monkeypatch):
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 10)  # easily met

        pgr_monthly = _make_pgr_monthly(n_months=72)
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )
        assert "channel_mix_agency_pct" in df.columns
        assert "npw_growth_yoy" in df.columns


# ---------------------------------------------------------------------------
# 9–10. Features absent when column is all-NaN in pgr_monthly
# ---------------------------------------------------------------------------

class TestChannelMixAllNaN:
    def test_channel_mix_dropped_when_all_nan(self, tmp_path, monkeypatch):
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 12)

        pgr_monthly = _make_pgr_monthly(all_nan_channel_mix=True, all_nan_npw_growth=True)
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )
        assert "channel_mix_agency_pct" not in df.columns
        assert "npw_growth_yoy" not in df.columns


# ---------------------------------------------------------------------------
# 11. Both features present alongside existing Gainshare features
# ---------------------------------------------------------------------------

class TestChannelMixCoexistence:
    def test_channel_mix_and_gainshare_both_present(self, tmp_path, monkeypatch):
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 12)

        pgr_monthly = _make_pgr_monthly()
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )
        for col in ("combined_ratio_ttm", "pif_growth_yoy", "gainshare_est",
                    "channel_mix_agency_pct", "npw_growth_yoy"):
            assert col in df.columns, f"Expected '{col}' in feature matrix"

    def test_feature_columns_excludes_target(self, tmp_path, monkeypatch):
        """get_feature_columns should not return target_6m_return."""
        from src.processing.feature_engineering import get_feature_columns
        import config
        import src.processing.feature_engineering as fe
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
        monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "fm.parquet"))
        monkeypatch.setattr(config, "WFO_MIN_GAINSHARE_OBS", 12)

        pgr_monthly = _make_pgr_monthly()
        df = build_feature_matrix(
            _make_prices(), _make_dividends(), _make_splits(),
            pgr_monthly=pgr_monthly, force_refresh=True
        )
        feat_cols = get_feature_columns(df)
        assert "target_6m_return" not in feat_cols
        assert "channel_mix_agency_pct" in feat_cols
        assert "npw_growth_yoy" in feat_cols
