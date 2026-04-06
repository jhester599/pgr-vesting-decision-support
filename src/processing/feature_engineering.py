"""
Rolling monthly feature matrix construction.

Computes all predictive features on a month-end frequency, expanding the
effective dataset from ~30 semi-annual observations to 180+ monthly samples.
This directly addresses the "Large P, Small N" constraint identified in
the Gemini peer review.

No feature uses data from t+1 or later (strict no-leakage guarantee).
The target variable (6-month forward DRIP total return) is appended as the
final column, with NaN for the last 6 months of history.

Feature groups:
  Price-derived (always available):
    - mom_3m, mom_6m, mom_12m   (price momentum)
    - vol_21d, vol_63d           (realized volatility)

  Technical indicators (from Alpha Vantage cache):
    - sma_12m, rsi_14, macd_hist, bb_pct_b

  Fundamental (from FMP; ~5-year depth on free tier):
    - pe_ratio, pb_ratio, roe

  Optional — Gainshare / PGR-specific (from EDGAR cache):
    - combined_ratio_ttm, pif_growth_yoy, gainshare_est
    Dropped entirely if fewer than WFO_MIN_GAINSHARE_OBS non-NaN rows.

  PGR channel-mix features (v6.3, from expanded pgr_edgar_monthly schema):
    - channel_mix_agency_pct  (npw_agency / total_personal_lines_npw;
                               leading indicator of margin compression)
    - npw_growth_yoy          (companywide NPW 12-month YoY growth rate)
    Dropped silently if fewer than WFO_MIN_GAINSHARE_OBS non-NaN rows.

  PGR P2.x features (v6.4, from pgr_edgar_monthly 8-K monthly supplements):
    Underwriting income (P2.2):
    - underwriting_income         (net_premiums_earned × (1 − CR/100); DB pre-computed)
    - underwriting_income_3m      (3-month trailing average; smooths volatility)
    - underwriting_income_growth_yoy (12M YoY growth rate; captures margin inflection)
    Unearned premium pipeline (P2.3):
    - unearned_premium_growth_yoy  (DB pre-computed; leads earned premium by ~6M)
    - unearned_premium_to_npw_ratio (unearned / NPW; rising = accelerating new business)
    ROE trend (P2.4):
    - roe_net_income_ttm           (TTM ROE from 8-K supplement)
    - roe_trend                    (current ROE − 12M rolling mean; sign = direction)
    Investment portfolio (P2.1):
    - investment_income_growth_yoy (12M YoY growth; rate-environment proxy)
    - investment_book_yield        (fixed-income book yield; rate-sensitivity signal)
    Share repurchase signal (P2.5):
    - buyback_yield                (annualised buyback spend / estimated market cap)
    - buyback_acceleration         (current buyback / trailing 12M mean; >1 = accelerating)
    All dropped silently if fewer than WFO_MIN_GAINSHARE_OBS non-NaN rows.

  Price-derived (v6.0):
    - high_52w              (current price / 52-week high; George & Hwang 2004)

  FRED macro features (v3.0+, from fred_macro_monthly table):
    - yield_slope           (T10Y2Y — 10Y-2Y spread)
    - yield_curvature       (2×GS5 − GS2 − GS10)
    - real_rate_10y         (GS10 − T10YIE)
    - credit_spread_ig      (BAA10Y — investment-grade credit spread)
    - credit_spread_hy      (BAMLH0A0HYM2 — high-yield OAS)
    - nfci                  (Chicago Fed NFCI composite)
    - vix                   (VIXCLS — CBOE VIX for regime classification)
    PGR-specific FRED features (v3.1+):
    - insurance_cpi_mom3m   (3-month momentum of motor vehicle insurance CPI)
    - vmt_yoy               (year-over-year change in vehicle miles traveled)
    Dropped silently if FRED data is absent (backward-compatible with pre-v3.0 runs).

  Synthetic FRED-injected features (v6.0, computed from DB prices):
    - pgr_vs_peers_6m       (PGR 6M DRIP return minus equal-weight peer composite
                             return; peers = ALL, TRV, CB, HIG)
    - pgr_vs_vfh_6m         (PGR 6M return minus VFH broad-financials ETF 6M return;
                             wider lens than KIE — includes banks, diversified financials)

"""

import os
import sqlite3
import warnings
import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


_PROCESSED_PATH = os.path.join(config.DATA_PROCESSED_DIR, "feature_matrix.parquet")


# ---------------------------------------------------------------------------
# v4.1 — Publication lag helpers (authoritative enforcement point)
# ---------------------------------------------------------------------------

def _apply_fred_lags(fred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift each FRED series by its configured publication lag.

    This prevents look-ahead bias from FRED data revisions. The DB stores
    the latest-vintage values fetched on the bootstrap date; this function
    applies point-in-time publication lags so that month-T features only
    use data that was publicly available at time T.

    Args:
        fred_df: DataFrame indexed by month-end dates, columns are FRED series IDs.

    Returns:
        DataFrame with each series shifted by its lag from config.FRED_SERIES_LAGS.
    """
    result = fred_df.copy()
    for sid in result.columns:
        lag = config.FRED_SERIES_LAGS.get(sid, config.FRED_DEFAULT_LAG_MONTHS)
        if lag > 0:
            result[sid] = result[sid].shift(lag)
    return result


def _snap_to_business_month_end_index(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize any timestamp-like index to the last business day of month."""
    dt_index = pd.DatetimeIndex(pd.to_datetime(index))
    calendar_month_end = dt_index + pd.offsets.MonthEnd(0)
    snapped = [pd.offsets.BMonthEnd().rollback(ts) for ts in calendar_month_end]
    return pd.DatetimeIndex(snapped, name=dt_index.name)


def _resample_last_business_month_end(
    values: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Resample a daily/irregular series to last-business-day month-end."""
    result = values.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    return result.resample("BME").last()


def _apply_edgar_lag(edgar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift EDGAR fundamental data by the filing lag to prevent look-ahead bias.

    PGR's 10-Q for Q4 (period ending Dec 31) is filed ~late February.
    Using report_period as the index means Q4 data would appear in January
    features — 2 months before it was publicly available.

    Args:
        edgar_df: DataFrame indexed by month-end dates (report period end dates).

    Returns:
        DataFrame shifted forward by config.EDGAR_FILING_LAG_MONTHS months.
    """
    lag = config.EDGAR_FILING_LAG_MONTHS
    if lag <= 0:
        return edgar_df
    result = edgar_df.shift(lag, freq="MS")
    # Snap back to business month-end after the MonthStart shift
    result.index = _snap_to_business_month_end_index(result.index)
    return result


# Nominal trading days for each momentum lookback
_MOMENTUM_WINDOWS: dict[str, int] = {
    "mom_3m": 63,
    "mom_6m": 126,
    "mom_12m": 252,
}

_VOL_WINDOWS: dict[str, int] = {
    "vol_21d": 21,
    "vol_63d": 63,
}


def _normalize_pgr_monthly_columns(pgr_monthly: pd.DataFrame) -> pd.DataFrame:
    """Apply compatibility renames for known EDGAR monthly schema variants."""
    result = pgr_monthly.copy()
    if (
        "roe_net_income_ttm" not in result.columns
        and "roe_net_income_trailing_12m" in result.columns
    ):
        result = result.rename(
            columns={"roe_net_income_trailing_12m": "roe_net_income_ttm"}
        )
    return result


def _warn_if_all_null(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    """Warn when an expected feature group is present structurally but empty."""
    existing = [col for col in columns if col in df.columns]
    if existing and df[existing].isna().all().all():
        warnings.warn(
            f"{label} feature group is present but all-null: {existing}",
            RuntimeWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    price_history: pd.DataFrame,
    dividend_history: pd.DataFrame,
    split_history: pd.DataFrame,
    technical_indicators: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    pgr_monthly: pd.DataFrame | None = None,
    fred_macro: pd.DataFrame | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build the full monthly feature matrix for ML training.

    Args:
        price_history:       DataFrame from price_loader.load().
        dividend_history:    DataFrame from dividend_loader.load().
        split_history:       DataFrame from split_loader.load().
        technical_indicators: DataFrame from technical_loader.load() (optional).
        fundamentals:        DataFrame from fundamentals_loader.load() (optional).
        pgr_monthly:         DataFrame from pgr_monthly_loader.load() (optional).
        fred_macro:          Wide DataFrame from db_client.get_fred_macro() with
                             DatetimeIndex (month-end) and one column per FRED
                             series_id.  Optional; omitted in pre-v3.0 runs.
        force_refresh:       If True, recompute even if cached Parquet exists.

    Returns:
        DataFrame indexed by month-end date with feature columns and
        ``target_6m_return`` as the final column.
    """
    # Resample daily prices to month-end (last business day)
    monthly_close = _resample_last_business_month_end(price_history["close"])
    monthly_dates = monthly_close.index

    df = pd.DataFrame(index=monthly_dates)
    df.index.name = "date"

    # ------------------------------------------------------------------
    # Price momentum features (no-leakage: only data up to t used at t)
    # ------------------------------------------------------------------
    # shift(n) on the daily series moves each value n trading days into the
    # future, so daily_close.shift(n)[t] == close[t - n trading days].
    # Reindexing to month-end dates with ffill aligns the lagged price to
    # the last available trading day on or before each month-end.
    daily_close = price_history["close"]

    for col, n_days in _MOMENTUM_WINDOWS.items():
        lagged_daily = daily_close.shift(n_days)
        lagged_monthly = lagged_daily.reindex(monthly_dates, method="ffill")
        df[col] = (monthly_close / lagged_monthly) - 1.0

    # ------------------------------------------------------------------
    # Realized volatility
    # ------------------------------------------------------------------
    daily_log_ret = np.log(daily_close / daily_close.shift(1))

    for col, n_days in _VOL_WINDOWS.items():
        rolling_vol = (
            daily_log_ret.rolling(window=n_days, min_periods=n_days // 2)
            .std()
            * np.sqrt(252)
        )
        df[col] = rolling_vol.reindex(monthly_dates, method="ffill")

    # ------------------------------------------------------------------
    # v6.0 — 52-week high ratio (George & Hwang 2004)
    # ------------------------------------------------------------------
    # Ratio of the current month-end close to the rolling 252-trading-day
    # high (inclusive of the current day).  A value near 1.0 means the
    # stock is trading at the top of its 52-week range; anchoring theory
    # predicts that investors are more reluctant to push prices through
    # round-number / 52-week-high ceilings, creating a predictable
    # momentum signal.  min_periods=126 requires at least 6 months of
    # daily history before publishing a non-NaN value.
    rolling_52w_high = daily_close.rolling(window=252, min_periods=126).max()
    df["high_52w"] = (
        monthly_close / rolling_52w_high.reindex(monthly_dates, method="ffill")
    )

    # ------------------------------------------------------------------
    # Technical indicators (Alpha Vantage)
    # ------------------------------------------------------------------
    if technical_indicators is not None and not technical_indicators.empty:
        for col in ["sma_12m", "rsi_14", "macd_hist", "bb_pct_b"]:
            if col in technical_indicators.columns:
                df[col] = technical_indicators[col].reindex(
                    monthly_dates, method="ffill"
                )

    # ------------------------------------------------------------------
    # Fundamental features (FMP, ~5-year depth on free tier)
    # ------------------------------------------------------------------
    if fundamentals is not None and not fundamentals.empty:
        for col in ["pe_ratio", "pb_ratio", "roe"]:
            if col in fundamentals.columns:
                # Forward-fill quarterly data to monthly frequency
                df[col] = fundamentals[col].reindex(monthly_dates, method="ffill")

    # ------------------------------------------------------------------
    # PGR-specific Gainshare features (optional, from EDGAR cache)
    # ------------------------------------------------------------------
    if pgr_monthly is not None and not pgr_monthly.empty:
        pgr_monthly = _normalize_pgr_monthly_columns(pgr_monthly)
        # Trailing 12-month combined ratio (rolling mean of monthly values)
        if "combined_ratio" in pgr_monthly.columns:
            cr_monthly = pgr_monthly["combined_ratio"].reindex(
                monthly_dates, method="ffill"
            )
            df["combined_ratio_ttm"] = cr_monthly.rolling(12, min_periods=6).mean()
            # monthly_combined_ratio_delta: 3M average relative to PGR's
            # long-standing 96 combined-ratio target. Positive = worse than
            # target; negative = better than target.
            df["monthly_combined_ratio_delta"] = (
                cr_monthly.rolling(3, min_periods=2).mean() - 96.0
            )

        if "pif_growth_yoy" in pgr_monthly.columns:
            df["pif_growth_yoy"] = pgr_monthly["pif_growth_yoy"].reindex(
                monthly_dates, method="ffill"
            )

        if "gainshare_estimate" in pgr_monthly.columns:
            df["gainshare_est"] = pgr_monthly["gainshare_estimate"].reindex(
                monthly_dates, method="ffill"
            )

        # cr_acceleration: 3-period second difference of combined_ratio_ttm.
        # Captures the rate-of-change in underwriting margin deterioration or
        # improvement.  A positive cr_acceleration means the combined ratio is
        # worsening faster (bearish); negative means it is improving faster (bullish).
        # Requires combined_ratio_ttm to be already computed above.
        if "combined_ratio_ttm" in df.columns:
            df["cr_acceleration"] = df["combined_ratio_ttm"].diff(3)

        # pif_growth_acceleration: 3M annualized policy growth minus 12M
        # trailing growth. Positive = unit growth is accelerating.
        if (
            "pif_total" in pgr_monthly.columns
            and not pgr_monthly["pif_total"].isna().all()
        ):
            pif_total = pgr_monthly["pif_total"].reindex(monthly_dates, method="ffill")
            valid_pif_total = pif_total.where(pif_total > 0)
            pif_3m_annualized = (valid_pif_total / valid_pif_total.shift(3)) ** 4 - 1.0
            pif_12m_yoy = (valid_pif_total / valid_pif_total.shift(12)) - 1.0
            df["pif_growth_acceleration"] = pif_3m_annualized - pif_12m_yoy

        # ------------------------------------------------------------------
        # v6.3 — Channel-mix features (P1.4)
        # These are pre-computed and stored in pgr_edgar_monthly by the CSV
        # loader (scripts/edgar_8k_fetcher.py --load-from-csv) and by the
        # live EDGAR fetch once P2.6 updates the HTML parser.
        # ------------------------------------------------------------------

        # channel_mix_agency_pct: agency NPW / (agency + direct NPW).
        # Increasing agency share signals traditional-channel reliance, often
        # associated with higher acquisition costs; declining share (direct
        # gaining) signals improved unit economics and PGR's competitive
        # advantage in direct-to-consumer distribution.
        if (
            "channel_mix_agency_pct" in pgr_monthly.columns
            and not pgr_monthly["channel_mix_agency_pct"].isna().all()
        ):
            df["channel_mix_agency_pct"] = pgr_monthly[
                "channel_mix_agency_pct"
            ].reindex(monthly_dates, method="ffill")

        # npw_growth_yoy: companywide net premiums written YoY % growth.
        # Strong NPW growth (> 10%) signals rate adequacy and market share
        # gains; weak or negative growth signals competitive pressure or
        # underwriting tightening.  Pre-computed 12M pct_change stored in DB.
        if (
            "npw_growth_yoy" in pgr_monthly.columns
            and not pgr_monthly["npw_growth_yoy"].isna().all()
        ):
            df["npw_growth_yoy"] = pgr_monthly["npw_growth_yoy"].reindex(
                monthly_dates, method="ffill"
            )

        # npw_per_pif_yoy: average premium per policy, a pricing-power
        # decomposition that separates rate actions from unit growth.
        if (
            "net_premiums_written" in pgr_monthly.columns
            and "pif_total" in pgr_monthly.columns
            and not pgr_monthly["net_premiums_written"].isna().all()
            and not pgr_monthly["pif_total"].isna().all()
        ):
            npw_m = pgr_monthly["net_premiums_written"].reindex(
                monthly_dates, method="ffill"
            )
            pif_total = pgr_monthly["pif_total"].reindex(
                monthly_dates, method="ffill"
            )
            valid_pif_total = pif_total.where(pif_total > 0)
            npw_per_pif = npw_m / valid_pif_total
            df["npw_per_pif_yoy"] = npw_per_pif.pct_change(
                periods=12,
                fill_method=None,
            )

        # npw_vs_npe_spread_pct: written-vs-earned premium spread. Positive
        # = premium pipeline still building faster than earnings recognition.
        if (
            "net_premiums_written" in pgr_monthly.columns
            and "net_premiums_earned" in pgr_monthly.columns
            and not pgr_monthly["net_premiums_written"].isna().all()
            and not pgr_monthly["net_premiums_earned"].isna().all()
        ):
            npw_m = pgr_monthly["net_premiums_written"].reindex(
                monthly_dates, method="ffill"
            )
            npe_m = pgr_monthly["net_premiums_earned"].reindex(
                monthly_dates, method="ffill"
            )
            valid_npe_m = npe_m.where(npe_m.abs() > 1e-12)
            df["npw_vs_npe_spread_pct"] = (npw_m - npe_m) / valid_npe_m

        # ------------------------------------------------------------------
        # v6.4 — P2.x features: Underwriting income, unearned premiums,
        #         ROE trend, investment portfolio, share repurchase signal
        # All source from pgr_monthly (PGR 8-K monthly supplements).
        # ------------------------------------------------------------------

        # --- P2.2: Underwriting income ---
        # Pre-computed in DB: net_premiums_earned × (1 − combined_ratio / 100).
        # More direct profitability signal than combined_ratio alone because
        # it captures scale — a flat CR on growing premiums = growing margin $$.
        if (
            "underwriting_income" in pgr_monthly.columns
            and not pgr_monthly["underwriting_income"].isna().all()
        ):
            uw_monthly = pgr_monthly["underwriting_income"].reindex(
                monthly_dates, method="ffill"
            )
            df["underwriting_income"] = uw_monthly
            # 3-month trailing average: smooths single-weather-event spikes
            df["underwriting_income_3m"] = uw_monthly.rolling(3, min_periods=2).mean()
            # YoY growth rate: positive slope = margin expansion trend
            df["underwriting_income_growth_yoy"] = uw_monthly.pct_change(
                periods=12,
                fill_method=None,
            )

        # underwriting_margin_ttm: underwriting profit over trailing-12-month
        # earned premium. More stationary than nominal underwriting income.
        if (
            "underwriting_income" in pgr_monthly.columns
            and "net_premiums_earned" in pgr_monthly.columns
            and not pgr_monthly["underwriting_income"].isna().all()
            and not pgr_monthly["net_premiums_earned"].isna().all()
        ):
            uw_monthly = pgr_monthly["underwriting_income"].reindex(
                monthly_dates, method="ffill"
            )
            npe_monthly = pgr_monthly["net_premiums_earned"].reindex(
                monthly_dates, method="ffill"
            )
            uw_ttm = uw_monthly.rolling(12, min_periods=6).sum()
            npe_ttm = npe_monthly.rolling(12, min_periods=6).sum()
            valid_npe_ttm = npe_ttm.where(npe_ttm.abs() > 1e-12)
            df["underwriting_margin_ttm"] = uw_ttm / valid_npe_ttm

        # --- P2.3: Unearned premium reserve growth (earned-premium pipeline) ---
        # unearned_premium_growth_yoy is pre-computed in DB (12M pct_change).
        # Rising unearned premiums lead earned premium growth by ~6 months.
        if (
            "unearned_premium_growth_yoy" in pgr_monthly.columns
            and not pgr_monthly["unearned_premium_growth_yoy"].isna().all()
        ):
            df["unearned_premium_growth_yoy"] = pgr_monthly[
                "unearned_premium_growth_yoy"
            ].reindex(monthly_dates, method="ffill")

        # unearned_premium_to_npw_ratio: unearned reserve / written premium.
        # Ratio > 0.5 and rising signals accelerating new-business inflow
        # faster than recognition — a forward earnings quality indicator.
        if (
            "unearned_premiums" in pgr_monthly.columns
            and "net_premiums_written" in pgr_monthly.columns
            and not pgr_monthly["unearned_premiums"].isna().all()
            and not pgr_monthly["net_premiums_written"].isna().all()
        ):
            unprem = pgr_monthly["unearned_premiums"].reindex(
                monthly_dates, method="ffill"
            )
            npw_m = pgr_monthly["net_premiums_written"].reindex(
                monthly_dates, method="ffill"
            )
            valid_npw_m = npw_m.where(npw_m > 0)
            df["unearned_premium_to_npw_ratio"] = unprem / valid_npw_m

        # --- P2.4: ROE trend (capital-efficiency momentum) ---
        # roe_net_income_ttm from 8-K supplement: monthly frequency, lag applied.
        # Superior to quarterly XBRL roe for the same reason eps_basic is
        # preferred over quarterly EPS — 4× more observations.
        if (
            "roe_net_income_ttm" in pgr_monthly.columns
            and not pgr_monthly["roe_net_income_ttm"].isna().all()
        ):
            roe_monthly = pgr_monthly["roe_net_income_ttm"].reindex(
                monthly_dates, method="ffill"
            )
            df["roe_net_income_ttm"] = roe_monthly
            # roe_trend: current ROE − trailing 12M mean.
            # Positive = improving efficiency vs. recent history (bullish);
            # negative = deteriorating capital returns (bearish signal).
            df["roe_trend"] = roe_monthly - roe_monthly.rolling(12, min_periods=6).mean()

        # book_value_per_share_growth_yoy: insurance-native balance-sheet
        # compounding signal preferred over more generic valuation anchors.
        if (
            "book_value_per_share" in pgr_monthly.columns
            and not pgr_monthly["book_value_per_share"].isna().all()
        ):
            bvps_monthly = pgr_monthly["book_value_per_share"].reindex(
                monthly_dates, method="ffill"
            )
            df["book_value_per_share_growth_yoy"] = bvps_monthly.pct_change(
                periods=12,
                fill_method=None,
            )

        # pgr_premium_to_surplus: trailing written premium over equity, an
        # insurer-specific operating-leverage / capital-strain signal.
        if (
            "net_premiums_written" in pgr_monthly.columns
            and "shareholders_equity" in pgr_monthly.columns
            and not pgr_monthly["net_premiums_written"].isna().all()
            and not pgr_monthly["shareholders_equity"].isna().all()
        ):
            npw_monthly = pgr_monthly["net_premiums_written"].reindex(
                monthly_dates, method="ffill"
            )
            equity_monthly = pgr_monthly["shareholders_equity"].reindex(
                monthly_dates, method="ffill"
            )
            npw_ttm = npw_monthly.rolling(12, min_periods=6).sum()
            valid_equity = equity_monthly.where(equity_monthly.abs() > 1e-12)
            df["pgr_premium_to_surplus"] = npw_ttm / valid_equity

        # reserve_to_npe_ratio: reserves scaled by trailing earned premium.
        if (
            "loss_lae_reserves" in pgr_monthly.columns
            and "net_premiums_earned" in pgr_monthly.columns
            and not pgr_monthly["loss_lae_reserves"].isna().all()
            and not pgr_monthly["net_premiums_earned"].isna().all()
        ):
            reserves_monthly = pgr_monthly["loss_lae_reserves"].reindex(
                monthly_dates, method="ffill"
            )
            npe_monthly = pgr_monthly["net_premiums_earned"].reindex(
                monthly_dates, method="ffill"
            )
            npe_ttm = npe_monthly.rolling(12, min_periods=6).sum()
            valid_npe_ttm = npe_ttm.where(npe_ttm.abs() > 1e-12)
            df["reserve_to_npe_ratio"] = reserves_monthly / valid_npe_ttm

        # direct_channel_pif_share_ttm and channel_mix_direct_pct_yoy capture
        # the structural and changing share of the direct auto channel.
        if (
            "pif_direct_auto" in pgr_monthly.columns
            and "pif_total_personal_lines" in pgr_monthly.columns
            and not pgr_monthly["pif_direct_auto"].isna().all()
            and not pgr_monthly["pif_total_personal_lines"].isna().all()
        ):
            direct_pif = pgr_monthly["pif_direct_auto"].reindex(
                monthly_dates, method="ffill"
            )
            personal_lines_pif = pgr_monthly["pif_total_personal_lines"].reindex(
                monthly_dates, method="ffill"
            )
            valid_personal_lines_pif = personal_lines_pif.where(personal_lines_pif > 0)
            direct_share = direct_pif / valid_personal_lines_pif
            df["direct_channel_pif_share_ttm"] = direct_share.rolling(
                12, min_periods=6
            ).mean()
            df["channel_mix_direct_pct_yoy"] = direct_share - direct_share.shift(12)

        # realized_gain_to_net_income_ratio: earnings-quality check on how much
        # of reported net income is coming from realized gains.
        if (
            "total_net_realized_gains" in pgr_monthly.columns
            and "net_income" in pgr_monthly.columns
            and not pgr_monthly["total_net_realized_gains"].isna().all()
            and not pgr_monthly["net_income"].isna().all()
        ):
            realized_gains = pgr_monthly["total_net_realized_gains"].reindex(
                monthly_dates, method="ffill"
            )
            net_income_monthly = pgr_monthly["net_income"].reindex(
                monthly_dates, method="ffill"
            )
            realized_gains_ttm = realized_gains.rolling(12, min_periods=6).sum()
            net_income_ttm = net_income_monthly.rolling(12, min_periods=6).sum()
            valid_net_income_ttm = net_income_ttm.where(net_income_ttm.abs() > 1e-12)
            df["realized_gain_to_net_income_ratio"] = (
                realized_gains_ttm / valid_net_income_ttm
            )

        # unrealized_gain_pct_equity: OCI-style capital buffer relative to equity.
        if (
            "net_unrealized_gains_fixed" in pgr_monthly.columns
            and "shareholders_equity" in pgr_monthly.columns
            and not pgr_monthly["net_unrealized_gains_fixed"].isna().all()
            and not pgr_monthly["shareholders_equity"].isna().all()
        ):
            unrealized_fixed = pgr_monthly["net_unrealized_gains_fixed"].reindex(
                monthly_dates, method="ffill"
            )
            equity_monthly = pgr_monthly["shareholders_equity"].reindex(
                monthly_dates, method="ffill"
            )
            valid_equity = equity_monthly.where(equity_monthly.abs() > 1e-12)
            df["unrealized_gain_pct_equity"] = unrealized_fixed / valid_equity

        # loss_ratio_ttm and expense_ratio_ttm decompose underwriting quality.
        if (
            "losses_lae" in pgr_monthly.columns
            and "net_premiums_earned" in pgr_monthly.columns
            and not pgr_monthly["losses_lae"].isna().all()
            and not pgr_monthly["net_premiums_earned"].isna().all()
        ):
            losses_monthly = pgr_monthly["losses_lae"].reindex(
                monthly_dates, method="ffill"
            )
            npe_monthly = pgr_monthly["net_premiums_earned"].reindex(
                monthly_dates, method="ffill"
            )
            losses_ttm = losses_monthly.rolling(12, min_periods=6).sum()
            npe_ttm = npe_monthly.rolling(12, min_periods=6).sum()
            valid_npe_ttm = npe_ttm.where(npe_ttm.abs() > 1e-12)
            df["loss_ratio_ttm"] = 100.0 * losses_ttm / valid_npe_ttm
        elif (
            "loss_lae_ratio" in pgr_monthly.columns
            and not pgr_monthly["loss_lae_ratio"].isna().all()
        ):
            loss_ratio_monthly = pgr_monthly["loss_lae_ratio"].reindex(
                monthly_dates, method="ffill"
            )
            df["loss_ratio_ttm"] = loss_ratio_monthly.rolling(12, min_periods=6).mean()

        if (
            "policy_acquisition_costs" in pgr_monthly.columns
            and "other_underwriting_expenses" in pgr_monthly.columns
            and "net_premiums_earned" in pgr_monthly.columns
            and not pgr_monthly["policy_acquisition_costs"].isna().all()
            and not pgr_monthly["other_underwriting_expenses"].isna().all()
            and not pgr_monthly["net_premiums_earned"].isna().all()
        ):
            acq_monthly = pgr_monthly["policy_acquisition_costs"].reindex(
                monthly_dates, method="ffill"
            )
            other_uw_monthly = pgr_monthly["other_underwriting_expenses"].reindex(
                monthly_dates, method="ffill"
            )
            npe_monthly = pgr_monthly["net_premiums_earned"].reindex(
                monthly_dates, method="ffill"
            )
            expense_ttm = (acq_monthly + other_uw_monthly).rolling(12, min_periods=6).sum()
            npe_ttm = npe_monthly.rolling(12, min_periods=6).sum()
            valid_npe_ttm = npe_ttm.where(npe_ttm.abs() > 1e-12)
            df["expense_ratio_ttm"] = 100.0 * expense_ttm / valid_npe_ttm
        elif (
            "expense_ratio" in pgr_monthly.columns
            and not pgr_monthly["expense_ratio"].isna().all()
        ):
            expense_ratio_monthly = pgr_monthly["expense_ratio"].reindex(
                monthly_dates, method="ffill"
            )
            df["expense_ratio_ttm"] = expense_ratio_monthly.rolling(
                12, min_periods=6
            ).mean()

        # --- P2.1: Investment portfolio features ---
        # investment_income_growth_yoy: YoY growth in net investment income.
        # Rising growth signals either rate-environment uplift (reinvesting
        # maturing bonds at higher yields) or a growing invested asset base.
        if (
            "investment_income" in pgr_monthly.columns
            and not pgr_monthly["investment_income"].isna().all()
        ):
            inv_inc = pgr_monthly["investment_income"].reindex(
                monthly_dates, method="ffill"
            )
            df["investment_income_growth_yoy"] = inv_inc.pct_change(
                periods=12,
                fill_method=None,
            )

        # investment_book_yield: current yield on the fixed-income portfolio.
        # Rising book yield = portfolio rolling into higher-coupon bonds,
        # boosting recurring income; a direct complement to yield_slope.
        if (
            "investment_book_yield" in pgr_monthly.columns
            and not pgr_monthly["investment_book_yield"].isna().all()
        ):
            df["investment_book_yield"] = pgr_monthly["investment_book_yield"].reindex(
                monthly_dates, method="ffill"
            )

        # duration_rate_shock_3m: fixed-income duration times the 3M change
        # in the 10Y Treasury yield. Captures insurer bond-book sensitivity.
        if (
            "fixed_income_duration" in pgr_monthly.columns
            and not pgr_monthly["fixed_income_duration"].isna().all()
        ):
            duration_monthly = pgr_monthly["fixed_income_duration"].reindex(
                monthly_dates, method="ffill"
            )
            if fred_macro is not None and "GS10" in _apply_fred_lags(fred_macro).columns:
                fred_aligned_local = _apply_fred_lags(fred_macro).reindex(
                    monthly_dates, method="ffill"
                )
                rate_change_3m = fred_aligned_local["GS10"].diff(3)
                df["duration_rate_shock_3m"] = duration_monthly * rate_change_3m

        # --- P2.5: Share repurchase signal ---
        # Buyback amount = shares_repurchased × avg_cost_per_share ($ value).
        # Management accelerates buybacks when they believe the stock is
        # undervalued; this is a high-quality insider-confidence signal.
        if (
            "shares_repurchased" in pgr_monthly.columns
            and "avg_cost_per_share" in pgr_monthly.columns
            and not pgr_monthly["shares_repurchased"].isna().all()
        ):
            bb_shares = pgr_monthly["shares_repurchased"].reindex(
                monthly_dates, method="ffill"
            )
            bb_cost = pgr_monthly["avg_cost_per_share"].reindex(
                monthly_dates, method="ffill"
            )
            buyback_amount = (bb_shares * bb_cost).clip(lower=0)

            # buyback_yield: annualised buyback spend as % of estimated market cap.
            # Market cap estimated as: shares_outstanding × monthly_price, where
            # shares_outstanding ≈ shareholders_equity / book_value_per_share.
            # This avoids requiring a separate shares-outstanding data feed.
            if (
                "shareholders_equity" in pgr_monthly.columns
                and "book_value_per_share" in pgr_monthly.columns
                and not pgr_monthly["shareholders_equity"].isna().all()
                and not pgr_monthly["book_value_per_share"].isna().all()
            ):
                equity_m = pgr_monthly["shareholders_equity"].reindex(
                    monthly_dates, method="ffill"
                )
                bvps_m = pgr_monthly["book_value_per_share"].reindex(
                    monthly_dates, method="ffill"
                )
                valid_bvps_m = bvps_m.where(bvps_m > 0)
                # shares_outstanding proxy (in millions if equity is in $M)
                shares_est = equity_m / valid_bvps_m
                mkt_cap_est = shares_est * monthly_close
                valid_mktcap = mkt_cap_est.where(mkt_cap_est > 0)
                # ×12 to annualise monthly spend
                df["buyback_yield"] = (buyback_amount * 12) / valid_mktcap

            # buyback_acceleration: current buyback vs trailing 12M mean.
            # Values >1 mean this month's repurchase pace is above the prior
            # year's average — a signal of accelerating management confidence.
            rolling_bb_mean = buyback_amount.rolling(12, min_periods=4).mean()
            valid_bb_mean = rolling_bb_mean.where(rolling_bb_mean > 0)
            df["buyback_acceleration"] = buyback_amount / valid_bb_mean

    # Drop optional Gainshare/channel-mix/P2.x columns if insufficient observations.
    # cr_acceleration is a derivative of combined_ratio_ttm (3-period diff),
    # so it is naturally gated: if combined_ratio_ttm is dropped here due to
    # sparse data, cr_acceleration becomes all-NaN and the WFO engine drops
    # it via its own feature-selection logic.  No separate threshold needed.
    for col in [
        "combined_ratio_ttm", "pif_growth_yoy", "gainshare_est",
        "monthly_combined_ratio_delta", "pif_growth_acceleration",
        "channel_mix_agency_pct", "npw_growth_yoy",
        "npw_per_pif_yoy", "npw_vs_npe_spread_pct",
        # v6.4 P2.x features
        "underwriting_income", "underwriting_income_3m",
        "underwriting_income_growth_yoy", "underwriting_margin_ttm",
        "unearned_premium_growth_yoy", "unearned_premium_to_npw_ratio",
        "roe_net_income_ttm", "roe_trend", "book_value_per_share_growth_yoy",
        "pgr_premium_to_surplus", "reserve_to_npe_ratio",
        "direct_channel_pif_share_ttm", "channel_mix_direct_pct_yoy",
        "realized_gain_to_net_income_ratio", "unrealized_gain_pct_equity",
        "loss_ratio_ttm", "expense_ratio_ttm",
        "investment_income_growth_yoy", "investment_book_yield",
        "duration_rate_shock_3m",
        "buyback_yield", "buyback_acceleration",
    ]:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            if n_valid < config.WFO_MIN_GAINSHARE_OBS:
                df = df.drop(columns=[col])

    # ------------------------------------------------------------------
    # FRED macro features (v3.0+) — derived from the fred_macro DataFrame.
    # Each feature is forward-filled from the FRED month-end observations
    # to align with the feature matrix dates.  If a required FRED series
    # is absent, the corresponding feature column is skipped silently so
    # that pre-v3.0 runs remain unaffected.
    # ------------------------------------------------------------------
    if fred_macro is not None and not fred_macro.empty:
        # Reindex to the feature matrix dates; forward-fill end-of-month gaps
        fred_aligned = fred_macro.reindex(monthly_dates, method="ffill")

        # yield_slope: 10Y-2Y spread (direct series)
        if "T10Y2Y" in fred_aligned.columns:
            df["yield_slope"] = fred_aligned["T10Y2Y"]

        # yield_curvature: butterfly = 2×GS5 − GS2 − GS10
        if all(s in fred_aligned.columns for s in ("GS5", "GS2", "GS10")):
            df["yield_curvature"] = (
                2.0 * fred_aligned["GS5"]
                - fred_aligned["GS2"]
                - fred_aligned["GS10"]
            )

        # real_rate_10y: nominal 10Y minus breakeven inflation
        if all(s in fred_aligned.columns for s in ("GS10", "T10YIE")):
            df["real_rate_10y"] = fred_aligned["GS10"] - fred_aligned["T10YIE"]
            df["real_yield_change_6m"] = df["real_rate_10y"].diff(6)
            df["breakeven_inflation_10y"] = fred_aligned["T10YIE"]
            df["breakeven_momentum_3m"] = fred_aligned["T10YIE"].diff(3)

        # credit_spread_ig: Moody's Baa minus 10Y Treasury
        if "BAA10Y" in fred_aligned.columns:
            df["credit_spread_ig"] = fred_aligned["BAA10Y"]
            df["baa10y_spread"] = fred_aligned["BAA10Y"]

        # credit_spread_hy: ICE BofA HY OAS
        if "BAMLH0A0HYM2" in fred_aligned.columns:
            df["credit_spread_hy"] = fred_aligned["BAMLH0A0HYM2"]

        # nfci: Chicago Fed National Financial Conditions Index
        if "NFCI" in fred_aligned.columns:
            df["nfci"] = fred_aligned["NFCI"]

        # vix: CBOE Volatility Index (for regime classification in v3.1+)
        if "VIXCLS" in fred_aligned.columns:
            df["vix"] = fred_aligned["VIXCLS"]

        # ------------------------------------------------------------------
        # v3.1 PGR-specific FRED features
        # CUSR0000SETC01 = Motor vehicle insurance CPI (rate adequacy proxy)
        # TRFVOLUSM227NFWA = Vehicle miles traveled NSA (claims frequency proxy)
        # ------------------------------------------------------------------

        # insurance_cpi_mom3m: 3-month momentum of motor vehicle insurance CPI
        if "CUSR0000SETC01" in fred_aligned.columns:
            ins_cpi = fred_aligned["CUSR0000SETC01"]
            df["insurance_cpi_mom3m"] = ins_cpi.pct_change(
                periods=3,
                fill_method=None,
            )

        # vmt_yoy: year-over-year % change in vehicle miles traveled
        if "TRFVOLUSM227NFWA" in fred_aligned.columns:
            vmt = fred_aligned["TRFVOLUSM227NFWA"]
            df["vmt_yoy"] = vmt.pct_change(periods=12, fill_method=None)

        # Research-only public macro series loaded into fred_macro_monthly by v19.
        if "DTWEXBGS" in fred_aligned.columns:
            usd_index = fred_aligned["DTWEXBGS"]
            df["usd_broad_return_3m"] = usd_index.pct_change(3, fill_method=None)
            df["usd_momentum_6m"] = usd_index.pct_change(6, fill_method=None)

        if "DCOILWTICO" in fred_aligned.columns:
            wti = fred_aligned["DCOILWTICO"].replace(0, np.nan)
            df["wti_return_3m"] = wti.pct_change(3, fill_method=None)

        if all(s in fred_aligned.columns for s in ("MORTGAGE30US", "GS10")):
            df["mortgage_spread_30y_10y"] = fred_aligned["MORTGAGE30US"] - fred_aligned["GS10"]

        if "THREEFYTP10" in fred_aligned.columns:
            df["term_premium_10y"] = fred_aligned["THREEFYTP10"]

        if all(s in fred_aligned.columns for s in ("WPU45110101", "PPIACO")):
            legal_ppi_yoy = fred_aligned["WPU45110101"].pct_change(12, fill_method=None)
            broad_ppi_yoy = fred_aligned["PPIACO"].pct_change(12, fill_method=None)
            df["legal_services_ppi_relative"] = legal_ppi_yoy - broad_ppi_yoy

        if "MRTSSM447USN" in fred_aligned.columns:
            gas_sales = fred_aligned["MRTSSM447USN"].replace(0, np.nan)
            df["gasoline_retail_sales_delta"] = gas_sales.pct_change(12, fill_method=None)

        # ------------------------------------------------------------------
        # v4.5 PGR-specific severity and pricing features
        # ------------------------------------------------------------------

        # used_car_cpi_yoy: YoY % change in used car & truck CPI (CUSR0000SETA02).
        # Rising used car prices drive higher total-loss settlement costs, a direct
        # headwind to PGR's combined ratio; the 2021–22 spike was a major headwind.
        if "CUSR0000SETA02" in fred_aligned.columns:
            df["used_car_cpi_yoy"] = fred_aligned["CUSR0000SETA02"].pct_change(
                12,
                fill_method=None,
            )

        # medical_cpi_yoy: YoY % change in medical care CPI (CUSR0000SAM2).
        # Bodily injury and PIP claim severity tracks medical inflation directly.
        if "CUSR0000SAM2" in fred_aligned.columns:
            df["medical_cpi_yoy"] = fred_aligned["CUSR0000SAM2"].pct_change(
                12,
                fill_method=None,
            )

        if {"used_car_cpi_yoy", "medical_cpi_yoy"}.issubset(df.columns):
            df["severity_index_yoy"] = df[
                ["used_car_cpi_yoy", "medical_cpi_yoy"]
            ].mean(axis=1)

        # ppi_auto_ins_yoy: YoY % change in PPI for Private Passenger Auto Insurance
        # (PCU5241265241261).  Replaces the originally planned CUSR0000SETC01 (motor
        # vehicle insurance CPI) which is unavailable via FRED.  The PPI captures
        # cost-based pricing pressure upstream of the consumer CPI; rising PPI signals
        # that carriers have pricing power and are raising premiums.
        # Validated 2026-03-29: partial IC=0.353 (p<0.0001), hit-rate 76.1%.
        if "PCU5241265241261" in fred_aligned.columns:
            df["ppi_auto_ins_yoy"] = fred_aligned["PCU5241265241261"].pct_change(
                12,
                fill_method=None,
            )

        if "CUSR0000SETE" in fred_aligned.columns:
            df["motor_vehicle_ins_cpi_yoy"] = fred_aligned["CUSR0000SETE"].pct_change(
                12,
                fill_method=None,
            )

        if {"ppi_auto_ins_yoy", "severity_index_yoy"}.issubset(df.columns):
            df["rate_adequacy_gap_yoy"] = (
                df["ppi_auto_ins_yoy"] - df["severity_index_yoy"]
            )

        if {"ppi_auto_ins_yoy", "motor_vehicle_ins_cpi_yoy"}.issubset(df.columns):
            df["auto_pricing_power_spread"] = (
                df["ppi_auto_ins_yoy"] - df["motor_vehicle_ins_cpi_yoy"]
            )

        # pgr_vs_kie_6m: PGR trailing 6M return minus KIE trailing 6M return.
        # Captures PGR's idiosyncratic alpha vs. the broad insurance sector (KIE =
        # SPDR S&P Insurance ETF).  High recent relative strength signals continued
        # momentum or mean-reversion depending on macro context.
        # Pre-computed in build_feature_matrix_from_db() and injected as a synthetic
        # FRED column so it flows through the same lag-guarded pipeline.
        if "pgr_vs_kie_6m" in fred_aligned.columns:
            df["pgr_vs_kie_6m"] = fred_aligned["pgr_vs_kie_6m"]

        # pgr_vs_peers_6m: PGR trailing 6M return minus equal-weight peer composite
        # (ALL, TRV, CB, HIG) 6M return.  Captures PGR's idiosyncratic alpha vs.
        # its direct insurance-line competitors — a tighter peer group than KIE
        # (which includes diversified financials and non-P&C lines).
        # Pre-computed in build_feature_matrix_from_db() and injected as a synthetic
        # FRED column so it flows through the same lag-guarded pipeline.
        if "pgr_vs_peers_6m" in fred_aligned.columns:
            df["pgr_vs_peers_6m"] = fred_aligned["pgr_vs_peers_6m"]

        # pgr_vs_vfh_6m: PGR trailing 6M return minus VFH (Vanguard Financials ETF)
        # 6M return.  KIE benchmarks PGR against the pure insurance sub-sector;
        # VFH broadens the lens to all US financials (banks, insurance, diversified).
        # The spread captures whether PGR is gaining or losing vs. the wider financial
        # sector, independent of pure insurance-sector dynamics.
        # Pre-computed in build_feature_matrix_from_db() and injected as a synthetic
        # FRED column so it flows through the same lag-guarded pipeline.
        if "pgr_vs_vfh_6m" in fred_aligned.columns:
            df["pgr_vs_vfh_6m"] = fred_aligned["pgr_vs_vfh_6m"]

        # v18.0 benchmark-side relative spreads from existing benchmark prices.
        if "vwo_vxus_spread_6m" in fred_aligned.columns:
            df["vwo_vxus_spread_6m"] = fred_aligned["vwo_vxus_spread_6m"]

        if "gold_vs_treasury_6m" in fred_aligned.columns:
            df["gold_vs_treasury_6m"] = fred_aligned["gold_vs_treasury_6m"]

        if "commodity_equity_momentum" in fred_aligned.columns:
            df["commodity_equity_momentum"] = fred_aligned["commodity_equity_momentum"]

        if {"credit_spread_hy", "baa10y_spread"}.issubset(df.columns):
            valid_ig = df["baa10y_spread"].where(df["baa10y_spread"].abs() > 1e-12)
            df["credit_spread_ratio"] = df["credit_spread_hy"] / valid_ig
            df["excess_bond_premium_proxy"] = df["credit_spread_hy"] - df["baa10y_spread"]

        if all(s in fred_aligned.columns for s in ("SP500_PE_RATIO_MULTPL",)):
            valid_market_pe = fred_aligned["SP500_PE_RATIO_MULTPL"].where(
                fred_aligned["SP500_PE_RATIO_MULTPL"] > 0
            )
            if "pe_ratio" in df.columns:
                valid_pe = df["pe_ratio"].where(df["pe_ratio"] > 0)
                df["pgr_pe_vs_market_pe"] = valid_pe / valid_market_pe

        if all(s in fred_aligned.columns for s in ("SP500_PRICE_TO_BOOK_MULTPL",)):
            valid_market_pb = fred_aligned["SP500_PRICE_TO_BOOK_MULTPL"].where(
                fred_aligned["SP500_PRICE_TO_BOOK_MULTPL"] > 0
            )
            if "pb_ratio" in df.columns:
                valid_pb = df["pb_ratio"].where(df["pb_ratio"] > 0)
                df["pgr_price_to_book_relative"] = valid_pb / valid_market_pb

        if all(s in fred_aligned.columns for s in ("SP500_EARNINGS_YIELD_MULTPL", "GS10")):
            df["equity_risk_premium"] = (
                fred_aligned["SP500_EARNINGS_YIELD_MULTPL"] - fred_aligned["GS10"]
            )

    if pgr_monthly is not None and not pgr_monthly.empty:
        _warn_if_all_null(
            df,
            ["roe_net_income_ttm", "roe_trend"],
            "EDGAR ROE",
        )
    if fred_macro is not None and not fred_macro.empty:
        _warn_if_all_null(
            df,
            [
                "yield_slope",
                "yield_curvature",
                "real_rate_10y",
                "credit_spread_ig",
                "credit_spread_hy",
                "nfci",
                "vix",
            ],
            "FRED macro",
        )

    # ------------------------------------------------------------------
    # Target variable: 6-month forward DRIP total return
    # ------------------------------------------------------------------
    from src.processing.total_return import build_monthly_returns
    target = build_monthly_returns(
        price_history, dividend_history, split_history, forward_months=6
    )
    df["target_6m_return"] = target.reindex(monthly_dates)

    # ------------------------------------------------------------------
    # Final cleanup
    # ------------------------------------------------------------------
    # Drop rows where ALL price-derived features are NaN (burn-in period)
    price_feature_cols = list(_MOMENTUM_WINDOWS.keys()) + list(_VOL_WINDOWS.keys())
    df = df.dropna(subset=price_feature_cols, how="all")

    # v4.3: drop redundant features to improve obs/feature ratio (~3.5:1 → ~4:1)
    for col in getattr(config, "FEATURES_TO_DROP", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Coerce all columns to float64.  EDGAR XBRL returns None for pe_ratio /
    # pb_ratio (not available via XBRL); reindexing these Series produces
    # object-dtype columns.  numpy ≥ 2.0 rejects object arrays in nanmedian,
    # so we normalise here — the authoritative enforcement point — rather than
    # patching every downstream consumer.
    df = df.apply(pd.to_numeric, errors="coerce")

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    df.to_parquet(_PROCESSED_PATH)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (excludes the target variable)."""
    return [c for c in df.columns if c != "target_6m_return"]


def get_model_feature_columns(
    df: pd.DataFrame,
    model_type: str | None = None,
) -> list[str]:
    """Return feature columns, applying any configured model-specific override."""
    feature_cols = get_feature_columns(df)
    if model_type is None:
        return feature_cols

    override_cols = getattr(config, "MODEL_FEATURE_OVERRIDES", {}).get(model_type)
    if not override_cols:
        return feature_cols

    selected = [col for col in override_cols if col in feature_cols]
    return selected or feature_cols


def build_feature_matrix_from_db(
    conn: sqlite3.Connection,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """DB-backed entry point for the PGR feature matrix.

    Loads PGR price, dividend, split, fundamental, and EDGAR data from the
    v2 SQLite database and delegates to the existing build_feature_matrix().
    The Parquet cache is always bypassed because the DB is the authoritative
    source for v2 runs.

    Args:
        conn:          Open SQLite connection with v2 schema.
        force_refresh: When True, forces recomputation even if a Parquet cache
                       file already exists (always True internally; parameter
                       preserved for API symmetry with build_feature_matrix).

    Returns:
        DataFrame identical in structure to build_feature_matrix() output:
        monthly DatetimeIndex, feature columns, and ``target_6m_return``.

    Raises:
        ValueError: If PGR has no price data in the database.
    """
    from src.database import db_client

    prices = db_client.get_prices(conn, "PGR")

    dividends_raw = db_client.get_dividends(conn, "PGR")
    if dividends_raw.empty:
        dividends = pd.DataFrame(
            columns=["dividend", "source"],
            index=pd.DatetimeIndex([], name="ex_date"),
        )
    else:
        dividends = dividends_raw.rename(columns={"amount": "dividend"})

    splits_raw = db_client.get_splits(conn, "PGR")
    if splits_raw.empty:
        splits = pd.DataFrame(
            columns=["split_ratio", "numerator", "denominator"],
            index=pd.DatetimeIndex([], name="split_date"),
        )
    else:
        splits = splits_raw

    fundamentals_raw = db_client.get_pgr_fundamentals(conn)

    # Load monthly EDGAR 8-K data first (needed for pb_ratio via BVPS below).
    edgar_raw = db_client.get_pgr_edgar_monthly(conn)
    if not edgar_raw.empty:
        # Apply filing lag to prevent EDGAR period-end vs filing date look-ahead bias (v4.1)
        edgar_raw = _apply_edgar_lag(edgar_raw)
    pgr_monthly = edgar_raw if not edgar_raw.empty else None

    # --- Derive pe_ratio, pb_ratio, and roe from EDGAR data (v6.x) ---
    # pe_ratio: monthly_price / TTM_EPS
    #   Source: pgr_edgar_monthly.eps_basic (monthly 8-K).  TTM EPS =
    #   rolling 12-month sum.  Lag already applied to edgar_raw above.
    #   Superior to quarterly XBRL: 256 monthly obs vs ~86 quarterly;
    #   no frequency interpolation needed; consistent source with pb_ratio.
    # pb_ratio: monthly_price / book_value_per_share
    #   Source: pgr_edgar_monthly.book_value_per_share (monthly 8-K).
    #   Lag already applied to edgar_raw above.
    # roe: from quarterly XBRL pgr_fundamentals_quarterly (EDGAR 10-Q/10-K).
    #   roe_net_income_trailing_12m in the 8-K CSV is a candidate to replace
    #   this in a future sprint once that column is added to the DB schema.
    fundamentals = None
    if not prices.empty:
        monthly_close_vals = _resample_last_business_month_end(prices["close"].copy())
        monthly_fundamentals = pd.DataFrame(index=monthly_close_vals.index)

        # pe_ratio from monthly EPS (8-K supplements, lag already applied)
        if (
            not edgar_raw.empty
            and "eps_basic" in edgar_raw.columns
            and not edgar_raw["eps_basic"].isna().all()
        ):
            eps = edgar_raw["eps_basic"].copy()
            eps.index = pd.to_datetime(eps.index)
            # TTM EPS: rolling 12-month sum of monthly EPS figures
            eps_ttm = eps.rolling(12, min_periods=12).sum()
            eps_aligned = eps_ttm.reindex(monthly_fundamentals.index, method="ffill")
            # Avoid division by zero or negative TTM EPS
            valid_eps = eps_aligned.where(eps_aligned > 0)
            monthly_fundamentals["pe_ratio"] = monthly_close_vals / valid_eps

        # roe from quarterly XBRL (forward-filled to monthly, EDGAR lag applied)
        if not fundamentals_raw.empty and "roe" in fundamentals_raw.columns:
            roe_q = fundamentals_raw["roe"].copy()
            roe_q.index = pd.to_datetime(roe_q.index)
            roe_monthly = _resample_last_business_month_end(roe_q).ffill()
            roe_monthly = roe_monthly.shift(config.EDGAR_FILING_LAG_MONTHS, freq="MS")
            roe_monthly.index = _snap_to_business_month_end_index(roe_monthly.index)
            monthly_fundamentals["roe"] = roe_monthly.reindex(
                monthly_fundamentals.index, method="ffill"
            )

        # pb_ratio from monthly BVPS (PGR 8-K supplements, lag already applied)
        if (
            not edgar_raw.empty
            and "book_value_per_share" in edgar_raw.columns
            and not edgar_raw["book_value_per_share"].isna().all()
        ):
            bvps = edgar_raw["book_value_per_share"].copy()
            bvps.index = pd.to_datetime(bvps.index)
            # Align monthly prices to BVPS index, then compute ratio
            price_aligned = monthly_close_vals.reindex(bvps.index, method="ffill")
            valid_bvps = bvps.where(bvps > 0)
            pb_series = (price_aligned / valid_bvps).rename("pb_ratio")
            monthly_fundamentals["pb_ratio"] = pb_series.reindex(
                monthly_fundamentals.index, method="ffill"
            )

        fundamentals = (
            monthly_fundamentals
            if not monthly_fundamentals.dropna(how="all").empty
            else None
        )

    # v3.0+: load FRED macro + v3.1/v4.5 PGR-specific series if the table is populated
    all_fred_series = (
        list(config.FRED_SERIES_MACRO)
        + list(config.FRED_SERIES_PGR)
        + list(getattr(config, "V19_PUBLIC_MACRO_SERIES", []))
    )
    fred_raw = db_client.get_fred_macro(conn, all_fred_series)
    if not fred_raw.empty:
        # Apply publication lags to prevent FRED revision look-ahead bias (v4.1)
        fred_raw = _apply_fred_lags(fred_raw)

    # v4.5: pgr_vs_kie_6m — PGR trailing 6M return minus KIE trailing 6M return.
    # Computed from DB prices and injected as a synthetic column so it passes through
    # the same lag-guarded FRED pipeline in build_feature_matrix().
    try:
        pgr_prices_raw = db_client.get_prices(conn, "PGR")
        kie_prices_raw = db_client.get_prices(conn, "KIE")
        if not pgr_prices_raw.empty and not kie_prices_raw.empty:
            def _monthly_close(price_df: pd.DataFrame) -> pd.Series:
                """Resample daily prices to month-end close."""
                close = price_df["close"].copy()
                close.index = pd.to_datetime(close.index)
                return _resample_last_business_month_end(close)

            pgr_m = _monthly_close(pgr_prices_raw)
            kie_m = _monthly_close(kie_prices_raw)
            pgr_6m = pgr_m.pct_change(6, fill_method=None)
            kie_6m = kie_m.pct_change(6, fill_method=None)
            pgr_vs_kie = (pgr_6m - kie_6m).rename("pgr_vs_kie_6m")
            if fred_raw.empty:
                fred_raw = pgr_vs_kie.to_frame()
            else:
                fred_raw = fred_raw.join(pgr_vs_kie, how="left")
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Could not build synthetic feature pgr_vs_kie_6m; continuing without it. Error=%r",
            exc,
        )

    # v6.0: pgr_vs_peers_6m — PGR trailing 6M return minus equal-weight composite
    # of the four direct P&C insurance peers (ALL, TRV, CB, HIG).
    # Requires peer prices bootstrapped via scripts/peer_fetch.py; silently absent
    # if the peer tables are not yet populated.
    try:
        peer_price_frames = [
            db_client.get_prices(conn, t) for t in config.PEER_TICKER_UNIVERSE
        ]
        available_peer_frames = [df for df in peer_price_frames if not df.empty]
        if available_peer_frames and not prices.empty:
            def _m_close(price_df: pd.DataFrame) -> pd.Series:
                """Resample daily prices to month-end close."""
                c = price_df["close"].copy()
                c.index = pd.to_datetime(c.index)
                return _resample_last_business_month_end(c)

            pgr_m_v60 = _m_close(prices)
            peer_monthly_df = pd.concat(
                [_m_close(df) for df in available_peer_frames], axis=1
            )
            peer_composite_6m = peer_monthly_df.pct_change(6, fill_method=None).mean(axis=1)
            pgr_6m_v60 = pgr_m_v60.pct_change(6, fill_method=None)
            pgr_vs_peers = (pgr_6m_v60 - peer_composite_6m).rename("pgr_vs_peers_6m")
            if fred_raw.empty:
                fred_raw = pgr_vs_peers.to_frame()
            else:
                fred_raw = fred_raw.join(pgr_vs_peers, how="left")
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Could not build synthetic feature pgr_vs_peers_6m; continuing without it. Error=%r",
            exc,
        )

    # v6.0: pgr_vs_vfh_6m — PGR trailing 6M return minus VFH (Vanguard Financials ETF)
    # 6M return.  VFH is fetched weekly as part of the standard ETF benchmark universe,
    # so no separate bootstrap is needed — data is always current.
    try:
        vfh_prices_raw = db_client.get_prices(conn, "VFH")
        if not vfh_prices_raw.empty and not prices.empty:
            def _mc_vfh(price_df: pd.DataFrame) -> pd.Series:
                """Resample daily prices to month-end close."""
                c = price_df["close"].copy()
                c.index = pd.to_datetime(c.index)
                return _resample_last_business_month_end(c)

            pgr_m_vfh = _mc_vfh(prices)
            vfh_m = _mc_vfh(vfh_prices_raw)
            pgr_6m_vfh = pgr_m_vfh.pct_change(6, fill_method=None)
            vfh_6m = vfh_m.pct_change(6, fill_method=None)
            pgr_vs_vfh = (pgr_6m_vfh - vfh_6m).rename("pgr_vs_vfh_6m")
            if fred_raw.empty:
                fred_raw = pgr_vs_vfh.to_frame()
            else:
                fred_raw = fred_raw.join(pgr_vs_vfh, how="left")
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Could not build synthetic feature pgr_vs_vfh_6m; continuing without it. Error=%r",
            exc,
        )

    # v18.0: benchmark-side relative features from existing benchmark prices.
    # These are meant to help explain reduced-universe directional bias without
    # requiring new paid data or expanding model complexity.
    try:
        vwo_prices_raw = db_client.get_prices(conn, "VWO")
        vxus_prices_raw = db_client.get_prices(conn, "VXUS")
        gld_prices_raw = db_client.get_prices(conn, "GLD")
        bnd_prices_raw = db_client.get_prices(conn, "BND")
        dbc_prices_raw = db_client.get_prices(conn, "DBC")
        voo_prices_raw = db_client.get_prices(conn, "VOO")

        def _monthly_close_generic(price_df: pd.DataFrame) -> pd.Series:
            close = price_df["close"].copy()
            close.index = pd.to_datetime(close.index)
            return _resample_last_business_month_end(close)

        synthetic_frames: list[pd.Series] = []

        if not vwo_prices_raw.empty and not vxus_prices_raw.empty:
            vwo_m = _monthly_close_generic(vwo_prices_raw)
            vxus_m = _monthly_close_generic(vxus_prices_raw)
            vwo_vxus_spread = (
                vwo_m.pct_change(6, fill_method=None)
                - vxus_m.pct_change(6, fill_method=None)
            ).rename("vwo_vxus_spread_6m")
            synthetic_frames.append(vwo_vxus_spread)

        if not gld_prices_raw.empty and not bnd_prices_raw.empty:
            gld_m = _monthly_close_generic(gld_prices_raw)
            bnd_m = _monthly_close_generic(bnd_prices_raw)
            gold_vs_treasury = (
                gld_m.pct_change(6, fill_method=None)
                - bnd_m.pct_change(6, fill_method=None)
            ).rename("gold_vs_treasury_6m")
            synthetic_frames.append(gold_vs_treasury)

        if not dbc_prices_raw.empty and not voo_prices_raw.empty:
            dbc_m = _monthly_close_generic(dbc_prices_raw)
            voo_m = _monthly_close_generic(voo_prices_raw)
            commodity_equity = (
                dbc_m.pct_change(6, fill_method=None)
                - voo_m.pct_change(6, fill_method=None)
            ).rename("commodity_equity_momentum")
            synthetic_frames.append(commodity_equity)

        for synthetic_series in synthetic_frames:
            if fred_raw.empty:
                fred_raw = synthetic_series.to_frame()
            else:
                fred_raw = fred_raw.join(synthetic_series, how="left")
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Could not build optional benchmark-side relative features; continuing without them. Error=%r",
            exc,
        )

    fred_macro = fred_raw if not fred_raw.empty else None

    return build_feature_matrix(
        price_history=prices,
        dividend_history=dividends,
        split_history=splits,          # may be empty DataFrame with DatetimeIndex
        fundamentals=fundamentals,
        pgr_monthly=pgr_monthly,
        fred_macro=fred_macro,
        force_refresh=True,            # always recompute from DB; never use stale Parquet
    )


def get_X_y(
    df: pd.DataFrame,
    drop_na_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the feature matrix into X (features) and y (target).

    Args:
        df:             DataFrame from build_feature_matrix().
        drop_na_target: If True, drop rows where target_6m_return is NaN
                        (i.e., the final 6 months of history).

    Returns:
        (X, y) tuple where X is the feature DataFrame and y is the target Series.
    """
    if drop_na_target:
        df = df.dropna(subset=["target_6m_return"])
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["target_6m_return"].copy()
    return X, y


def get_X_y_relative(
    df: pd.DataFrame,
    relative_returns: pd.Series,
    drop_na_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align the feature matrix to a relative return target series.

    Used by the v2 WFO engine to build (X, y) pairs where y is the
    PGR-minus-ETF relative return rather than the absolute PGR return.

    Args:
        df:               Feature matrix from build_feature_matrix() or
                          build_feature_matrix_from_db().  Must have a
                          DatetimeIndex of month-end dates.
        relative_returns: Pre-computed relative return Series (e.g. from
                          load_relative_return_matrix() or a column from
                          build_relative_return_targets()).  DatetimeIndex
                          must overlap with ``df``.
        drop_na_target:   If True, drop rows where ``relative_returns`` is NaN.

    Returns:
        ``(X, y)`` where X is the aligned feature DataFrame (same columns as
        ``get_feature_columns(df)``) and y is the relative return Series.

    Raises:
        ValueError: If the aligned index is empty (no date overlap between
                    ``df`` and ``relative_returns``, or all NaN after dropping).
    """
    feature_cols = get_feature_columns(df)
    X_raw = df[feature_cols]

    # Inner join: retain only dates present in both the feature matrix and
    # the relative return series.
    aligned = X_raw.join(relative_returns, how="inner")

    if drop_na_target:
        aligned = aligned.dropna(subset=[relative_returns.name])

    if aligned.empty:
        raise ValueError(
            f"No overlapping non-NaN dates between feature matrix and relative "
            f"return series '{relative_returns.name}'. Verify that both cover "
            f"the same date range and that the series is not all-NaN."
        )

    X = aligned[feature_cols].copy()
    y = aligned[relative_returns.name].copy()
    return X, y


# ---------------------------------------------------------------------------
# v4.0 — Fractional Differentiation (López de Prado FFD method)
# ---------------------------------------------------------------------------

def _fracdiff_weights(d: float, size: int, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute the fixed-width window (FFD) fractional differencing weights.

    w_k = -(d - k + 1) / k × w_{k-1}  for k = 1, 2, ...

    Weights decay to zero; truncate when abs(w_k) < threshold.  The fixed-width
    window retains memory while achieving (approximate) stationarity.

    Args:
        d:         Fractional differencing order in [0, 1].
        size:      Maximum window length.
        threshold: Stop adding weights when |w_k| < threshold.

    Returns:
        Array of weights w_0, w_1, ..., w_K (K ≤ size - 1).
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w[::-1])


def _find_min_d(
    series: pd.Series,
    max_d: float = 0.5,
    corr_threshold: float = 0.90,
    adf_alpha: float = 0.05,
    n_grid: int = 11,
) -> float:
    """
    Find the minimum d* that achieves stationarity while preserving memory.

    Searches d in [0, max_d] using a grid of n_grid evenly-spaced values.
    Returns the smallest d such that:
      1. ADF test rejects unit root at adf_alpha level, AND
      2. Pearson correlation with the original ≥ corr_threshold.

    Args:
        series:         Input time series (non-stationary, e.g. log price).
        max_d:          Maximum d to try.
        corr_threshold: Minimum correlation with original series.
        adf_alpha:      ADF significance level for stationarity test.
        n_grid:         Number of grid points in [0, max_d].

    Returns:
        Minimum d* in [0, max_d] satisfying both conditions, or max_d if none found.
    """
    from scipy.stats import pearsonr

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return max_d

    series_clean = series.dropna()
    if len(series_clean) < 20:
        return max_d

    original = series_clean.values
    d_grid = np.linspace(0.0, max_d, n_grid)

    for d in d_grid:
        weights = _fracdiff_weights(d, len(original))
        width = len(weights)
        if width > len(original):
            continue

        # Apply weights as a convolution (moving window)
        result = np.full(len(original), np.nan)
        for i in range(width - 1, len(original)):
            window = original[i - width + 1: i + 1]
            result[i] = np.dot(weights, window)

        valid_mask = ~np.isnan(result)
        if valid_mask.sum() < 10:
            continue

        diff_series = result[valid_mask]
        orig_aligned = original[valid_mask]

        # Stationarity check via ADF
        try:
            adf_pval = adfuller(diff_series, autolag="AIC")[1]
        except Exception:  # noqa: BLE001
            continue

        # Memory preservation via Pearson correlation
        try:
            corr_val, _ = pearsonr(diff_series, orig_aligned)
        except Exception:  # noqa: BLE001
            continue

        if adf_pval < adf_alpha and abs(corr_val) >= corr_threshold:
            return float(d)

    return float(max_d)


def apply_fracdiff(
    series: pd.Series,
    max_d: float | None = None,
    corr_threshold: float | None = None,
    adf_alpha: float | None = None,
) -> tuple[pd.Series, float]:
    """
    Apply fractional differentiation to achieve stationarity while preserving memory.

    Uses the Fixed-width Window (FFD) approach from López de Prado (2018),
    "Advances in Financial Machine Learning", Chapter 5.

    This is designed for non-stationary level series (e.g., log prices) that
    integer differencing would make too noisy.  Integer differencing (d=1) is
    equivalent to simple returns — highly stationary but loses all memory of
    price levels.  Fractional differencing at d∈(0,0.5) preserves long memory
    while achieving statistical stationarity.

    Do NOT apply to return series (which are already differenced).

    The minimum d* is found by grid search, selecting the smallest d such that:
      1. ADF test rejects the unit root (stationarity), AND
      2. Pearson correlation with the original series ≥ corr_threshold (memory).

    Note: The ``fracdiff`` package is not available for Python ≥ 3.11.  This
    implementation uses numpy/scipy and produces equivalent results for the
    FFD method used in this project.

    Args:
        series:          Input Series (non-stationary level, e.g. log price).
                         Must have a DatetimeIndex.
        max_d:           Maximum fractional order to try.  Default: config.FRACDIFF_MAX_D.
        corr_threshold:  Minimum Pearson correlation.  Default: config.FRACDIFF_CORR_THRESHOLD.
        adf_alpha:       ADF significance level.  Default: config.FRACDIFF_ADF_ALPHA.

    Returns:
        Tuple of (differenced_series, d_star) where:
          - differenced_series has the same DatetimeIndex as input, with NaN
            in the burn-in window.
          - d_star is the minimum differencing order used.

    Raises:
        ValueError: If ``series`` has fewer than 20 non-NaN observations.
    """
    if max_d is None:
        max_d = config.FRACDIFF_MAX_D
    if corr_threshold is None:
        corr_threshold = config.FRACDIFF_CORR_THRESHOLD
    if adf_alpha is None:
        adf_alpha = config.FRACDIFF_ADF_ALPHA

    series_clean = series.dropna()
    if len(series_clean) < 20:
        raise ValueError(
            f"apply_fracdiff requires ≥ 20 non-NaN observations, got {len(series_clean)}."
        )

    d_star = _find_min_d(
        series_clean,
        max_d=max_d,
        corr_threshold=corr_threshold,
        adf_alpha=adf_alpha,
    )

    # Apply FFD at d_star
    original = series_clean.values
    weights = _fracdiff_weights(d_star, len(original))
    width = len(weights)

    result = np.full(len(original), np.nan)
    for i in range(width - 1, len(original)):
        window = original[i - width + 1: i + 1]
        result[i] = np.dot(weights, window)

    diff_series = pd.Series(result, index=series_clean.index, name=series.name)
    # Reindex to original index (NaN for any missing dates)
    diff_series = diff_series.reindex(series.index)
    return diff_series, d_star


# ---------------------------------------------------------------------------
# v7.4 — Observations-to-feature ratio guard
# ---------------------------------------------------------------------------

def compute_obs_feature_ratio(
    X: pd.DataFrame,
    min_ratio: float = 4.0,
    warn: bool = True,
) -> dict:
    """Compute the observations-to-feature ratio and flag dangerous territory.

    With ~280 monthly observations and 25+ features the per-WFO-fold ratio
    approaches 2.4:1 (60-month training window ÷ 25 features), increasing
    the risk of spurious in-sample fit.  A ratio below 4.0 is a warning sign;
    below 2.0 is a failure condition where regularised regression is unlikely
    to generalise.

    This function operates on the full feature matrix (not per-fold).  The
    per-fold ratio is approximately:
        per_fold_ratio = (WFO_TRAIN_WINDOW_MONTHS) / n_features

    Args:
        X:         Feature DataFrame (rows = monthly observations, columns = features).
                   NaN rows are excluded from the observation count.
        min_ratio: Minimum acceptable full-matrix obs/feature ratio.
                   Default: 4.0 (matches the target set when FEATURES_TO_DROP
                   was introduced in v4.3).
        warn:      If True, emit an import-time warning when ratio < min_ratio.
                   Set False in tests to suppress output.

    Returns:
        Dict with keys:
          n_obs (int)          — non-NaN row count (rows with all features present)
          n_features (int)     — number of feature columns
          ratio (float)        — n_obs / n_features
          per_fold_ratio (float) — WFO_TRAIN_WINDOW_MONTHS / n_features
          verdict (str)        — "OK", "WARNING", or "FAIL"
          message (str)        — human-readable summary
    """
    import warnings
    import config as _config

    feature_cols = [c for c in X.columns if c != "target_6m_return"]
    n_features = len(feature_cols)

    if n_features == 0:
        return {
            "n_obs": 0,
            "n_features": 0,
            "ratio": float("nan"),
            "per_fold_ratio": float("nan"),
            "verdict": "FAIL",
            "message": "No feature columns found in X.",
        }

    # Count rows where all feature columns are non-NaN.
    n_obs = int(X[feature_cols].dropna(how="any").shape[0])
    ratio = n_obs / n_features if n_features > 0 else float("nan")
    per_fold_ratio = _config.WFO_TRAIN_WINDOW_MONTHS / n_features

    if ratio < 2.0 or per_fold_ratio < 2.0:
        verdict = "FAIL"
    elif ratio < min_ratio or per_fold_ratio < min_ratio:
        verdict = "WARNING"
    else:
        verdict = "OK"

    message = (
        f"obs/feature ratio: {ratio:.1f} (full matrix), "
        f"{per_fold_ratio:.1f} (per WFO fold, {_config.WFO_TRAIN_WINDOW_MONTHS}M window).  "
        f"n_obs={n_obs}, n_features={n_features}.  Verdict: {verdict}."
    )

    if warn and verdict != "OK":
        warnings.warn(
            f"v7.4 obs/feature ratio guard — {message}",
            UserWarning,
            stacklevel=2,
        )

    return {
        "n_obs": n_obs,
        "n_features": n_features,
        "ratio": ratio,
        "per_fold_ratio": per_fold_ratio,
        "verdict": verdict,
        "message": message,
    }
