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

import numpy as np
import pandas as pd

import config


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
    # Snap back to month-end after the MonthStart shift
    result.index = result.index + pd.offsets.MonthEnd(0)
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
    monthly_close = price_history["close"].resample("BME").last()
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
        # Trailing 12-month combined ratio (rolling mean of monthly values)
        if "combined_ratio" in pgr_monthly.columns:
            cr_monthly = pgr_monthly["combined_ratio"].reindex(
                monthly_dates, method="ffill"
            )
            df["combined_ratio_ttm"] = cr_monthly.rolling(12, min_periods=6).mean()

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
            df["underwriting_income_growth_yoy"] = uw_monthly.pct_change(periods=12)

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
            df["investment_income_growth_yoy"] = inv_inc.pct_change(periods=12)

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
        "channel_mix_agency_pct", "npw_growth_yoy",
        # v6.4 P2.x features
        "underwriting_income", "underwriting_income_3m",
        "underwriting_income_growth_yoy",
        "unearned_premium_growth_yoy", "unearned_premium_to_npw_ratio",
        "roe_net_income_ttm", "roe_trend",
        "investment_income_growth_yoy", "investment_book_yield",
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

        # credit_spread_ig: Moody's Baa minus 10Y Treasury
        if "BAA10Y" in fred_aligned.columns:
            df["credit_spread_ig"] = fred_aligned["BAA10Y"]

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
            df["insurance_cpi_mom3m"] = ins_cpi.pct_change(periods=3)

        # vmt_yoy: year-over-year % change in vehicle miles traveled
        if "TRFVOLUSM227NFWA" in fred_aligned.columns:
            vmt = fred_aligned["TRFVOLUSM227NFWA"]
            df["vmt_yoy"] = vmt.pct_change(periods=12)

        # ------------------------------------------------------------------
        # v4.5 PGR-specific severity and pricing features
        # ------------------------------------------------------------------

        # used_car_cpi_yoy: YoY % change in used car & truck CPI (CUSR0000SETA02).
        # Rising used car prices drive higher total-loss settlement costs, a direct
        # headwind to PGR's combined ratio; the 2021–22 spike was a major headwind.
        if "CUSR0000SETA02" in fred_aligned.columns:
            df["used_car_cpi_yoy"] = fred_aligned["CUSR0000SETA02"].pct_change(12)

        # medical_cpi_yoy: YoY % change in medical care CPI (CUSR0000SAM2).
        # Bodily injury and PIP claim severity tracks medical inflation directly.
        if "CUSR0000SAM2" in fred_aligned.columns:
            df["medical_cpi_yoy"] = fred_aligned["CUSR0000SAM2"].pct_change(12)

        # ppi_auto_ins_yoy: YoY % change in PPI for Private Passenger Auto Insurance
        # (PCU5241265241261).  Replaces the originally planned CUSR0000SETC01 (motor
        # vehicle insurance CPI) which is unavailable via FRED.  The PPI captures
        # cost-based pricing pressure upstream of the consumer CPI; rising PPI signals
        # that carriers have pricing power and are raising premiums.
        # Validated 2026-03-29: partial IC=0.353 (p<0.0001), hit-rate 76.1%.
        if "PCU5241265241261" in fred_aligned.columns:
            df["ppi_auto_ins_yoy"] = fred_aligned["PCU5241265241261"].pct_change(12)

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
        monthly_close_vals = (
            prices["close"].copy()
            .pipe(lambda s: s.set_axis(pd.to_datetime(s.index)))
            .resample("ME")
            .last()
        )
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
            roe_monthly = roe_q.resample("ME").last().ffill()
            roe_monthly = roe_monthly.shift(config.EDGAR_FILING_LAG_MONTHS, freq="MS")
            roe_monthly.index = roe_monthly.index + pd.offsets.MonthEnd(0)
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
    all_fred_series = list(config.FRED_SERIES_MACRO) + list(config.FRED_SERIES_PGR)
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
                return close.resample("ME").last()

            pgr_m = _monthly_close(pgr_prices_raw)
            kie_m = _monthly_close(kie_prices_raw)
            pgr_6m = pgr_m.pct_change(6)
            kie_6m = kie_m.pct_change(6)
            pgr_vs_kie = (pgr_6m - kie_6m).rename("pgr_vs_kie_6m")
            pgr_vs_kie.index = pgr_vs_kie.index + pd.offsets.MonthEnd(0)
            if fred_raw.empty:
                fred_raw = pgr_vs_kie.to_frame()
            else:
                fred_raw = fred_raw.join(pgr_vs_kie, how="left")
    except Exception:  # noqa: BLE001
        pass  # KIE not yet in DB — feature silently absent until data accumulates

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
                return c.resample("ME").last()

            pgr_m_v60 = _m_close(prices)
            peer_monthly_df = pd.concat(
                [_m_close(df) for df in available_peer_frames], axis=1
            )
            peer_composite_6m = peer_monthly_df.pct_change(6).mean(axis=1)
            pgr_6m_v60 = pgr_m_v60.pct_change(6)
            pgr_vs_peers = (pgr_6m_v60 - peer_composite_6m).rename("pgr_vs_peers_6m")
            pgr_vs_peers.index = pgr_vs_peers.index + pd.offsets.MonthEnd(0)
            if fred_raw.empty:
                fred_raw = pgr_vs_peers.to_frame()
            else:
                fred_raw = fred_raw.join(pgr_vs_peers, how="left")
    except Exception:  # noqa: BLE001
        pass  # Peer prices not yet in DB — feature silently absent until bootstrapped

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
                return c.resample("ME").last()

            pgr_m_vfh = _mc_vfh(prices)
            vfh_m = _mc_vfh(vfh_prices_raw)
            pgr_6m_vfh = pgr_m_vfh.pct_change(6)
            vfh_6m = vfh_m.pct_change(6)
            pgr_vs_vfh = (pgr_6m_vfh - vfh_6m).rename("pgr_vs_vfh_6m")
            pgr_vs_vfh.index = pgr_vs_vfh.index + pd.offsets.MonthEnd(0)
            if fred_raw.empty:
                fred_raw = pgr_vs_vfh.to_frame()
            else:
                fred_raw = fred_raw.join(pgr_vs_vfh, how="left")
    except Exception:  # noqa: BLE001
        pass  # VFH not yet in DB — feature silently absent until data accumulates

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
