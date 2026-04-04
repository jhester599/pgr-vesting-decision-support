"""Point-in-time safe weekly snapshot experiments for v9 follow-up."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from scripts.candidate_model_bakeoff import candidate_feature_sets
from src.database import db_client
from src.models.regularized_models import (
    build_bayesian_ridge_pipeline,
    build_elasticnet_pipeline,
    build_gbt_pipeline,
    build_ridge_pipeline,
)
from src.processing.feature_engineering import build_feature_matrix_from_db
from src.processing.total_return import build_position_series, compute_total_return
from src.research.evaluation import summarize_predictions
from src.research.policy_metrics import evaluate_policy_series
from src.reporting.backtest_report import compute_oos_r_squared


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")
DEFAULT_BENCHMARKS = ["VXUS", "VEA", "VHT", "VPU", "BNDX", "BND", "VNQ"]
DAILY_MOMENTUM_MONTHS: dict[str, int] = {
    "mom_3m": 3,
    "mom_6m": 6,
    "mom_12m": 12,
}
DAILY_VOL_WINDOWS: dict[str, int] = {
    "vol_21d": 21,
    "vol_63d": 63,
}


def _load_ticker_data(
    conn: sqlite3.Connection,
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load price/dividend/split tables for one ticker."""
    prices = db_client.get_prices(conn, ticker)
    dividends_raw = db_client.get_dividends(conn, ticker)
    if dividends_raw.empty:
        dividends = pd.DataFrame(
            columns=["dividend", "source"],
            index=pd.DatetimeIndex([], name="ex_date"),
        )
    else:
        dividends = dividends_raw.rename(columns={"amount": "dividend"})
    splits = db_client.get_splits(conn, ticker)
    if splits.empty:
        splits = pd.DataFrame(
            columns=["split_ratio", "numerator", "denominator"],
            index=pd.DatetimeIndex([], name="split_date"),
        )
    return prices, dividends, splits


def _build_snapshot_dates(
    price_history: pd.DataFrame,
    snapshot_rule: str = "W-FRI",
) -> pd.DatetimeIndex:
    """Use the last available close in each snapshot period."""
    close = price_history["close"].copy()
    close.index = pd.to_datetime(close.index)
    snapshot_dates = close.resample(snapshot_rule).last().dropna().index
    return pd.DatetimeIndex(snapshot_dates, name="date")


def _asof_calendar_return(
    daily_close: pd.Series,
    snapshot_dates: pd.DatetimeIndex,
    months: int,
) -> pd.Series:
    """Compute calendar-month return as of each snapshot date."""
    close_now = daily_close.reindex(snapshot_dates, method="ffill")
    lag_dates = snapshot_dates - pd.DateOffset(months=months)
    lag_values = pd.Series(
        [daily_close.asof(ts) for ts in lag_dates],
        index=snapshot_dates,
        dtype=float,
    )
    return (close_now / lag_values) - 1.0


def _rolling_vol_on_snapshots(
    daily_close: pd.Series,
    snapshot_dates: pd.DatetimeIndex,
    window: int,
) -> pd.Series:
    """Annualized realized vol at each snapshot date."""
    daily_log_ret = np.log(daily_close / daily_close.shift(1))
    rolling_vol = daily_log_ret.rolling(window=window, min_periods=max(5, window // 2)).std() * np.sqrt(252)
    return rolling_vol.reindex(snapshot_dates, method="ffill")


def _high_52w_ratio(
    daily_close: pd.Series,
    snapshot_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Current price divided by trailing 252-day high."""
    trailing_high = daily_close.rolling(window=252, min_periods=126).max()
    close_now = daily_close.reindex(snapshot_dates, method="ffill")
    high_now = trailing_high.reindex(snapshot_dates, method="ffill")
    return close_now / high_now


def _relative_snapshot_feature(
    conn: sqlite3.Connection,
    base_ticker: str,
    comp_tickers: list[str],
    snapshot_dates: pd.DatetimeIndex,
    months: int = 6,
) -> pd.Series:
    """PGR-versus-comparator return spread at snapshot dates."""
    base_prices = db_client.get_prices(conn, base_ticker)
    if base_prices.empty:
        return pd.Series(index=snapshot_dates, dtype=float)
    base_close = base_prices["close"].copy()
    base_close.index = pd.to_datetime(base_close.index)
    base_ret = _asof_calendar_return(base_close, snapshot_dates, months=months)

    comp_returns: list[pd.Series] = []
    for ticker in comp_tickers:
        prices = db_client.get_prices(conn, ticker)
        if prices.empty:
            continue
        close = prices["close"].copy()
        close.index = pd.to_datetime(close.index)
        comp_returns.append(_asof_calendar_return(close, snapshot_dates, months=months))
    if not comp_returns:
        return pd.Series(index=snapshot_dates, dtype=float)
    comp_df = pd.concat(comp_returns, axis=1)
    return (base_ret - comp_df.mean(axis=1)).rename(f"{base_ticker}_vs_comp_{months}m")


def build_weekly_snapshot_feature_matrix(
    conn: sqlite3.Connection,
    snapshot_rule: str = "W-FRI",
) -> pd.DataFrame:
    """Build weekly snapshots by carrying forward monthly state and recomputing price features."""
    monthly_df = build_feature_matrix_from_db(conn, force_refresh=True).copy()
    target_cols = [col for col in monthly_df.columns if col.startswith("target_")]
    monthly_features = monthly_df.drop(columns=target_cols, errors="ignore")

    pgr_prices = db_client.get_prices(conn, "PGR")
    if pgr_prices.empty:
        raise ValueError("PGR prices are required for weekly snapshots.")
    daily_close = pgr_prices["close"].copy()
    daily_close.index = pd.to_datetime(daily_close.index)
    snapshot_dates = _build_snapshot_dates(pgr_prices, snapshot_rule=snapshot_rule)
    snapshot_dates = snapshot_dates[snapshot_dates >= monthly_features.index.min()]

    weekly_df = monthly_features.reindex(snapshot_dates, method="ffill").copy()
    weekly_df.index.name = "date"

    for feature_name, months in DAILY_MOMENTUM_MONTHS.items():
        weekly_df[feature_name] = _asof_calendar_return(daily_close, snapshot_dates, months=months)
    for feature_name, window in DAILY_VOL_WINDOWS.items():
        weekly_df[feature_name] = _rolling_vol_on_snapshots(daily_close, snapshot_dates, window=window)
    weekly_df["high_52w"] = _high_52w_ratio(daily_close, snapshot_dates)

    weekly_df["pgr_vs_kie_6m"] = _relative_snapshot_feature(conn, "PGR", ["KIE"], snapshot_dates, months=6)
    weekly_df["pgr_vs_vfh_6m"] = _relative_snapshot_feature(conn, "PGR", ["VFH"], snapshot_dates, months=6)
    weekly_df["pgr_vs_peers_6m"] = _relative_snapshot_feature(
        conn,
        "PGR",
        list(config.PEER_TICKER_UNIVERSE),
        snapshot_dates,
        months=6,
    )

    last_month_end = monthly_features.index.to_series().reindex(snapshot_dates, method="ffill")
    weekly_df["days_since_month_end_state"] = (snapshot_dates - pd.DatetimeIndex(last_month_end.values)).days.astype(float)
    return weekly_df


def build_weekly_relative_target(
    conn: sqlite3.Connection,
    benchmark: str,
    snapshot_dates: pd.DatetimeIndex,
    forward_months: int,
) -> pd.Series:
    """Compute forward relative DRIP return from each weekly snapshot date."""
    pgr_prices, pgr_divs, pgr_splits = _load_ticker_data(conn, "PGR")
    bench_prices, bench_divs, bench_splits = _load_ticker_data(conn, benchmark)
    if pgr_prices.empty or bench_prices.empty:
        return pd.Series(name=f"{benchmark}_{forward_months}m_weekly", dtype=float)

    pgr_position = build_position_series(pgr_prices, pgr_divs, pgr_splits, initial_shares=1.0)
    bench_position = build_position_series(bench_prices, bench_divs, bench_splits, initial_shares=1.0)

    data_end = min(pgr_position.index.max(), bench_position.index.max())
    returns: dict[pd.Timestamp, float] = {}
    for t in snapshot_dates:
        t_end = t + pd.DateOffset(months=forward_months)
        if t_end > data_end:
            returns[t] = np.nan
            continue
        try:
            pgr_tr = compute_total_return(pgr_position, t, t_end)
            bench_tr = compute_total_return(bench_position, t, t_end)
            returns[t] = float(pgr_tr - bench_tr)
        except (ValueError, KeyError):
            returns[t] = np.nan
    series = pd.Series(returns, name=f"{benchmark}_{forward_months}m_weekly")
    series.index.name = "date"
    return series


def _make_weekly_splitter(
    n_obs: int,
    train_weeks: int = 156,
    test_weeks: int = 13,
    gap_weeks: int = 26,
) -> TimeSeriesSplit:
    """Research-only weekly splitter with a larger purge gap."""
    available = n_obs - train_weeks - gap_weeks
    n_splits = max(1, available // test_weeks)
    return TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=train_weeks,
        test_size=test_weeks,
        gap=gap_weeks,
    )


def _fit_model_pipeline(model_type: str) -> Pipeline:
    """Use the same model classes as the monthly production pipeline."""
    if model_type == "elasticnet":
        return build_elasticnet_pipeline()
    if model_type == "ridge":
        return build_ridge_pipeline()
    if model_type == "bayesian_ridge":
        return build_bayesian_ridge_pipeline()
    if model_type == "gbt":
        return build_gbt_pipeline()
    raise ValueError(f"Unsupported model_type '{model_type}'.")


def evaluate_weekly_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    feature_columns: list[str],
    train_weeks: int = 156,
    test_weeks: int = 13,
    gap_weeks: int = 26,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Run a weekly walk-forward evaluation for one candidate."""
    selected = [col for col in feature_columns if col in X.columns]
    if not selected:
        raise ValueError("feature_columns did not match any weekly snapshot columns.")
    X_selected = X[selected].copy()

    aligned = pd.concat([X_selected, y], axis=1).dropna()
    if len(aligned) < train_weeks + test_weeks + gap_weeks:
        raise ValueError("Not enough weekly observations for the requested WFO windows.")
    X_aligned = aligned[selected]
    y_aligned = aligned[y.name]

    splitter = _make_weekly_splitter(
        n_obs=len(X_aligned),
        train_weeks=train_weeks,
        test_weeks=test_weeks,
        gap_weeks=gap_weeks,
    )

    predictions: list[float] = []
    realized: list[float] = []
    dates: list[pd.Timestamp] = []
    for train_idx, test_idx in splitter.split(X_aligned.values):
        X_train = X_aligned.iloc[train_idx].to_numpy(copy=True)
        X_test = X_aligned.iloc[test_idx].to_numpy(copy=True)
        y_train = y_aligned.iloc[train_idx].to_numpy(copy=True)
        y_test = y_aligned.iloc[test_idx].to_numpy(copy=True)

        train_medians = np.nanmedian(X_train, axis=0)
        train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
        for col_idx in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, col_idx]), col_idx] = train_medians[col_idx]
            X_test[np.isnan(X_test[:, col_idx]), col_idx] = train_medians[col_idx]

        pipeline = _fit_model_pipeline(model_type)
        pipeline.fit(X_train, y_train)
        y_hat = pipeline.predict(X_test)

        predictions.extend(y_hat.tolist())
        realized.extend(y_test.tolist())
        dates.extend(list(X_aligned.index[test_idx]))

    pred_series = pd.Series(predictions, index=pd.DatetimeIndex(dates), name="y_hat")
    realized_series = pd.Series(realized, index=pd.DatetimeIndex(dates), name="y_true")
    summary = summarize_predictions(pred_series, realized_series, target_horizon_months=6)
    metrics = {
        "n_obs": float(summary.n_obs),
        "ic": float(summary.ic),
        "hit_rate": float(summary.hit_rate),
        "oos_r2": float(summary.oos_r2),
        "mae": float(summary.mae),
        "nw_ic": float(summary.nw_ic),
        "nw_p_value": float(summary.nw_p_value),
    }
    return pred_series, realized_series, metrics


def evaluate_weekly_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    train_weeks: int = 156,
    test_weeks: int = 13,
    gap_weeks: int = 26,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Weekly baseline on the same purged snapshot folds."""
    del X
    y_aligned = y.dropna().copy()
    if len(y_aligned) < train_weeks + test_weeks + gap_weeks:
        raise ValueError("Not enough weekly observations for the requested WFO windows.")

    splitter = _make_weekly_splitter(
        n_obs=len(y_aligned),
        train_weeks=train_weeks,
        test_weeks=test_weeks,
        gap_weeks=gap_weeks,
    )

    predictions: list[float] = []
    realized: list[float] = []
    dates: list[pd.Timestamp] = []
    for train_idx, test_idx in splitter.split(y_aligned.to_frame().values):
        y_train = y_aligned.iloc[train_idx]
        y_test = y_aligned.iloc[test_idx]
        if strategy == "historical_mean":
            pred_value = float(y_train.mean())
        elif strategy == "last_value":
            pred_value = float(y_train.iloc[-1])
        elif strategy == "zero":
            pred_value = 0.0
        else:
            raise ValueError(f"Unknown baseline strategy '{strategy}'.")

        predictions.extend([pred_value] * len(test_idx))
        realized.extend(y_test.tolist())
        dates.extend(list(y_test.index))

    pred_series = pd.Series(predictions, index=pd.DatetimeIndex(dates), name="y_hat")
    realized_series = pd.Series(realized, index=pd.DatetimeIndex(dates), name="y_true")
    summary = summarize_predictions(pred_series, realized_series, target_horizon_months=6)
    metrics = {
        "n_obs": float(summary.n_obs),
        "ic": float(summary.ic),
        "hit_rate": float(summary.hit_rate),
        "oos_r2": float(summary.oos_r2),
        "mae": float(summary.mae),
        "nw_ic": float(summary.nw_ic),
        "nw_p_value": float(summary.nw_p_value),
    }
    return pred_series, realized_series, metrics


def summarize_weekly_snapshot_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly snapshot results by candidate."""
    if detail_df.empty:
        return pd.DataFrame()

    modeled_df = detail_df.loc[detail_df["status"] == "ok"].copy()
    if modeled_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for candidate_name, group in modeled_df.groupby("candidate_name", dropna=False):
        candidate_all = detail_df.loc[detail_df["candidate_name"] == candidate_name]
        rows.append(
            {
                "candidate_name": candidate_name,
                "model_type": group["model_type"].iloc[0],
                "n_features": int(group["n_features"].iloc[0]),
                "n_benchmarks": int(group["benchmark"].nunique()),
                "n_benchmarks_attempted": int(candidate_all["benchmark"].nunique()),
                "n_skipped": int((candidate_all["status"] != "ok").sum()),
                "mean_ic": float(group["ic"].mean()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_oos_r2": float(group["oos_r2"].mean()),
                "mean_mae": float(group["mae"].mean()),
                "mean_policy_return_sign": float(group["policy_return_sign"].mean()),
                "mean_policy_return_tiered": float(group["policy_return_tiered"].mean()),
                "mean_uplift_vs_sell_50_sign": float(group["policy_uplift_vs_sell_50_sign"].mean()),
                "mean_uplift_vs_sell_50_tiered": float(group["policy_uplift_vs_sell_50_tiered"].mean()),
                "notes": group["notes"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["mean_policy_return_sign", "mean_oos_r2", "mean_ic"],
        ascending=[False, False, False],
    )


def run_weekly_snapshot_experiments(
    conn: Any,
    benchmarks: list[str],
    output_dir: str,
    snapshot_rule: str = "W-FRI",
    forward_months: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pilot weekly-snapshot evaluation on the reduced benchmark universe."""
    X_weekly = build_weekly_snapshot_feature_matrix(conn, snapshot_rule=snapshot_rule)
    available_columns = set(X_weekly.columns)
    detail_rows: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        y_weekly = build_weekly_relative_target(
            conn,
            benchmark=benchmark,
            snapshot_dates=X_weekly.index,
            forward_months=forward_months,
        )
        if y_weekly.dropna().empty:
            continue

        for candidate_name, spec in candidate_feature_sets().items():
            selected = [feature for feature in spec["features"] if feature in available_columns]
            if not selected:
                continue
            try:
                pred_series, realized_series, metrics = evaluate_weekly_candidate(
                    X_weekly,
                    y_weekly,
                    model_type=str(spec["model_type"]),
                    feature_columns=selected,
                )
            except ValueError as exc:
                detail_rows.append(
                    {
                        "benchmark": benchmark,
                        "candidate_name": candidate_name,
                        "model_type": spec["model_type"],
                        "n_features": len(selected),
                        "feature_columns": ",".join(selected),
                        "notes": spec["notes"],
                        "status": "skipped",
                        "policy_return_sign": np.nan,
                        "policy_return_tiered": np.nan,
                        "policy_uplift_vs_sell_50_sign": np.nan,
                        "policy_uplift_vs_sell_50_tiered": np.nan,
                        "n_obs": np.nan,
                        "ic": np.nan,
                        "hit_rate": np.nan,
                        "oos_r2": np.nan,
                        "mae": np.nan,
                        "nw_ic": np.nan,
                        "nw_p_value": np.nan,
                        "skip_reason": str(exc),
                    }
                )
                continue
            sign_policy = evaluate_policy_series(pred_series, realized_series, "sign_hold_vs_sell")
            tiered_policy = evaluate_policy_series(pred_series, realized_series, "tiered_25_50_100")
            detail_rows.append(
                {
                    "benchmark": benchmark,
                    "candidate_name": candidate_name,
                    "model_type": spec["model_type"],
                    "n_features": len(selected),
                    "feature_columns": ",".join(selected),
                    "notes": spec["notes"],
                    "status": "ok",
                    "policy_return_sign": sign_policy.mean_policy_return,
                    "policy_return_tiered": tiered_policy.mean_policy_return,
                    "policy_uplift_vs_sell_50_sign": sign_policy.uplift_vs_sell_50,
                    "policy_uplift_vs_sell_50_tiered": tiered_policy.uplift_vs_sell_50,
                    "skip_reason": "",
                    **metrics,
                }
            )

        for strategy in ("historical_mean", "last_value", "zero"):
            pred_series, realized_series, metrics = evaluate_weekly_baseline(
                X_weekly,
                y_weekly,
                strategy=strategy,
            )
            sign_policy = evaluate_policy_series(pred_series, realized_series, "sign_hold_vs_sell")
            tiered_policy = evaluate_policy_series(pred_series, realized_series, "tiered_25_50_100")
            detail_rows.append(
                {
                    "benchmark": benchmark,
                    "candidate_name": f"baseline_{strategy}",
                    "model_type": "baseline",
                    "n_features": 0,
                    "feature_columns": "",
                    "notes": "Weekly snapshot non-ML baseline.",
                    "status": "ok",
                    "policy_return_sign": sign_policy.mean_policy_return,
                    "policy_return_tiered": tiered_policy.mean_policy_return,
                    "policy_uplift_vs_sell_50_sign": sign_policy.uplift_vs_sell_50,
                    "policy_uplift_vs_sell_50_tiered": tiered_policy.uplift_vs_sell_50,
                    "skip_reason": "",
                    **metrics,
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_weekly_snapshot_results(detail_df)

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = os.path.join(output_dir, f"weekly_snapshot_experiments_detail_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"weekly_snapshot_experiments_summary_{stamp}.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote weekly snapshot detail to {detail_path}")
    print(f"Wrote weekly snapshot summary to {summary_path}")
    return detail_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 weekly snapshot experiments.")
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmark tickers.",
    )
    parser.add_argument(
        "--snapshot-rule",
        default="W-FRI",
        help="Pandas resample rule for snapshot dates. Default: W-FRI",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    benchmarks = [value.strip() for value in args.benchmarks.split(",") if value.strip()]
    conn = db_client.get_connection(config.DB_PATH)
    run_weekly_snapshot_experiments(
        conn=conn,
        benchmarks=benchmarks,
        output_dir=args.output_dir,
        snapshot_rule=args.snapshot_rule,
    )


if __name__ == "__main__":
    main()
