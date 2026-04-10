"""v51 - Peer company pooling: peer spread feature and two-stage sector signal."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from config.features import PEER_TICKER_UNIVERSE
from src.processing.feature_engineering import get_X_y_relative
from src.processing.multi_total_return import build_etf_monthly_returns
from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    build_results_df,
    compute_metrics,
    custom_wfo,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    load_research_baseline_results,
    pool_metrics,
    print_delta,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)

EXTENDED_ALPHAS = np.logspace(0, 6, 100)
MACRO_FEATURES: list[str] = [
    "yield_slope",
    "real_rate_10y",
    "credit_spread_hy",
    "nfci",
    "vix",
    "real_yield_change_6m",
    "mom_12m",
    "vol_63d",
]


def ridge_factory() -> Pipeline:
    """Return a plain Ridge pipeline for research WFO loops."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
        ]
    )


def build_peer_composite_relative_series(
    conn,
    benchmark: str,
    horizon: int = 6,
) -> pd.Series | None:
    """Build equal-weight peer forward returns minus ETF forward returns."""
    peer_frames: list[pd.Series] = []
    for ticker in PEER_TICKER_UNIVERSE:
        peer_series = build_etf_monthly_returns(conn, ticker, horizon)
        if not peer_series.empty:
            peer_frames.append(peer_series.rename(ticker))

    benchmark_series = build_etf_monthly_returns(conn, benchmark, horizon)
    if not peer_frames or benchmark_series.empty:
        return None

    peer_df = pd.concat(peer_frames, axis=1).sort_index()
    peer_composite = peer_df.mean(axis=1).rename("peer_composite_return")
    relative = (peer_composite - benchmark_series).dropna()
    relative.name = f"peer_composite_minus_{benchmark}_{horizon}m"
    return relative


def generate_sector_oos_predictions(
    feature_df: pd.DataFrame,
    conn,
    benchmark: str,
) -> pd.Series | None:
    """Generate stage-1 OOS predictions for peer-composite performance."""
    sector_target = build_peer_composite_relative_series(conn, benchmark, horizon=6)
    if sector_target is None or sector_target.empty:
        return None

    x_df, y = get_X_y_relative(feature_df, sector_target, drop_na_target=True)
    macro_cols = [col for col in MACRO_FEATURES if col in x_df.columns]
    if not macro_cols:
        return None

    x = x_df[macro_cols].to_numpy()
    y_values = y.to_numpy()

    available = len(x) - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=MAX_TRAIN_MONTHS,
        test_size=TEST_SIZE_MONTHS,
        gap=GAP_MONTHS,
    )

    sector_signal = pd.Series(np.nan, index=x_df.index, name="sector_signal")
    for train_idx, test_idx in tscv.split(x):
        x_train = x[train_idx].copy()
        x_test = x[test_idx].copy()
        y_train = y_values[train_idx]

        medians = np.nanmedian(x_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for col_idx in range(x_train.shape[1]):
            x_train[np.isnan(x_train[:, col_idx]), col_idx] = medians[col_idx]
            x_test[np.isnan(x_test[:, col_idx]), col_idx] = medians[col_idx]

        pipe = ridge_factory()
        pipe.fit(x_train, y_train)
        sector_signal.iloc[test_idx] = pipe.predict(x_test)

    return sector_signal


def main() -> None:
    """Run peer-pooling experiments against the v38 research baseline."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        output_frames: list[pd.DataFrame] = []

        rows_peers: list[dict[str, object]] = []
        rows_sector: list[dict[str, object]] = []

        for benchmark in BENCHMARKS:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)

            if "pgr_vs_peers_6m" in x_df.columns:
                peer_feature_cols = [
                    col for col in RIDGE_FEATURES_12 + ["pgr_vs_peers_6m"]
                    if col in x_df.columns
                ]
                y_true_peers, y_hat_peers = custom_wfo(
                    x_df[peer_feature_cols].to_numpy(),
                    y.to_numpy(),
                    ridge_factory,
                    MAX_TRAIN_MONTHS,
                    TEST_SIZE_MONTHS,
                    GAP_MONTHS,
                )
                metrics_peers = compute_metrics(y_true_peers, y_hat_peers)
                rows_peers.append(
                    {
                        "benchmark": benchmark,
                        **metrics_peers,
                        "_y_true": y_true_peers,
                        "_y_hat": y_hat_peers,
                    }
                )

            sector_signal = generate_sector_oos_predictions(feature_df, conn, benchmark)
            if sector_signal is not None:
                x_aug = x_df.copy()
                x_aug["sector_signal"] = sector_signal.reindex(x_aug.index)
                sector_feature_cols = [
                    col for col in RIDGE_FEATURES_12 + ["sector_signal"]
                    if col in x_aug.columns
                ]
                y_true_sector, y_hat_sector = custom_wfo(
                    x_aug[sector_feature_cols].to_numpy(),
                    y.to_numpy(),
                    ridge_factory,
                    MAX_TRAIN_MONTHS,
                    TEST_SIZE_MONTHS,
                    GAP_MONTHS,
                )
                metrics_sector = compute_metrics(y_true_sector, y_hat_sector)
                rows_sector.append(
                    {
                        "benchmark": benchmark,
                        **metrics_sector,
                        "_y_true": y_true_sector,
                        "_y_hat": y_hat_sector,
                    }
                )

        if rows_peers:
            pooled_peers = pool_metrics(rows_peers)
            print_header("v51", "Peer Company Pooling - A_pgr_vs_peers_6m")
            print_per_benchmark(rows_peers)
            print_pooled(pooled_peers)
            print_delta(pooled_peers, baseline)
            print_footer()
            output_frames.append(
                build_results_df(
                    rows_peers,
                    pooled_peers,
                    extra_cols={"variant": "A_pgr_vs_peers_6m", "version": "v51"},
                )
            )

        if rows_sector:
            pooled_sector = pool_metrics(rows_sector)
            print_header("v51", "Peer Company Pooling - B_two_stage_sector")
            print_per_benchmark(rows_sector)
            print_pooled(pooled_sector)
            print_delta(pooled_sector, baseline)
            print_footer()
            output_frames.append(
                build_results_df(
                    rows_sector,
                    pooled_sector,
                    extra_cols={"variant": "B_two_stage_sector", "version": "v51"},
                )
            )

        if not output_frames:
            raise RuntimeError("No v51 results were generated. Peer data may be unavailable.")

        save_results(
            pd.concat(output_frames, ignore_index=True),
            "v51_peer_pooling_results.csv",
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
