"""v57 - Feature transformations: logs, rank-normalization, and lags."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from src.processing.feature_engineering import get_X_y_relative
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
SKEWED_FEATURES: list[str] = [
    "npw_growth_yoy",
    "investment_income_growth_yoy",
    "pif_growth_yoy",
    "book_value_per_share_growth_yoy",
]
LAG_FEATURES: list[str] = [
    "combined_ratio_ttm",
    "npw_growth_yoy",
    "investment_income_growth_yoy",
]


def variant_a_log_transform(x_df: pd.DataFrame) -> pd.DataFrame:
    """Apply signed log1p to skewed features."""
    df = x_df.copy()
    for col in SKEWED_FEATURES:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    return df


def variant_c_add_lags(df_full: pd.DataFrame, x_df: pd.DataFrame) -> pd.DataFrame:
    """Append 1M and 2M lags of slow-moving fundamentals."""
    df = x_df.copy()
    for feature_name in LAG_FEATURES:
        if feature_name in df_full.columns:
            df[f"{feature_name}_lag1"] = df_full[feature_name].shift(1).reindex(df.index)
            df[f"{feature_name}_lag2"] = df_full[feature_name].shift(2).reindex(df.index)
    return df


def rank_norm_wfo(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank-normalize features per fold, then fit a shallow GBT."""
    available = len(x) - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=MAX_TRAIN_MONTHS,
        test_size=TEST_SIZE_MONTHS,
        gap=GAP_MONTHS,
    )

    all_y_true: list[float] = []
    all_y_hat: list[float] = []
    for train_idx, test_idx in tscv.split(x):
        x_train = x[train_idx].copy()
        x_test = x[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        medians = np.nanmedian(x_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for col_idx in range(x_train.shape[1]):
            x_train[np.isnan(x_train[:, col_idx]), col_idx] = medians[col_idx]
            x_test[np.isnan(x_test[:, col_idx]), col_idx] = medians[col_idx]

        for col_idx in range(x_train.shape[1]):
            ranks_train = rankdata(x_train[:, col_idx], method="average") / len(x_train)
            sorted_train = np.sort(x_train[:, col_idx])
            mapped_test = np.searchsorted(sorted_train, x_test[:, col_idx], side="left")
            x_train[:, col_idx] = ranks_train
            x_test[:, col_idx] = np.clip(mapped_test / len(sorted_train), 0.0, 1.0)

        gbt = GradientBoostingRegressor(
            max_depth=2,
            n_estimators=50,
            learning_rate=0.01,
            min_samples_leaf=10,
            subsample=0.7,
            random_state=42,
        )
        gbt.fit(x_train, y_train)
        y_hat = gbt.predict(x_test)

        all_y_true.extend(y_test.tolist())
        all_y_hat.extend(y_hat.tolist())

    return np.asarray(all_y_true), np.asarray(all_y_hat)


def main() -> None:
    """Run feature-transformation experiments against the v38 baseline."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        output_frames: list[pd.DataFrame] = []

        rows_log: list[dict[str, object]] = []
        for benchmark in BENCHMARKS:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
            feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
            x_transformed = variant_a_log_transform(x_df[feature_cols])

            def ridge_factory() -> Pipeline:
                return Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
                    ]
                )

            y_true, y_hat = custom_wfo(
                x_transformed.to_numpy(),
                y.to_numpy(),
                ridge_factory,
                MAX_TRAIN_MONTHS,
                TEST_SIZE_MONTHS,
                GAP_MONTHS,
            )
            metrics = compute_metrics(y_true, y_hat)
            rows_log.append(
                {
                    "benchmark": benchmark,
                    **metrics,
                    "_y_true": y_true,
                    "_y_hat": y_hat,
                }
            )

        pooled_log = pool_metrics(rows_log)
        print_header("v57", "Feature Transformations - A_log_transform")
        print_per_benchmark(rows_log)
        print_pooled(pooled_log)
        print_delta(pooled_log, baseline)
        print_footer()
        output_frames.append(
            build_results_df(
                rows_log,
                pooled_log,
                extra_cols={"variant": "A_log_transform", "version": "v57"},
            )
        )

        rows_rank: list[dict[str, object]] = []
        for benchmark in BENCHMARKS:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
            feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
            y_true, y_hat = rank_norm_wfo(
                x_df[feature_cols].to_numpy(),
                y.to_numpy(),
            )
            metrics = compute_metrics(y_true, y_hat)
            rows_rank.append(
                {
                    "benchmark": benchmark,
                    **metrics,
                    "_y_true": y_true,
                    "_y_hat": y_hat,
                }
            )

        pooled_rank = pool_metrics(rows_rank)
        print_header("v57", "Feature Transformations - B_rank_norm_gbt")
        print_per_benchmark(rows_rank)
        print_pooled(pooled_rank)
        print_delta(pooled_rank, baseline)
        print_footer()
        output_frames.append(
            build_results_df(
                rows_rank,
                pooled_rank,
                extra_cols={"variant": "B_rank_norm_gbt", "version": "v57"},
            )
        )

        rows_lags: list[dict[str, object]] = []
        for benchmark in BENCHMARKS:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
            feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
            x_aug = variant_c_add_lags(feature_df, x_df[feature_cols])

            def ridge_factory_lags() -> Pipeline:
                return Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
                    ]
                )

            y_true, y_hat = custom_wfo(
                x_aug.to_numpy(),
                y.to_numpy(),
                ridge_factory_lags,
                MAX_TRAIN_MONTHS,
                TEST_SIZE_MONTHS,
                GAP_MONTHS,
            )
            metrics = compute_metrics(y_true, y_hat)
            rows_lags.append(
                {
                    "benchmark": benchmark,
                    **metrics,
                    "_y_true": y_true,
                    "_y_hat": y_hat,
                }
            )

        pooled_lags = pool_metrics(rows_lags)
        print_header("v57", "Feature Transformations - C_fundamental_lags")
        print_per_benchmark(rows_lags)
        print_pooled(pooled_lags)
        print_delta(pooled_lags, baseline)
        print_footer()
        output_frames.append(
            build_results_df(
                rows_lags,
                pooled_lags,
                extra_cols={"variant": "C_fundamental_lags", "version": "v57"},
            )
        )

        save_results(pd.concat(output_frames, ignore_index=True), "v57_transforms_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
