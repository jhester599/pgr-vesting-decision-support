"""v44 - Blockwise PCA dimensionality reduction within each WFO fold."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
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
    TEST_SIZE_MONTHS,
    build_results_df,
    compute_metrics,
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
MACRO_BLOCK = [
    "yield_slope",
    "real_rate_10y",
    "credit_spread_hy",
    "nfci",
    "vix",
    "real_yield_change_6m",
]
INSURANCE_BLOCK = [
    "combined_ratio_ttm",
    "npw_growth_yoy",
    "investment_income_growth_yoy",
    "pif_growth_yoy",
]
RAW_PASSTHROUGH = ["mom_12m", "vol_63d"]
VARIANTS: dict[str, dict[str, int]] = {
    "A_2comp_per_block": {"n_macro": 2, "n_insurance": 2},
    "B_1comp_per_block": {"n_macro": 1, "n_insurance": 1},
}


def _impute_train_test(
    train_block: np.ndarray,
    test_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaN using train-fold medians only."""
    if train_block.shape[1] == 0:
        return train_block, test_block

    medians = np.nanmedian(train_block, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train_filled = train_block.copy()
    test_filled = test_block.copy()
    for col_idx in range(train_filled.shape[1]):
        train_filled[np.isnan(train_filled[:, col_idx]), col_idx] = medians[col_idx]
        test_filled[np.isnan(test_filled[:, col_idx]), col_idx] = medians[col_idx]
    return train_filled, test_filled


def _transform_block_with_pca(
    train_block: np.ndarray,
    test_block: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale and reduce one feature block, or return empty arrays when absent."""
    if train_block.shape[1] == 0:
        return (
            np.empty((train_block.shape[0], 0), dtype=float),
            np.empty((test_block.shape[0], 0), dtype=float),
        )

    train_filled, test_filled = _impute_train_test(train_block, test_block)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_filled)
    test_scaled = scaler.transform(test_filled)

    resolved_components = min(n_components, train_scaled.shape[1])
    if resolved_components <= 0:
        return (
            np.empty((train_scaled.shape[0], 0), dtype=float),
            np.empty((test_scaled.shape[0], 0), dtype=float),
        )

    pca = PCA(n_components=resolved_components)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def blockwise_pca_wfo(
    feature_df: pd.DataFrame,
    rel_series: pd.Series,
    n_macro: int,
    n_insurance: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run blockwise PCA inside each fold, then fit Ridge on the compressed space."""
    x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)

    macro_cols = [col for col in MACRO_BLOCK if col in x_df.columns]
    insurance_cols = [col for col in INSURANCE_BLOCK if col in x_df.columns]
    raw_cols = [col for col in RAW_PASSTHROUGH if col in x_df.columns]

    x_macro = x_df[macro_cols].to_numpy() if macro_cols else np.empty((len(x_df), 0))
    x_insurance = x_df[insurance_cols].to_numpy() if insurance_cols else np.empty((len(x_df), 0))
    x_raw = x_df[raw_cols].to_numpy() if raw_cols else np.empty((len(x_df), 0))
    y_values = y.to_numpy()

    n_obs = len(y_values)
    available = n_obs - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=MAX_TRAIN_MONTHS,
        test_size=TEST_SIZE_MONTHS,
        gap=GAP_MONTHS,
    )

    all_y_true: list[float] = []
    all_y_hat: list[float] = []

    for train_idx, test_idx in splitter.split(y_values):
        x_macro_tr, x_macro_te = _transform_block_with_pca(
            x_macro[train_idx],
            x_macro[test_idx],
            n_macro,
        )
        x_ins_tr, x_ins_te = _transform_block_with_pca(
            x_insurance[train_idx],
            x_insurance[test_idx],
            n_insurance,
        )
        x_raw_tr, x_raw_te = _impute_train_test(x_raw[train_idx], x_raw[test_idx])

        x_train = np.column_stack([x_macro_tr, x_ins_tr, x_raw_tr])
        x_test = np.column_stack([x_macro_te, x_ins_te, x_raw_te])

        ridge = RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)
        ridge.fit(x_train, y_values[train_idx])
        y_hat = ridge.predict(x_test)

        all_y_true.extend(y_values[test_idx].tolist())
        all_y_hat.extend(y_hat.tolist())

    return np.asarray(all_y_true), np.asarray(all_y_hat)


def main() -> None:
    """Run the blockwise PCA research variants."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        output_frames: list[pd.DataFrame] = []

        for variant_name, cfg in VARIANTS.items():
            rows: list[dict[str, object]] = []
            for etf in BENCHMARKS:
                rel_series = load_relative_series(conn, etf, horizon=6)
                if rel_series.empty:
                    continue
                y_true, y_hat = blockwise_pca_wfo(
                    df,
                    rel_series,
                    cfg["n_macro"],
                    cfg["n_insurance"],
                )
                metrics = compute_metrics(y_true, y_hat)
                rows.append(
                    {
                        "benchmark": etf,
                        **metrics,
                        "_y_true": y_true,
                        "_y_hat": y_hat,
                    }
                )

            pooled = pool_metrics(rows)
            print_header("v44", f"Blockwise PCA - {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(rows, pooled, extra_cols={"variant": variant_name})
            )

        save_results(pd.concat(output_frames, ignore_index=True), "v44_pca_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
