"""v59 - Imputation strategies on an expanded 18-feature panel."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, MissingIndicator, SimpleImputer
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
ADDITIONAL_FEATURES: list[str] = [
    "pe_ratio",
    "pb_ratio",
    "roe",
    "buyback_yield",
    "term_premium_10y",
    "breakeven_inflation_10y",
]
ALL_18_FEATURES: list[str] = RIDGE_FEATURES_12 + ADDITIONAL_FEATURES


def impute_wfo(
    x: np.ndarray,
    y: np.ndarray,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run WFO with fold-local imputation."""
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

        if strategy == "ffill_median":
            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(x_train)
            x_test = imputer.transform(x_test)
        elif strategy == "iterative":
            imputer = IterativeImputer(max_iter=10, random_state=42)
            x_train = imputer.fit_transform(x_train)
            x_test = imputer.transform(x_test)
        elif strategy == "missing_indicator":
            imputer = IterativeImputer(max_iter=10, random_state=42)
            indicator = MissingIndicator(features="all")
            x_train_indicator = indicator.fit_transform(x_train)
            x_test_indicator = indicator.transform(x_test)
            x_train = np.column_stack([imputer.fit_transform(x_train), x_train_indicator])
            x_test = np.column_stack([imputer.transform(x_test), x_test_indicator])
        else:
            raise ValueError(f"Unknown imputation strategy '{strategy}'.")

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
            ]
        )
        pipe.fit(x_train, y_train)
        y_hat = pipe.predict(x_test)

        all_y_true.extend(y_test.tolist())
        all_y_hat.extend(y_hat.tolist())

    return np.asarray(all_y_true), np.asarray(all_y_hat)


def main() -> None:
    """Run imputation experiments against the v38 research baseline."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        available_features = [col for col in ALL_18_FEATURES if col in feature_df.columns]
        print(f"Features available: {len(available_features)}/18: {available_features}")
        output_frames: list[pd.DataFrame] = []

        strategies = ["ffill_median", "iterative", "missing_indicator"]
        for strategy in strategies:
            rows: list[dict[str, object]] = []
            for benchmark in BENCHMARKS:
                rel_series = load_relative_series(conn, benchmark, horizon=6)
                if rel_series.empty:
                    continue
                x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
                x_selected = x_df[available_features].copy()
                if strategy == "ffill_median":
                    x_selected = x_selected.ffill()

                y_true, y_hat = impute_wfo(
                    x_selected.to_numpy(),
                    y.to_numpy(),
                    strategy=strategy,
                )
                metrics = compute_metrics(y_true, y_hat)
                rows.append(
                    {
                        "benchmark": benchmark,
                        **metrics,
                        "_y_true": y_true,
                        "_y_hat": y_hat,
                    }
                )

            pooled = pool_metrics(rows)
            print_header("v59", f"Imputation - {strategy}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(
                    rows,
                    pooled,
                    extra_cols={"variant": strategy, "version": "v59"},
                )
            )

        save_results(
            pd.concat(output_frames, ignore_index=True),
            "v59_imputation_results.csv",
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
