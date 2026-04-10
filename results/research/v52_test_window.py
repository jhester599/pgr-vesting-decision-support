"""v52 - Shorter WFO test windows: 1M and 3M alternatives."""

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

from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
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
VARIANTS: dict[str, dict[str, int | None]] = {
    "A_test1_rolling60": {"test_size": 1, "max_train": MAX_TRAIN_MONTHS},
    "B_test3_rolling60": {"test_size": 3, "max_train": MAX_TRAIN_MONTHS},
    "C_test1_expanding": {"test_size": 1, "max_train": None},
}


def short_window_wfo(
    x: np.ndarray,
    y: np.ndarray,
    test_size: int,
    max_train: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run WFO with a custom test window and optional expanding train window."""
    base_train = MAX_TRAIN_MONTHS if max_train is None else max_train
    available = len(x) - base_train - GAP_MONTHS
    n_splits = max(1, available // test_size)
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=max_train,
        test_size=test_size,
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
    """Run shorter-window WFO experiments against the v38 baseline."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        output_frames: list[pd.DataFrame] = []

        for variant_name, cfg in VARIANTS.items():
            rows: list[dict[str, object]] = []
            for benchmark in BENCHMARKS:
                rel_series = load_relative_series(conn, benchmark, horizon=6)
                if rel_series.empty:
                    continue
                x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
                feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
                y_true, y_hat = short_window_wfo(
                    x_df[feature_cols].to_numpy(),
                    y.to_numpy(),
                    test_size=int(cfg["test_size"]),
                    max_train=cfg["max_train"],
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
            print_header("v52", f"Shorter Test Windows - {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print(f"  Total OOS observations: {sum(int(row['n']) for row in rows)}")
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(
                    rows,
                    pooled,
                    extra_cols={"variant": variant_name, "version": "v52"},
                )
            )

        save_results(
            pd.concat(output_frames, ignore_index=True),
            "v52_test_window_results.csv",
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
