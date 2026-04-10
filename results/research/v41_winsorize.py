"""v41 - Target winsorization within each WFO fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys
import warnings
from pathlib import Path
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
    load_baseline_results,
    load_feature_matrix,
    load_relative_series,
    pool_metrics,
    print_delta,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)

EXTENDED_ALPHAS = np.logspace(0, 6, 100)
CLIP_LEVELS: dict[str, tuple[int, int]] = {"p5_p95": (5, 95), "p10_p90": (10, 90)}


def wfo_with_target_winsorization(
    x_values: np.ndarray,
    y_values: np.ndarray,
    clip_pct: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Run a Ridge WFO, clipping only the training targets fold by fold."""
    n_obs = len(x_values)
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

    for train_idx, test_idx in splitter.split(x_values):
        x_train = x_values[train_idx].copy()
        x_test = x_values[test_idx].copy()
        y_train = y_values[train_idx]
        y_test = y_values[test_idx]

        medians = np.nanmedian(x_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for col_idx in range(x_train.shape[1]):
            x_train[np.isnan(x_train[:, col_idx]), col_idx] = medians[col_idx]
            x_test[np.isnan(x_test[:, col_idx]), col_idx] = medians[col_idx]

        lower = np.percentile(y_train, clip_pct[0])
        upper = np.percentile(y_train, clip_pct[1])
        y_train_clipped = np.clip(y_train, lower, upper)

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
            ]
        )
        pipe.fit(x_train, y_train_clipped)
        y_hat = pipe.predict(x_test)

        all_y_true.extend(y_test.tolist())
        all_y_hat.extend(y_hat.tolist())

    return np.asarray(all_y_true), np.asarray(all_y_hat)


def main() -> None:
    """Compare two target-winsorization levels against the baseline."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)
        baseline = load_baseline_results()
        output_frames: list[pd.DataFrame] = []

        for level_name, clip_pct in CLIP_LEVELS.items():
            rows: list[dict[str, object]] = []
            for etf in BENCHMARKS:
                rel_series = load_relative_series(conn, etf, horizon=6)
                if rel_series.empty:
                    continue
                x_df, y = get_X_y_relative(df, rel_series, drop_na_target=True)
                feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
                y_true, y_hat = wfo_with_target_winsorization(
                    x_df[feature_cols].to_numpy(),
                    y.to_numpy(),
                    clip_pct,
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
            print_header("v41", f"Target Winsorization - {level_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(rows, pooled, extra_cols={"clip_level": level_name})
            )

        save_results(
            pd.concat(output_frames, ignore_index=True),
            "v41_winsorize_results.csv",
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
