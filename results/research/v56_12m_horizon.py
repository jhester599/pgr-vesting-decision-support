"""v56 - 12-month prediction horizon: rolling and expanding Ridge variants."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from src.models.regularized_models import build_ridge_pipeline
from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    BENCHMARKS,
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

HORIZON_MONTHS = 12
GAP_12M = 15
ROLLING_TRAIN_12M = 60


def run_ridge_wfo_12m(
    x: np.ndarray,
    y: np.ndarray,
    max_train: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a 12M-horizon Ridge WFO with either rolling or expanding history."""
    base_train = ROLLING_TRAIN_12M if max_train is None else max_train
    available = len(x) - base_train - GAP_12M
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=max_train,
        test_size=TEST_SIZE_MONTHS,
        gap=GAP_12M,
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

        pipe = build_ridge_pipeline(target_horizon_months=HORIZON_MONTHS)
        pipe.fit(x_train, y_train)
        y_hat = pipe.predict(x_test)

        all_y_true.extend(y_test.tolist())
        all_y_hat.extend(y_hat.tolist())

    return np.asarray(all_y_true), np.asarray(all_y_hat)


def main() -> None:
    """Run 12M-horizon Ridge experiments against the 6M v38 baseline."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        output_frames: list[pd.DataFrame] = []

        for variant_name, max_train in [
            ("A_rolling60", ROLLING_TRAIN_12M),
            ("B_expanding", None),
        ]:
            rows: list[dict[str, object]] = []
            for benchmark in BENCHMARKS:
                rel_series = load_relative_series(conn, benchmark, horizon=HORIZON_MONTHS)
                if rel_series.empty:
                    continue
                x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
                y_true, y_hat = run_ridge_wfo_12m(
                    x_df.to_numpy(),
                    y.to_numpy(),
                    max_train=max_train,
                )
                metrics = compute_metrics(y_true, y_hat)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "horizon": HORIZON_MONTHS,
                        **metrics,
                        "_y_true": y_true,
                        "_y_hat": y_hat,
                    }
                )

            if not rows:
                continue

            pooled = pool_metrics(rows)
            print_header("v56", f"12-Month Horizon - {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print("  Note: this is a directional comparison vs the 6M v38 baseline only.")
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(
                    rows,
                    pooled,
                    extra_cols={
                        "variant": variant_name,
                        "horizon": HORIZON_MONTHS,
                        "version": "v56",
                    },
                )
            )

        if output_frames:
            save_results(pd.concat(output_frames, ignore_index=True), "v56_12m_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
