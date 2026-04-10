"""v48 - Panel pooling: stack all 8 benchmarks with optional fixed effects."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    compute_metrics,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    load_research_baseline_results,
    print_delta,
    print_footer,
    print_header,
    print_pooled,
    save_results,
)

EXTENDED_ALPHAS = np.logspace(0, 6, 100)


def build_panel(feature_df: pd.DataFrame, conn, etfs: list[str]) -> pd.DataFrame:
    """Build a month-sorted stacked panel of feature rows and relative-return targets."""
    panel_rows: list[pd.DataFrame] = []
    for etf in etfs:
        rel_series = load_relative_series(conn, etf, horizon=6)
        if rel_series.empty:
            continue
        x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
        feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
        frame = x_df[feature_cols].copy()
        frame["_month"] = frame.index
        frame["_benchmark"] = etf
        frame["_target"] = y.to_numpy()
        panel_rows.append(frame.reset_index(drop=True))

    if not panel_rows:
        raise ValueError("No panel rows could be built for v48.")

    return pd.concat(panel_rows, ignore_index=True).sort_values(["_month", "_benchmark"]).reset_index(drop=True)


def panel_wfo_with_fixed_effects(
    panel: pd.DataFrame,
    include_fixed_effects: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run month-level panel WFO so all benchmarks for a month stay in the same fold."""
    feature_cols = [col for col in RIDGE_FEATURES_12 if col in panel.columns]
    months = sorted(panel["_month"].unique().tolist())
    n_months = len(months)
    available = n_months - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=MAX_TRAIN_MONTHS,
        test_size=TEST_SIZE_MONTHS,
        gap=GAP_MONTHS,
    )

    benchmark_levels = BENCHMARKS[1:]
    all_y_true: list[float] = []
    all_y_hat: list[float] = []

    for train_month_idx, test_month_idx in splitter.split(months):
        train_months = {months[i] for i in train_month_idx}
        test_months = {months[i] for i in test_month_idx}
        train_panel = panel.loc[panel["_month"].isin(train_months)].copy()
        test_panel = panel.loc[panel["_month"].isin(test_months)].copy()
        if train_panel.empty or test_panel.empty:
            continue

        x_train = train_panel[feature_cols].to_numpy()
        x_test = test_panel[feature_cols].to_numpy()
        y_train = train_panel["_target"].to_numpy()
        y_test = test_panel["_target"].to_numpy()

        medians = np.nanmedian(x_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for col_idx in range(x_train.shape[1]):
            x_train[np.isnan(x_train[:, col_idx]), col_idx] = medians[col_idx]
            x_test[np.isnan(x_test[:, col_idx]), col_idx] = medians[col_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        if include_fixed_effects:
            train_dummies = pd.get_dummies(train_panel["_benchmark"]).reindex(columns=benchmark_levels, fill_value=0)
            test_dummies = pd.get_dummies(test_panel["_benchmark"]).reindex(columns=benchmark_levels, fill_value=0)
            x_train_scaled = np.column_stack([x_train_scaled, train_dummies.to_numpy(dtype=float)])
            x_test_scaled = np.column_stack([x_test_scaled, test_dummies.to_numpy(dtype=float)])

        ridge = RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)
        ridge.fit(x_train_scaled, y_train)
        y_hat = ridge.predict(x_test_scaled)

        all_y_true.extend(y_test.tolist())
        all_y_hat.extend(y_hat.tolist())

    return np.asarray(all_y_true), np.asarray(all_y_hat)


def main() -> None:
    """Run the panel pooling variants."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        panel = build_panel(df, conn, BENCHMARKS)
        output_rows: list[dict[str, object]] = []

        for variant_name, include_fe in [
            ("A_panel_fixed_effects", True),
            ("B_panel_shared_only", False),
        ]:
            y_true, y_hat = panel_wfo_with_fixed_effects(panel, include_fe)
            metrics = compute_metrics(y_true, y_hat)

            print_header("v48", f"Panel Pooling - {variant_name}")
            print(f"  N_OOS: {metrics['n']} (pooled across all benchmarks)")
            print_pooled(metrics)
            print_delta(metrics, baseline)
            print_footer()

            output_rows.append(
                {
                    "variant": variant_name,
                    "version": "v48",
                    "n_total_obs": metrics["n"],
                    **metrics,
                }
            )

        save_results(pd.DataFrame(output_rows), "v48_panel_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
