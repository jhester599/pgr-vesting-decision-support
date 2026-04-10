"""v49 - Regime indicator features: hard_market, high_vol, inverted_curve."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
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


def add_regime_features(x_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Append regime indicators and interactions derived only from same-date features."""
    df = x_df.copy()
    df["hard_market"] = (df["combined_ratio_ttm"] > 100).astype(float) if "combined_ratio_ttm" in df.columns else 0.0
    df["high_vol"] = (df["vix"] > 20).astype(float) if "vix" in df.columns else 0.0
    df["inverted_curve"] = (df["yield_slope"] < 0).astype(float) if "yield_slope" in df.columns else 0.0

    base_cols = list(x_df.columns)
    if variant == "A":
        return df[base_cols + ["hard_market", "high_vol"]]
    if variant == "B":
        return df[base_cols + ["hard_market", "high_vol", "inverted_curve"]]
    if variant == "C":
        df["hm_x_slope"] = df["hard_market"] * df["yield_slope"] if "yield_slope" in df.columns else 0.0
        df["hv_x_spread"] = df["high_vol"] * df["credit_spread_hy"] if "credit_spread_hy" in df.columns else 0.0
        return df[base_cols + ["hard_market", "high_vol", "hm_x_slope", "hv_x_spread"]]
    raise ValueError(f"Unknown variant '{variant}'.")


def main() -> None:
    """Run the regime-indicator augmentation variants."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        output_frames: list[pd.DataFrame] = []

        for variant_name in ["A", "B", "C"]:
            rows: list[dict[str, object]] = []
            for etf in BENCHMARKS:
                rel_series = load_relative_series(conn, etf, horizon=6)
                if rel_series.empty:
                    continue
                x_df, y = get_X_y_relative(df, rel_series, drop_na_target=True)
                base_features = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
                x_aug = add_regime_features(x_df[base_features], variant_name)

                def factory() -> Pipeline:
                    return Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
                        ]
                    )

                y_true, y_hat = custom_wfo(
                    x_aug.to_numpy(),
                    y.to_numpy(),
                    factory,
                    MAX_TRAIN_MONTHS,
                    TEST_SIZE_MONTHS,
                    GAP_MONTHS,
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
            print_header("v49", f"Regime Indicators - Variant {variant_name}")
            print_per_benchmark(rows)
            print_pooled(pooled)
            print_delta(pooled, baseline)
            print_footer()
            output_frames.append(
                build_results_df(rows, pooled, extra_cols={"variant": variant_name})
            )

        save_results(pd.concat(output_frames, ignore_index=True), "v49_regime_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
