"""v47 - Composite benchmark target: equal-weight, inv-vol-weight, equity-only."""

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
    compute_metrics,
    custom_wfo,
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
EQUITY_ETFS = ["VOO", "VXUS", "VWO", "VDE"]


def build_composite_target(
    feature_df: pd.DataFrame,
    conn,
    etfs: list[str],
    weighting: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct a composite relative-return target and align features."""
    rel_dict: dict[str, pd.Series] = {}
    for etf in etfs:
        rel_series = load_relative_series(conn, etf, horizon=6)
        if not rel_series.empty:
            rel_dict[etf] = rel_series

    rel_df = pd.DataFrame(rel_dict).dropna(how="all")
    if rel_df.empty:
        raise ValueError("No relative-return series available for composite benchmark.")

    if weighting == "equal":
        composite = rel_df.mean(axis=1)
    elif weighting == "inv_vol":
        vol = rel_df.std().replace(0.0, np.nan)
        inv_vol = 1.0 / vol
        weights = (inv_vol / inv_vol.sum()).fillna(0.0)
        composite = rel_df.dot(weights)
    else:
        raise ValueError(f"Unknown weighting '{weighting}'.")

    composite = composite.rename("composite_relative_return").dropna()
    feature_aligned = feature_df.reindex(composite.index)
    return feature_aligned, composite


def main() -> None:
    """Run composite-target experiments."""
    conn = get_connection()
    try:
        df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()
        variants = {
            "A_equal_weighted": {"etfs": BENCHMARKS, "weighting": "equal"},
            "B_inv_vol_weighted": {"etfs": BENCHMARKS, "weighting": "inv_vol"},
            "C_equity_only": {"etfs": EQUITY_ETFS, "weighting": "equal"},
        }

        rows: list[dict[str, object]] = []
        for variant_name, cfg in variants.items():
            feature_aligned, composite = build_composite_target(
                df,
                conn,
                cfg["etfs"],
                str(cfg["weighting"]),
            )
            x_df, y = get_X_y_relative(feature_aligned, composite, drop_na_target=True)
            feature_cols = [col for col in RIDGE_FEATURES_12 if col in x_df.columns]
            x_values = x_df[feature_cols].to_numpy()

            def factory() -> Pipeline:
                return Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
                    ]
                )

            y_true, y_hat = custom_wfo(
                x_values,
                y.to_numpy(),
                factory,
                MAX_TRAIN_MONTHS,
                TEST_SIZE_MONTHS,
                GAP_MONTHS,
            )
            metrics = compute_metrics(y_true, y_hat)

            print_header("v47", f"Composite Benchmark - {variant_name}")
            print(f"  N_OOS: {metrics['n']}   ETFs used: {cfg['etfs']}")
            print_pooled(metrics)
            print_delta(metrics, baseline)
            print_footer()

            rows.append(
                {
                    "variant": variant_name,
                    "benchmark": "COMPOSITE",
                    "version": "v47",
                    **metrics,
                }
            )

        save_results(pd.DataFrame(rows), "v47_composite_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
