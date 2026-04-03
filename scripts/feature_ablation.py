"""
Feature Ablation Backtest — v7.0

Systematically measures the marginal predictive contribution of every feature
group added since v5.0 and flags groups that are net-negative (i.e., their
inclusion *decreases* mean IC relative to the prior cumulative group).

Usage:
    python scripts/feature_ablation.py [--benchmarks VTI,VOO,BND] [--horizons 6]

Output:
    results/backtests/feature_ablation_YYYYMMDD.csv

CSV columns:
    feature_group, benchmark, model_type, n_obs, n_features, ic, hit_rate,
    mae, oos_r2
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd

# Ensure repo root is on sys.path when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from src.database import db_client
from src.models.wfo_engine import run_wfo
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_feature_columns,
    get_X_y_relative,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.reporting.backtest_report import compute_oos_r_squared


# ---------------------------------------------------------------------------
# Feature group definitions (cumulative — each group contains all prior cols)
# ---------------------------------------------------------------------------

# Group A: price-derived momentum and volatility only.
_GROUP_A: list[str] = [
    "mom_3m", "mom_6m", "mom_12m", "vol_63d",
]

# Group B: add macro/credit/volatility regime features.
_GROUP_B: list[str] = _GROUP_A + [
    "yield_slope", "yield_curvature", "real_rate_10y",
    "credit_spread_hy", "nfci", "vix", "vmt_yoy",
]

# Group C: add core EDGAR fundamentals and valuation ratios.
_GROUP_C: list[str] = _GROUP_B + [
    "combined_ratio_ttm", "pif_growth_yoy", "gainshare_est",
    "pe_ratio", "pb_ratio",
]

# Group D: add v6.0 peer-relative and claims-severity features.
_GROUP_D: list[str] = _GROUP_C + [
    "high_52w", "pgr_vs_peers_6m", "pgr_vs_vfh_6m",
    "pgr_vs_kie_6m", "used_car_cpi_yoy", "medical_cpi_yoy",
    "cr_acceleration",
]

# Group E: add v6.3/v6.4 channel-mix, NPW, underwriting, ROE, investment,
# and buyback features.
_GROUP_E: list[str] = _GROUP_D + [
    "channel_mix_agency_pct", "npw_growth_yoy",
    "underwriting_income", "underwriting_income_3m",
    "underwriting_income_growth_yoy",
    "unearned_premium_growth_yoy", "unearned_premium_to_npw_ratio",
    "roe_net_income_ttm", "roe_trend",
    "investment_income_growth_yoy", "investment_book_yield",
    "buyback_yield", "buyback_acceleration",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "A_price_only": _GROUP_A,
    "B_plus_macro": _GROUP_B,
    "C_plus_edgar_core": _GROUP_C,
    "D_plus_v60": _GROUP_D,
    "E_plus_v63_v64": _GROUP_E,
}

# Ordered group labels for incremental analysis.
GROUP_ORDER: list[str] = list(FEATURE_GROUPS.keys())

# Default benchmark set (representative coverage across asset classes).
DEFAULT_BENCHMARKS: list[str] = [
    "VTI", "VOO", "VFH", "BND", "VHT", "GLD", "VNQ", "VXUS",
]

DEFAULT_HORIZONS: list[int] = [6]
DEFAULT_OUTPUT_DIR: str = os.path.join("results", "backtests")

# Model types to evaluate for each group.
MODEL_TYPES: list[str] = ["elasticnet", "gbt"]


# ---------------------------------------------------------------------------
# Core ablation logic
# ---------------------------------------------------------------------------

def _filter_available(
    group_cols: list[str],
    df_cols: set[str],
) -> list[str]:
    """Return only columns from ``group_cols`` that exist in ``df_cols``."""
    return [c for c in group_cols if c in df_cols]


def run_ablation(
    conn: Any,
    benchmarks: list[str],
    horizons: list[int],
    output_dir: str,
) -> pd.DataFrame:
    """Run the full feature ablation and return the results DataFrame.

    Args:
        conn:        Open SQLite connection (schema already initialized).
        benchmarks:  List of ETF tickers to test against.
        horizons:    List of target horizons in months.
        output_dir:  Directory for the output CSV.

    Returns:
        DataFrame with one row per (group, benchmark, model_type, horizon).
    """
    if not benchmarks:
        raise ValueError("benchmarks list must not be empty.")

    print("Building feature matrix from DB …")
    df = build_feature_matrix_from_db(conn)
    all_feature_cols: set[str] = set(get_feature_columns(df))

    records: list[dict] = []

    for horizon in horizons:
        print(f"\n=== Horizon: {horizon}M ===")
        for group_label, group_cols in FEATURE_GROUPS.items():
            if not group_cols:
                raise ValueError(
                    f"Feature group '{group_label}' is empty.  "
                    "Each group must define at least one column."
                )

            available = _filter_available(group_cols, all_feature_cols)
            n_features = len(available)

            if n_features == 0:
                print(
                    f"  [{group_label}] All columns absent in feature matrix "
                    "— skipping."
                )
                continue

            print(
                f"  [{group_label}] {n_features}/{len(group_cols)} features "
                f"available: {available[:5]}{'…' if n_features > 5 else ''}"
            )

            for benchmark in benchmarks:
                rel_series = load_relative_return_matrix(
                    conn, benchmark, horizon
                )

                if rel_series.empty:
                    print(
                        f"    [{benchmark}] No relative returns in DB — skip."
                    )
                    continue

                # Build (X, y) aligned to this benchmark's relative returns.
                X_group = df[available].copy()
                try:
                    X_aligned, y_aligned = get_X_y_relative(
                        X_group.assign(**{c: df[c] for c in get_feature_columns(df) if c not in available}).pipe(
                            # Rebuild a DataFrame that has the full feature
                            # column list but restricts to available cols.
                            lambda _full: X_group
                        ),
                        rel_series,
                        drop_na_target=True,
                    )
                except ValueError as exc:
                    print(f"    [{benchmark}] get_X_y_relative error: {exc}")
                    continue

                # Ensure X_aligned only has the available columns.
                X_aligned = X_aligned[[c for c in available if c in X_aligned.columns]]
                n_obs = len(y_aligned)

                for model_type in MODEL_TYPES:
                    print(
                        f"    [{benchmark}] [{model_type}] "
                        f"n_obs={n_obs}, n_feat={X_aligned.shape[1]} …",
                        end=" ",
                        flush=True,
                    )
                    try:
                        result = run_wfo(
                            X_aligned,
                            y_aligned,
                            model_type=model_type,  # type: ignore[arg-type]
                            target_horizon_months=horizon,
                            benchmark=benchmark,
                        )
                        ic = result.information_coefficient
                        hit_rate = result.hit_rate
                        mae = result.mean_absolute_error
                        y_true = pd.Series(result.y_true_all)
                        y_hat = pd.Series(result.y_hat_all)
                        oos_r2 = compute_oos_r_squared(y_hat, y_true)
                        print(f"IC={ic:.4f}, hit={hit_rate:.3f}, OOS-R²={oos_r2:.4f}")
                    except (ValueError, RuntimeError) as exc:
                        print(f"FAILED: {exc}")
                        ic = hit_rate = mae = oos_r2 = float("nan")

                    records.append({
                        "feature_group": group_label,
                        "benchmark": benchmark,
                        "model_type": model_type,
                        "horizon_months": horizon,
                        "n_obs": n_obs,
                        "n_features": X_aligned.shape[1],
                        "ic": ic,
                        "hit_rate": hit_rate,
                        "mae": mae,
                        "oos_r2": oos_r2,
                    })

    results_df = pd.DataFrame(records)

    # Write CSV.
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.today().strftime("%Y%m%d")
    csv_path = os.path.join(output_dir, f"feature_ablation_{date_str}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults written to: {csv_path}")

    return results_df


def print_summary(results_df: pd.DataFrame) -> None:
    """Print a summary table and flag net-negative groups.

    A group is net-negative if its mean IC is lower than the prior group's
    mean IC (averaged across all benchmarks and model types).
    """
    if results_df.empty:
        print("\n[Summary] No results to display.")
        return

    # Mean IC and hit rate per group, averaged across benchmarks and models.
    summary = (
        results_df.groupby("feature_group")[["ic", "hit_rate"]]
        .mean()
        .reindex([g for g in GROUP_ORDER if g in results_df["feature_group"].values])
    )

    print("\n" + "=" * 70)
    print("FEATURE ABLATION SUMMARY (mean across all benchmarks and models)")
    print("=" * 70)
    print(f"{'Group':<22}  {'Mean IC':>10}  {'Mean Hit Rate':>14}")
    print("-" * 50)

    net_negative_groups: list[str] = []
    prev_ic: float | None = None

    for group_label, row in summary.iterrows():
        mean_ic = row["ic"]
        mean_hr = row["hit_rate"]
        flag = ""
        if prev_ic is not None and mean_ic < prev_ic:
            flag = "  [NET-NEGATIVE]"
            net_negative_groups.append(str(group_label))
        print(f"  {group_label:<20}  {mean_ic:>+10.4f}  {mean_hr:>13.3%}{flag}")
        prev_ic = mean_ic

    print("-" * 50)

    if net_negative_groups:
        print("\n[WARNING] The following groups DECREASE mean IC vs the prior group:")
        for g in net_negative_groups:
            print(f"   - {g}")

        # Identify the new features introduced by the net-negative groups.
        to_drop: list[str] = []
        group_names = list(FEATURE_GROUPS.keys())
        for g in net_negative_groups:
            idx = group_names.index(g)
            if idx > 0:
                prior = set(FEATURE_GROUPS[group_names[idx - 1]])
                new_cols = [c for c in FEATURE_GROUPS[g] if c not in prior]
                to_drop.extend(new_cols)

        if to_drop:
            print("\nRecommendation: add the following to config.FEATURES_TO_DROP:")
            for col in sorted(set(to_drop)):
                print(f"   \"{col}\",")
    else:
        print("\n[OK] All feature groups are non-negative - no features flagged.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature ablation backtest for the PGR vesting model.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(DEFAULT_BENCHMARKS),
        help=(
            "Comma-separated list of ETF benchmarks to test against. "
            f"Default: {','.join(DEFAULT_BENCHMARKS)}"
        ),
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help=(
            "Comma-separated target horizons in months. "
            f"Default: {','.join(str(h) for h in DEFAULT_HORIZONS)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output CSV. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    print("PGR Feature Ablation Backtest — v7.0")
    print(f"Benchmarks : {benchmarks}")
    print(f"Horizons   : {horizons}M")
    print(f"Output dir : {args.output_dir}")

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    try:
        results_df = run_ablation(
            conn=conn,
            benchmarks=benchmarks,
            horizons=horizons,
            output_dir=args.output_dir,
        )
        print_summary(results_df)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
