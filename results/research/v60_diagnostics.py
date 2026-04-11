"""v60 - Clark-West diagnostics, MSE decomposition, and CE gain."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from config.features import MODEL_FEATURE_OVERRIDES
from src.models.evaluation import reconstruct_ensemble_oos_predictions
from src.models.forecast_diagnostics import compute_clark_west_result
from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.research.v37_utils import (
    BENCHMARKS,
    RESULTS_DIR,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    print_footer,
    print_header,
    save_results,
)


def mse_decompose(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    """Split MSE into simple bias and prediction-variance proportions."""
    mse = float(np.mean((y_hat - y_true) ** 2))
    bias_sq = float((y_hat.mean() - y_true.mean()) ** 2)
    var_pred = float(np.var(y_hat))
    return {
        "mse": mse,
        "bias_sq": bias_sq,
        "var_pred": var_pred,
        "var_pct": var_pred / mse if mse > 0 else np.nan,
        "bias_pct": bias_sq / mse if mse > 0 else np.nan,
    }


def certainty_equivalent_gain(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    gamma: float = 2.0,
    annualize: float = 2.0,
) -> float:
    """Estimate CE gain of a sign-based strategy vs a naive half-long stance."""
    model_signal = np.sign(y_hat)
    model_returns = model_signal * y_true
    naive_returns = 0.5 * y_true

    def crra_ce(returns: np.ndarray) -> float:
        mu = float(np.mean(returns))
        sigma_sq = float(np.var(returns))
        return mu - 0.5 * gamma * sigma_sq

    return annualize * (crra_ce(model_returns) - crra_ce(naive_returns))


def main() -> None:
    """Run v60 diagnostics on the production-style v37 ensemble."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rel_matrix: dict[str, pd.Series] = {}
        for benchmark in BENCHMARKS:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if not rel_series.empty:
                rel_matrix[benchmark] = rel_series

        ensemble_results = run_ensemble_benchmarks(
            feature_df,
            pd.DataFrame(rel_matrix),
            target_horizon_months=6,
            model_feature_overrides=MODEL_FEATURE_OVERRIDES,
        )

        print_header("v60", "Clark-West Test + Evaluation Diagnostics")
        print(
            f"\n  {'Benchmark':<10}  {'CW_stat':>8}  {'p-value':>8}  {'Sig?':>5}  "
            f"{'Bias%':>7}  {'Var%':>7}  {'CE_gain':>9}"
        )

        rows: list[dict[str, object]] = []
        pooled_y_true: list[float] = []
        pooled_y_hat: list[float] = []
        for benchmark in BENCHMARKS:
            ens_result = ensemble_results.get(benchmark)
            if ens_result is None:
                continue
            y_hat_series, y_true_series = reconstruct_ensemble_oos_predictions(ens_result)
            y_hat = y_hat_series.to_numpy(dtype=float)
            y_true = y_true_series.to_numpy(dtype=float)
            pooled_y_true.extend(y_true.tolist())
            pooled_y_hat.extend(y_hat.tolist())

            cw_summary = compute_clark_west_result(
                pd.Series(y_hat),
                pd.Series(y_true),
                lags=5,
            )
            mse_parts = mse_decompose(y_true, y_hat)
            ce_gain = certainty_equivalent_gain(y_true, y_hat)
            sig = (
                "***" if cw_summary.p_value < 0.01 else
                ("**" if cw_summary.p_value < 0.05 else ("*" if cw_summary.p_value < 0.10 else ""))
            )

            print(
                f"  {benchmark:<10}  {cw_summary.t_stat:>8.3f}  {cw_summary.p_value:>8.4f}  {sig:>5}  "
                f"{mse_parts['bias_pct']:>6.1%}  {mse_parts['var_pct']:>6.1%}  {ce_gain:>9.4f}"
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "cw_stat": cw_summary.t_stat,
                    "cw_p_value": cw_summary.p_value,
                    "mean_cw": cw_summary.mean_adjusted_differential,
                    "ce_gain": ce_gain,
                    **{f"mse_{key}": value for key, value in mse_parts.items()},
                    "version": "v60",
                }
            )

        pooled_true = np.asarray(pooled_y_true)
        pooled_hat = np.asarray(pooled_y_hat)
        cw_summary = compute_clark_west_result(
            pd.Series(pooled_hat),
            pd.Series(pooled_true),
            lags=5,
        )
        pooled_mse = mse_decompose(pooled_true, pooled_hat)
        pooled_ce = certainty_equivalent_gain(pooled_true, pooled_hat)
        sig = (
            "***" if cw_summary.p_value < 0.01 else
            ("**" if cw_summary.p_value < 0.05 else ("*" if cw_summary.p_value < 0.10 else "no"))
        )

        print(
            f"\n  {'POOLED':<10}  {cw_summary.t_stat:>8.3f}  {cw_summary.p_value:>8.4f}  {sig:>5}  "
            f"{pooled_mse['bias_pct']:>6.1%}  {pooled_mse['var_pct']:>6.1%}  {pooled_ce:>9.4f}"
        )
        print("\n  Interpretation:")
        print("    CW p < 0.05 -> model adds predictive value beyond the historical mean.")
        print("    Higher Var% than Bias% -> error is dominated more by spread than mean offset.")
        print("    Positive CE gain -> signal may still add economic value for a risk-averse holder.")
        print_footer()

        rows.append(
            {
                "benchmark": "POOLED",
                "cw_stat": cw_summary.t_stat,
                "cw_p_value": cw_summary.p_value,
                "mean_cw": cw_summary.mean_adjusted_differential,
                "ce_gain": pooled_ce,
                **{f"mse_{key}": value for key, value in pooled_mse.items()},
                "version": "v60",
            }
        )
        save_results(pd.DataFrame(rows), "v60_diagnostics_results.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
