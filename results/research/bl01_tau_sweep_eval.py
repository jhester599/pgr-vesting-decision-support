"""BL-01: Monte Carlo parameter sweep for Black-Litterman tau / risk_aversion tuning.

Run:  python results/research/bl01_tau_sweep_eval.py
Writes: results/research/bl01_tau_candidate.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.black_litterman import build_bl_weights
from src.models.multi_benchmark_wfo import EnsembleWFOResult
from src.models.wfo_engine import FoldResult, WFOResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BENCHMARKS: list[str] = ["VOO", "VXUS", "VWO", "VMBS", "BND", "GLD", "DBC", "VDE"]
TAU_GRID: list[float] = [0.01, 0.025, 0.05, 0.10, 0.20]
RISK_AVERSION_GRID: list[float] = [1.5, 2.0, 2.5, 3.0, 4.0]
N_SCENARIOS: int = 50
N_MONTHS: int = 60
INCUMBENT_TAU: float = 0.05
INCUMBENT_RA: float = 2.5
WIN_THRESHOLD_RANK_CORR: float = 0.05

CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "bl01_tau_candidate.json"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (numpy only, no scipy).

    Note: uses the exact Spearman formula which assumes no ties. With ties,
    argsort breaks them arbitrarily. In the sweep context IC values are drawn
    from continuous uniform distributions making ties essentially impossible.
    """
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    return float(1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1)))


def _make_signal(
    benchmark: str,
    mean_ic: float,
    mean_mae: float = 0.05,
    n_preds: int = 6,
    seed: int = 0,
) -> EnsembleWFOResult:
    """Build a minimal EnsembleWFOResult for use in the BL sweep.

    The fold y_hat is set to mean_ic * 0.12 so that the BL view Q vector
    reflects signal quality ordering (higher IC → higher predicted return).
    """
    rng = np.random.default_rng(seed)
    y_hat = np.full(n_preds, mean_ic * 0.12)
    y_true = y_hat + rng.normal(0, mean_mae, size=n_preds)
    fold = FoldResult(
        fold_idx=0,
        train_start=pd.Timestamp("2019-01-31"),
        train_end=pd.Timestamp("2022-12-31"),
        test_start=pd.Timestamp("2023-01-31"),
        test_end=pd.Timestamp("2023-06-30"),
        y_true=y_true,
        y_hat=y_hat,
        optimal_alpha=0.01,
        feature_importances={},
        n_train=48,
        n_test=n_preds,
    )
    wfo = WFOResult(
        folds=[fold],
        benchmark=benchmark,
        target_horizon=6,
        model_type="elasticnet",
    )
    return EnsembleWFOResult(
        benchmark=benchmark,
        target_horizon=6,
        mean_ic=mean_ic,
        mean_hit_rate=max(0.5 + mean_ic * 2, 0.0),
        mean_mae=mean_mae,
        model_results={"elasticnet": wfo},
    )


def _make_scenario(
    seed: int,
    benchmarks: list[str],
    n_months: int = 60,
) -> tuple[pd.DataFrame, dict[str, EnsembleWFOResult]]:
    """Generate synthetic monthly returns + ensemble signals for one MC scenario."""
    rng = np.random.default_rng(seed)
    n = len(benchmarks)
    base_corr = 0.40 * np.ones((n, n)) + 0.60 * np.eye(n)
    monthly_std = 0.04
    cov = (monthly_std ** 2) * base_corr
    idx = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    returns_arr = rng.multivariate_normal(np.full(n, 0.008), cov, size=n_months)
    returns_df = pd.DataFrame(returns_arr, index=idx, columns=benchmarks)
    signals: dict[str, EnsembleWFOResult] = {}
    for i, bm in enumerate(benchmarks):
        ic = float(rng.uniform(-0.05, 0.20))
        mae = float(rng.uniform(0.04, 0.08))
        signals[bm] = _make_signal(bm, ic, mae, seed=seed * 100 + i)
    return returns_df, signals


def _compute_ic_rank_correlation(
    weights: dict[str, float],
    signals: dict[str, EnsembleWFOResult],
) -> float:
    """Spearman rank correlation between BL weights and benchmark IC values."""
    common = sorted(set(weights) & set(signals))
    if len(common) < 3:
        return 0.0
    w = np.array([weights[bm] for bm in common])
    ic = np.array([signals[bm].mean_ic for bm in common])
    return _rank_corr(w, ic)


def run_tau_sweep(
    tau_grid: list[float],
    ra_grid: list[float],
    n_scenarios: int,
    benchmarks: list[str],
    n_months: int = 60,
) -> list[dict[str, Any]]:
    """Run the full (tau × risk_aversion) grid sweep.

    Returns list of result rows, one per (tau, risk_aversion) combination.
    Row keys: tau, risk_aversion, mean_rank_corr, fallback_rate, n_valid_scenarios, n_scenarios.
    """
    rows: list[dict[str, Any]] = []
    for tau in tau_grid:
        for ra in ra_grid:
            rank_corrs: list[float] = []
            n_fallbacks: int = 0
            for seed in range(n_scenarios):
                returns_df, signals = _make_scenario(seed, benchmarks, n_months)
                weights, diag = build_bl_weights(
                    signals,
                    returns_df,
                    risk_aversion=ra,
                    tau=tau,
                    risk_free_rate=0.0,
                    return_diagnostics=True,
                )
                if diag.fallback_used:
                    n_fallbacks += 1
                    continue
                rank_corrs.append(_compute_ic_rank_correlation(weights, signals))
            rows.append({
                "tau": tau,
                "risk_aversion": ra,
                "mean_rank_corr": float(np.mean(rank_corrs)) if rank_corrs else float("nan"),
                "fallback_rate": n_fallbacks / n_scenarios,
                "n_valid_scenarios": len(rank_corrs),
                "n_scenarios": n_scenarios,
            })
    return rows


def select_winner(
    rows: list[dict[str, Any]],
    incumbent_tau: float,
    incumbent_ra: float,
    win_threshold: float,
) -> dict[str, Any]:
    """Select the winning (tau, risk_aversion) from sweep rows."""
    incumbent_row = next(
        (r for r in rows if r["tau"] == incumbent_tau and r["risk_aversion"] == incumbent_ra),
        {"tau": incumbent_tau, "risk_aversion": incumbent_ra, "mean_rank_corr": 0.0, "fallback_rate": 0.0},
    )
    incumbent_corr = incumbent_row["mean_rank_corr"]
    eligible = [r for r in rows if r["fallback_rate"] < 0.50]
    all_filtered = not eligible
    if all_filtered:
        eligible = rows
    best_row = max(eligible, key=lambda r: r["mean_rank_corr"])
    best_corr = best_row["mean_rank_corr"]
    delta = best_corr - incumbent_corr
    if delta >= win_threshold and (best_row["tau"] != incumbent_tau or best_row["risk_aversion"] != incumbent_ra):
        winner_tau = best_row["tau"]
        winner_ra = best_row["risk_aversion"]
        recommendation = "update_bl_params"
    else:
        winner_tau = incumbent_tau
        winner_ra = incumbent_ra
        recommendation = "keep_incumbent"
    return {
        "bl_tau_winner": winner_tau,
        "bl_risk_aversion_winner": winner_ra,
        "incumbent_tau": incumbent_tau,
        "incumbent_risk_aversion": incumbent_ra,
        "winner_rank_corr": best_corr if recommendation == "update_bl_params" else incumbent_corr,
        "incumbent_rank_corr": incumbent_corr,
        "delta_rank_corr": round(delta, 6),
        "win_threshold": win_threshold,
        "recommendation": recommendation,
        "quality_filter_bypassed": all_filtered,
        "rows": rows,
    }


def run_bl01_sweep(
    candidate_path: Path = CANDIDATE_PATH,
    tau_grid: list[float] | None = None,
    ra_grid: list[float] | None = None,
    n_scenarios: int = N_SCENARIOS,
) -> dict[str, Any]:
    """Run the full BL-01 sweep and write the candidate JSON."""
    tau_grid = tau_grid if tau_grid is not None else TAU_GRID
    ra_grid = ra_grid if ra_grid is not None else RISK_AVERSION_GRID
    print(f"Running BL-01 sweep: {len(tau_grid)}×{len(ra_grid)} grid × {n_scenarios} scenarios...")
    rows = run_tau_sweep(tau_grid, ra_grid, n_scenarios, BENCHMARKS)
    candidate = select_winner(rows, INCUMBENT_TAU, INCUMBENT_RA, WIN_THRESHOLD_RANK_CORR)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    print(f"Candidate written to {candidate_path}")
    print(f"  Recommendation: {candidate['recommendation']}")
    print(f"  Winner:   tau={candidate['bl_tau_winner']}, risk_aversion={candidate['bl_risk_aversion_winner']}")
    print(f"  Incumbent tau={candidate['incumbent_tau']}, risk_aversion={candidate['incumbent_risk_aversion']}")
    print(f"  Delta rank_corr: {candidate['delta_rank_corr']:+.4f}  (threshold={WIN_THRESHOLD_RANK_CORR})")
    return candidate


if __name__ == "__main__":
    run_bl01_sweep()
