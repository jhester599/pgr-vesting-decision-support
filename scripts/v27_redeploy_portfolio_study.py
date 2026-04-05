"""v27 redeploy-portfolio study for monthly sell-proceeds recommendations."""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
import sys

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.database.db_client import get_connection
from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.processing.feature_engineering import build_feature_matrix_from_db
from src.processing.multi_total_return import build_etf_monthly_returns, load_relative_return_matrix
from src.research.diversification import score_benchmarks_against_pgr
from src.research.evaluation import reconstruct_ensemble_oos_predictions
from src.research.v27 import (
    summarize_dynamic_portfolio,
    simulate_dynamic_portfolio,
    v27_benchmark_pruning_review,
)


RESULTS_DIR = Path("results") / "v27"


BASE_PORTFOLIOS: dict[str, dict[str, float]] = {
    "chatgpt_proxy_90_10": {
        "VTI": 0.45,
        "VXUS": 0.25,
        "VWO": 0.10,
        "SCHD": 0.10,
        "BND": 0.10,
    },
    "gemini_proxy_100_equity": {
        "VOO": 0.35,
        "VGT": 0.30,
        "VXUS": 0.15,
        "VWO": 0.10,
        "SCHD": 0.10,
    },
    "balanced_pref_95_5": {
        "VOO": 0.40,
        "VGT": 0.20,
        "SCHD": 0.15,
        "VXUS": 0.10,
        "VWO": 0.10,
        "BND": 0.05,
    },
    "hybrid_equity_95_5": {
        "VOO": 0.38,
        "VGT": 0.22,
        "SCHD": 0.15,
        "VXUS": 0.12,
        "VWO": 0.08,
        "BND": 0.05,
    },
}

TILT_VALUES: tuple[float, ...] = (0.25, 0.35, 0.45)


def _build_historical_signal_inputs(
    conn,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = build_feature_matrix_from_db(conn, force_refresh=True)
    relative_return_matrix = pd.DataFrame(
        {
            ticker: load_relative_return_matrix(conn, ticker, 6)
            for ticker in tickers
            if not load_relative_return_matrix(conn, ticker, 6).empty
        }
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        ensemble_results = run_ensemble_benchmarks(X, relative_return_matrix, target_horizon_months=6)

    predicted_panel = pd.DataFrame(
        {
            ticker: reconstruct_ensemble_oos_predictions(result)[0].rename(ticker)
            for ticker, result in ensemble_results.items()
        }
    ).sort_index()
    fund_return_panel = pd.DataFrame(
        {
            ticker: build_etf_monthly_returns(conn, ticker, 1).rename(ticker)
            for ticker in predicted_panel.columns
        }
    ).sort_index()
    signal_metric_frame = pd.DataFrame(
        [
            {
                "benchmark": ticker,
                "ic": result.mean_ic,
                "hit_rate": result.mean_hit_rate,
                "mae": result.mean_mae,
                "n_obs": len(reconstruct_ensemble_oos_predictions(result)[0]),
            }
            for ticker, result in ensemble_results.items()
        ]
    )
    diversification_scoreboard = score_benchmarks_against_pgr(conn, list(predicted_panel.columns))
    return predicted_panel, fund_return_panel, signal_metric_frame, diversification_scoreboard


def _selection_score(row: pd.Series) -> float:
    return float(
        row["annualized_return"]
        - (0.20 * row["annualized_vol"])
        - (0.08 * max(row["corr_to_pgr"], 0.0))
        - (0.15 * row["mean_turnover"])
    )


def _choose_recommended_strategy(summary: pd.DataFrame) -> pd.Series:
    dynamic = summary[summary["tilt_strength"] > 0].copy()
    if dynamic.empty:
        raise ValueError("No dynamic v27 strategies were evaluated.")

    top_return = float(dynamic["annualized_return"].max())
    bond_preferred = dynamic[
        (dynamic["bond_weight"] >= 0.05)
        & (dynamic["bond_weight"] <= 0.10)
        & (dynamic["annualized_return"] >= top_return - 0.03)
    ].copy()
    candidate_pool = bond_preferred if not bond_preferred.empty else dynamic
    candidate_pool["selection_score"] = candidate_pool.apply(_selection_score, axis=1)
    return candidate_pool.sort_values(
        by=["selection_score", "annualized_return"],
        ascending=[False, False],
    ).iloc[0]


def run_v27_study() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = get_connection(config.DB_PATH)
    all_tickers = sorted({ticker for weights in BASE_PORTFOLIOS.values() for ticker in weights})
    predicted_panel, fund_return_panel, signal_metric_frame, diversification_scoreboard = _build_historical_signal_inputs(
        conn,
        all_tickers,
    )
    pgr_monthly_returns = build_etf_monthly_returns(conn, "PGR", 1).sort_index()

    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for name, weights in BASE_PORTFOLIOS.items():
        stock_weight = float(sum(weight for ticker, weight in weights.items() if ticker != "BND"))
        bond_weight = float(weights.get("BND", 0.0))
        for tilt_strength in (0.0, *TILT_VALUES):
            detail = simulate_dynamic_portfolio(
                predicted_relative_panel=predicted_panel,
                monthly_return_panel=fund_return_panel,
                signal_metric_frame=signal_metric_frame,
                diversification_scoreboard=diversification_scoreboard,
                base_weights=weights,
                tilt_strength=tilt_strength,
            )
            stats = summarize_dynamic_portfolio(detail, pgr_monthly_returns)
            summary_rows.append(
                {
                    "strategy_name": name,
                    "tilt_strength": tilt_strength,
                    "stock_weight": stock_weight,
                    "bond_weight": bond_weight,
                    **stats,
                }
            )
            labeled = detail.copy()
            labeled["strategy_name"] = name
            labeled["tilt_strength"] = tilt_strength
            detail_frames.append(labeled)

    summary = pd.DataFrame(summary_rows).sort_values(
        by=["annualized_return", "corr_to_pgr"],
        ascending=[False, True],
    )
    recommended = _choose_recommended_strategy(summary)
    summary["recommended"] = (
        (summary["strategy_name"] == recommended["strategy_name"])
        & (summary["tilt_strength"] == recommended["tilt_strength"])
    )
    detail_df = pd.concat(detail_frames, ignore_index=True)
    pruning_df = pd.DataFrame(v27_benchmark_pruning_review())
    return summary, detail_df, pruning_df


def write_outputs(summary: pd.DataFrame, detail_df: pd.DataFrame, pruning_df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    summary.to_csv(RESULTS_DIR / f"v27_redeploy_backtest_summary_{stamp}.csv", index=False)
    detail_df.to_csv(RESULTS_DIR / f"v27_redeploy_backtest_detail_{stamp}.csv", index=False)
    pruning_df.to_csv(RESULTS_DIR / f"v27_benchmark_pruning_review_{stamp}.csv", index=False)


def main() -> None:
    summary, detail_df, pruning_df = run_v27_study()
    write_outputs(summary, detail_df, pruning_df)
    recommended = summary.loc[summary["recommended"]].iloc[0]
    print("v27 redeploy study complete.")
    print(
        "Recommended strategy:",
        recommended["strategy_name"],
        f"(tilt={recommended['tilt_strength']:.2f})",
    )


if __name__ == "__main__":
    main()
