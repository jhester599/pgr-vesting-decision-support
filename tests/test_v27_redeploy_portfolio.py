from __future__ import annotations

import pandas as pd

from src.research.v27 import (
    portfolio_rows_to_frame,
    recommend_redeploy_portfolio,
    render_redeploy_portfolio_markdown_lines,
    v27_benchmark_pruning_review,
)


def _sample_signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark": "VOO",
                "predicted_relative_return": -0.04,
                "ic": 0.08,
                "hit_rate": 0.56,
                "prob_outperform": 0.42,
                "calibrated_prob_outperform": 0.40,
            },
            {
                "benchmark": "VGT",
                "predicted_relative_return": -0.06,
                "ic": 0.12,
                "hit_rate": 0.58,
                "prob_outperform": 0.39,
                "calibrated_prob_outperform": 0.37,
            },
            {
                "benchmark": "SCHD",
                "predicted_relative_return": -0.02,
                "ic": 0.02,
                "hit_rate": 0.54,
                "prob_outperform": 0.48,
                "calibrated_prob_outperform": 0.47,
            },
            {
                "benchmark": "VXUS",
                "predicted_relative_return": -0.03,
                "ic": 0.06,
                "hit_rate": 0.57,
                "prob_outperform": 0.45,
                "calibrated_prob_outperform": 0.43,
            },
            {
                "benchmark": "VWO",
                "predicted_relative_return": -0.01,
                "ic": 0.05,
                "hit_rate": 0.55,
                "prob_outperform": 0.49,
                "calibrated_prob_outperform": 0.48,
            },
            {
                "benchmark": "BND",
                "predicted_relative_return": 0.01,
                "ic": 0.15,
                "hit_rate": 0.64,
                "prob_outperform": 0.55,
                "calibrated_prob_outperform": 0.53,
            },
        ]
    ).set_index("benchmark")


def _sample_scoreboard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"benchmark": "VOO", "corr_to_pgr": 0.14, "diversification_score": 0.46},
            {"benchmark": "VGT", "corr_to_pgr": 0.36, "diversification_score": 0.36},
            {"benchmark": "SCHD", "corr_to_pgr": 0.29, "diversification_score": 0.40},
            {"benchmark": "VXUS", "corr_to_pgr": 0.28, "diversification_score": 0.41},
            {"benchmark": "VWO", "corr_to_pgr": 0.30, "diversification_score": 0.38},
            {"benchmark": "BND", "corr_to_pgr": 0.04, "diversification_score": 0.52},
        ]
    )


def test_recommend_redeploy_portfolio_returns_bounded_weights() -> None:
    portfolio = recommend_redeploy_portfolio(
        signals=_sample_signals(),
        diversification_scoreboard=_sample_scoreboard(),
        recommendation_mode_label="MONITORING-ONLY",
    )

    rows = portfolio["rows"]
    assert rows
    total_weight = sum(row.allocation for row in rows)
    assert abs(total_weight - 1.0) < 1e-9
    assert sum(row.allocation for row in rows if row.ticker != "BND") >= 0.90
    assert rows[0].ticker in {"VOO", "VGT"}


def test_render_redeploy_portfolio_markdown_lines_includes_table() -> None:
    portfolio = recommend_redeploy_portfolio(
        signals=_sample_signals(),
        diversification_scoreboard=_sample_scoreboard(),
        recommendation_mode_label="ACTIONABLE",
    )

    rendered = "\n".join(render_redeploy_portfolio_markdown_lines(portfolio))
    assert "## Suggested Redeploy Portfolio" in rendered
    assert "| Fund | Allocation | Sleeve |" in rendered
    assert "VOO" in rendered
    assert "The current project universe does not yet include a dedicated small-cap ETF" in rendered


def test_portfolio_rows_to_frame_preserves_allocations() -> None:
    portfolio = recommend_redeploy_portfolio(
        signals=_sample_signals(),
        diversification_scoreboard=_sample_scoreboard(),
        recommendation_mode_label="DEFER-TO-TAX-DEFAULT",
    )
    frame = portfolio_rows_to_frame(portfolio, as_of="2026-04-05")
    assert not frame.empty
    assert abs(frame["allocation"].sum() - sum(row.allocation for row in portfolio["rows"])) < 1e-12
    assert set(frame["fund"]) == {row.ticker for row in portfolio["rows"]}


def test_benchmark_pruning_review_marks_contextual_only() -> None:
    review = pd.DataFrame(v27_benchmark_pruning_review())
    assert review.loc[review["benchmark"] == "VFH", "status"].iloc[0] == "contextual_only"
    assert review.loc[review["benchmark"] == "VOO", "status"].iloc[0] == "keep_for_redeploy"
