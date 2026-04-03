from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.monthly_decision import (
    _build_executive_summary_lines,
    _build_vest_decision_lines,
    _determine_recommendation_mode,
)
from src.models.wfo_engine import CPCVResult
from src.tax.capital_gains import ThreeScenarioResult, TaxScenario


def _cpcv(verdict: str) -> CPCVResult:
    path_ics = [0.05] * 28
    if verdict == "FAIL":
        path_ics = [-0.02] * 20 + [0.01] * 8
    elif verdict == "MARGINAL":
        path_ics = [0.03] * 15 + [-0.01] * 13
    return CPCVResult(
        model_type="elasticnet",
        benchmark="VTI",
        n_splits=28,
        n_paths=28,
        path_ics=path_ics,
        mean_ic=0.05,
        ic_std=0.01,
        split_ics=[],
    )


def test_determine_recommendation_mode_defers_when_quality_is_weak() -> None:
    mode = _determine_recommendation_mode(
        consensus="OUTPERFORM",
        mean_predicted=0.08,
        mean_ic=0.01,
        mean_hr=0.50,
        aggregate_health={"oos_r2": -0.05},
        representative_cpcv=_cpcv("FAIL"),
    )
    assert mode["mode"] == "defer-to-tax-default"
    assert mode["sell_pct"] == 0.50


def test_determine_recommendation_mode_actionable_when_all_quality_checks_pass() -> None:
    mode = _determine_recommendation_mode(
        consensus="OUTPERFORM",
        mean_predicted=0.18,
        mean_ic=0.08,
        mean_hr=0.58,
        aggregate_health={"oos_r2": 0.03},
        representative_cpcv=_cpcv("GOOD"),
    )
    assert mode["mode"] == "actionable"
    assert mode["sell_pct"] == 0.25


def test_build_executive_summary_lines_mentions_quality_and_change_trigger() -> None:
    lines = _build_executive_summary_lines(
        as_of=date(2026, 4, 2),
        consensus="NEUTRAL",
        confidence_tier="LOW",
        mean_predicted=-0.02,
        sell_pct=0.50,
        recommendation_mode={
            "mode": "defer-to-tax-default",
            "label": "DEFER-TO-TAX-DEFAULT",
            "summary": "Model quality is too weak to justify a prediction-led vesting action.",
            "action_note": "Use the default rule.",
        },
        aggregate_health={"oos_r2": -0.01, "nw_ic": 0.02, "agg_hit": 0.51},
        previous_summary={
            "as_of": "2026-03-20",
            "consensus": "NEUTRAL",
            "predicted": "+1.30%",
            "mean_ic": "-0.0056",
        },
        next_vest_summary={"vest_date": date(2026, 7, 17), "rsu_type": "performance"},
    )
    combined = "\n".join(lines)
    assert "What changed since last month" in combined
    assert "How trustworthy it is" in combined
    assert "What would change the recommendation" in combined


def test_build_vest_decision_lines_renders_scenario_table() -> None:
    scenario_result = ThreeScenarioResult(
        vest_date=date(2026, 7, 17),
        rsu_type="performance",
        current_price=210.0,
        cost_basis_per_share=150.0,
        shares=8.0,
        stcg_ltcg_breakeven=0.2125,
        days_to_ltcg=366,
        recommended_scenario="SELL_NOW_STCG",
        scenarios=[
            TaxScenario("SELL_NOW_STCG", date(2026, 7, 17), 0.37, 0, 0.0, 210.0, 1680.0, 177.6, 1502.4, 0.0, 1.0, "sell now"),
            TaxScenario("HOLD_TO_LTCG", date(2027, 7, 18), 0.20, 366, 0.12, 235.2, 1881.6, 136.32, 1745.28, 0.2125, 0.55, "hold"),
            TaxScenario("HOLD_FOR_LOSS", date(2027, 1, 17), 0.37, 184, -0.10, 189.0, 1512.0, -62.16, 1574.16, 0.0, 0.45, "harvest"),
        ],
    )
    lines = _build_vest_decision_lines(
        {
            "vest_date": date(2026, 7, 17),
            "rsu_type": "performance",
            "current_price": 210.0,
            "shares": 8.0,
            "avg_basis": 150.0,
            "scenario": scenario_result,
        },
        {
            "mode": "defer-to-tax-default",
            "label": "DEFER-TO-TAX-DEFAULT",
            "action_note": "Use the default rule.",
        },
        0.50,
    )
    combined = "\n".join(lines)
    assert "## Next Vest Decision" in combined
    assert "SELL_NOW_STCG" in combined
    assert "informational only" in combined


def test_build_vest_decision_lines_uses_provisional_winner_label_when_actionable() -> None:
    scenario_result = ThreeScenarioResult(
        vest_date=date(2026, 7, 17),
        rsu_type="performance",
        current_price=210.0,
        cost_basis_per_share=150.0,
        shares=8.0,
        stcg_ltcg_breakeven=0.2125,
        days_to_ltcg=366,
        recommended_scenario="HOLD_TO_LTCG",
        scenarios=[
            TaxScenario("SELL_NOW_STCG", date(2026, 7, 17), 0.37, 0, 0.0, 210.0, 1680.0, 177.6, 1502.4, 0.0, 1.0, "sell now"),
            TaxScenario("HOLD_TO_LTCG", date(2027, 7, 18), 0.20, 366, 0.12, 235.2, 1881.6, 136.32, 1745.28, 0.2125, 0.55, "hold"),
            TaxScenario("HOLD_FOR_LOSS", date(2027, 1, 17), 0.37, 184, -0.10, 189.0, 1512.0, -62.16, 1574.16, 0.0, 0.45, "harvest"),
        ],
    )
    lines = _build_vest_decision_lines(
        {
            "vest_date": date(2026, 7, 17),
            "rsu_type": "performance",
            "current_price": 210.0,
            "shares": 8.0,
            "avg_basis": 150.0,
            "scenario": scenario_result,
        },
        {
            "mode": "actionable",
            "label": "ACTIONABLE",
            "action_note": "Use the point forecast.",
        },
        0.25,
    )
    combined = "\n".join(lines)
    assert "Provisional scenario winner" in combined
    assert "aligns with the current point forecast" in combined
