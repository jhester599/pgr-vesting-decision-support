from __future__ import annotations

from datetime import date

from src.reporting.decision_rendering import build_executive_summary_lines
from src.research.v29 import benchmark_role_for_ticker, build_confidence_snapshot


class _FakeCPCV:
    def __init__(self, verdict: str) -> None:
        self.stability_verdict = verdict


def test_benchmark_role_for_ticker_distinguishes_buyable_and_contextual() -> None:
    assert benchmark_role_for_ticker("VOO")["role"] == "Buy candidate"
    assert benchmark_role_for_ticker("VFH")["role"] == "Context only"
    assert benchmark_role_for_ticker("GLD")["role"] == "Forecast only"


def test_build_confidence_snapshot_counts_pass_fail() -> None:
    snapshot = build_confidence_snapshot(
        mean_ic=0.10,
        mean_hr=0.56,
        aggregate_health={"oos_r2": -0.10},
        representative_cpcv=_FakeCPCV("FAIL"),
    )
    assert snapshot["pass_count"] == 2
    assert snapshot["fail_count"] == 2
    statuses = {row["check"]: row["status"] for row in snapshot["rows"]}
    assert statuses["Mean IC"] == "PASS"
    assert statuses["Mean hit rate"] == "PASS"
    assert statuses["Aggregate OOS R^2"] == "FAIL"
    assert statuses["Representative CPCV"] == "FAIL"


def test_executive_summary_avoids_contradictory_direction_phrase() -> None:
    lines = build_executive_summary_lines(
        as_of=date(2026, 4, 4),
        consensus="OUTPERFORM",
        confidence_tier="LOW",
        mean_predicted=-0.0031,
        sell_pct=0.5,
        recommendation_mode={
            "mode": "defer-to-tax-default",
            "label": "DEFER-TO-TAX-DEFAULT",
            "summary": "Weak model quality.",
            "action_note": "Use the default diversification rule.",
        },
        aggregate_health={"oos_r2": -0.7883, "nw_ic": 0.1428, "agg_hit": 0.576},
        previous_summary=None,
        next_vest_summary=None,
    )
    joined = "\n".join(lines)
    assert "outperform the benchmark set by -0.31%" not in joined
    assert "Consensus signal is OUTPERFORM, but the average relative-return forecast is -0.31%" in joined
