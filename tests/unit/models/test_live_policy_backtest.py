"""Review 2026-09-25, step 6 (WP8, F20): the live ACTIONABLE sell-% mapping.

The fixture is the realised-only OOS panel of the production ensemble as of
2026-09-21 (``scripts/export_oos_panel.py`` on a copy of the DB after steps
1-5): 186 monthly OOS dates, 8 benchmarks. The policy-regression test replays
the exact live consensus and mapping over it and requires the mapping not to
lose to the always-50 % tax default.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.reporting.decision_rendering import (
    determine_recommendation_mode,
    sell_pct_from_consensus,
)

FIXTURE = Path(__file__).parent / "fixtures" / "live_mapping_oos_panel_2026-09-21.csv"


def _panel() -> pd.DataFrame:
    panel = pd.read_csv(FIXTURE, parse_dates=["date"])
    panel.attrs["horizon_months"] = 6
    return panel


@pytest.mark.parametrize("mean_predicted", [-0.20, -0.01, 0.0, 0.01, 0.04, 0.05, 0.06, 0.15, 0.30])
@pytest.mark.parametrize("mean_ic", [0.05, 0.07, 0.12, 0.40])
def test_bullish_signal_never_sells_more_than_the_default(mean_predicted: float, mean_ic: float) -> None:
    """OUTPERFORM must never sell more than the 50 % default (F20)."""
    assert sell_pct_from_consensus("OUTPERFORM", mean_predicted, mean_ic) <= 0.50


def test_actionable_outperform_with_small_forecast_sells_the_default() -> None:
    """The review's case: OUTPERFORM with a forecast <= 5 % sold 75 %."""
    health = {"oos_r2": 0.05, "pt_p_value": 0.01, "agg_hit": 0.70, "constant_rule_hit_rate": 0.60}

    class _Cpcv:
        stability_verdict = "GOOD"

    mode = determine_recommendation_mode("OUTPERFORM", 0.03, 0.10, 0.70, health, _Cpcv())
    assert mode["mode"] == "actionable"
    assert mode["sell_pct"] == pytest.approx(0.50)


def test_non_finite_ic_maps_to_the_default() -> None:
    """A missing IC is weak evidence (fail closed), not a pass."""
    assert sell_pct_from_consensus("UNDERPERFORM", -0.10, float("nan")) == pytest.approx(0.50)
    assert sell_pct_from_consensus("OUTPERFORM", 0.30, float("nan")) == pytest.approx(0.50)


def test_live_mapping_policy_regression_uplift_vs_always_50() -> None:
    """Policy regression: the live mapping must not lose to always-50 % OOS (F20).

    Unfixed mapping on this fixture: mean uplift -0.11 pp per decision (the
    75 % bucket on weak OUTPERFORM calls costs 2.4 pp on 22 dates).
    """
    from src.models.live_policy_backtest import historical_live_decisions, uplift_vs_default

    decisions = historical_live_decisions(_panel(), sell_pct_from_consensus)
    assert len(decisions) == 186
    uplift = uplift_vs_default(decisions)
    assert uplift.mean() >= 0.0, f"live mapping loses {uplift.mean():+.4%} per decision to always-50%"


def test_bullish_decisions_in_the_backtest_never_exceed_the_default() -> None:
    from src.models.live_policy_backtest import historical_live_decisions

    decisions = historical_live_decisions(_panel(), sell_pct_from_consensus)
    bullish = decisions[decisions["consensus"] == "OUTPERFORM"]
    assert not bullish.empty
    assert (bullish["sell_pct"] <= 0.50).all()


def test_backtest_replays_the_live_rules_on_a_hand_built_panel() -> None:
    """Two benchmarks, 20 months, forecasts perfectly ordered: every call checks by hand."""
    from src.models.live_policy_backtest import (
        MIN_REALISED_ROWS_FOR_IC,
        evaluate_live_mapping,
        historical_live_decisions,
        uplift_vs_default,
    )

    dates = pd.date_range("2020-01-31", periods=20, freq="ME")
    rows = []
    for i, d in enumerate(dates):
        for bench, shift in (("AAA", 0.0), ("BBB", 0.01)):
            y_hat = 0.02 * ((i % 5) - 2) + shift
            rows.append(
                {"benchmark": bench, "date": d, "y_true": 2.0 * y_hat, "z": y_hat,
                 "alpha": 1.0, "y_hat": y_hat, "naive": 0.0}
            )
    panel = pd.DataFrame(rows)
    panel.attrs["horizon_months"] = 6
    decisions = historical_live_decisions(panel, sell_pct_from_consensus)
    assert len(decisions) == 20

    # Month index 17 is the first with >= 12 realised rows per benchmark
    # (months 0..11 realised by month 17 with a 6-month horizon).
    first = MIN_REALISED_ROWS_FOR_IC + 6 - 1
    early = decisions.iloc[:first]
    assert (early["consensus"] == "NEUTRAL").all()
    assert (early["sell_pct"] == 0.50).all()

    late = decisions.iloc[first:].reset_index(drop=True)
    # i = 17, 18, 19 -> i % 5 = 2, 3, 4 -> y_hat AAA 0.00, 0.02, 0.04; BBB 0.01, 0.03, 0.05.
    # AAA 0.00 and BBB 0.01 are below the 1 % return bar only for AAA (|0.01| is not < 0.01).
    assert list(late["consensus"]) == ["NEUTRAL", "OUTPERFORM", "OUTPERFORM"]
    assert late.loc[1, "mean_predicted"] == pytest.approx(0.025)
    assert late.loc[2, "mean_predicted"] == pytest.approx(0.045)
    assert (late["sell_pct"] <= 0.50).all()
    assert late.loc[0, "realized"] == pytest.approx(0.01)

    summary = evaluate_live_mapping(decisions)
    expected_policy = ((1 - decisions["sell_pct"]) * decisions["realized"]).mean()
    assert summary.mean_policy_return == pytest.approx(expected_policy)
    manual_uplift = ((0.5 - decisions["sell_pct"]) * decisions["realized"]).mean()
    assert uplift_vs_default(decisions).mean() == pytest.approx(manual_uplift)
    assert math.isclose(summary.uplift_vs_sell_50, manual_uplift, abs_tol=1e-12)


def test_monthly_policy_backtest_includes_the_live_mapping() -> None:
    """The report's policy backtest scores the live mapping, not only tiered_25_50_100."""
    import scripts.monthly_decision as md
    from src.models.live_policy_backtest import LIVE_MAPPING_POLICY
    from tests.test_policy_backtest_monthly import _make_ensemble

    y_hat = [0.05, -0.02, 0.08, -0.04, 0.03, 0.01, -0.06, 0.09]
    y_true = [0.04, -0.03, 0.07, -0.05, 0.02, -0.01, 0.03, 0.06]
    summary = md._compute_policy_summary({"VTI": _make_ensemble(y_hat, y_true)})
    assert summary is not None
    assert LIVE_MAPPING_POLICY in summary
    assert summary[LIVE_MAPPING_POLICY].n_obs == len(y_true)
