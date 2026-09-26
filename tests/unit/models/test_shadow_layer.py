"""Review 2026-09-25, step 6 (WP8, F24): shadow-layer defects.

- Path B scores the decision row, through a scaled pipeline.
- Classifier-history maturity is recomputed when outcomes are attached.
- The veto gate and the "Aligned" label read the live action's direction.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.models.classification_gate_overlay import compute_shadow_gate_overlay
from src.models.path_b_classifier import fit_path_b_classifier


def _training_frame(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2010-01-31", periods=n, freq="ME")
    x = pd.DataFrame(
        {"small": rng.normal(size=n) * 0.01, "big": rng.normal(size=n) * 100.0},
        index=index,
    )
    logits = 150.0 * x["small"] - 0.01 * x["big"]
    y = pd.Series((rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(int), index=index)
    return x, y


# ---------------------------------------------------------------------------
# Path B
# ---------------------------------------------------------------------------


def test_path_b_scores_the_current_row_not_the_last_labelled_row() -> None:
    """The last labelled row and the decision row differ; Path B must score
    the decision row (the old code scored ``X.iloc[-1]`` of the labelled frame)."""
    x, y = _training_frame()
    current = pd.DataFrame({"small": [0.03], "big": [-150.0]}, index=[pd.Timestamp("2020-06-30")])
    p_current = fit_path_b_classifier(x, y, ["small", "big"], X_current=current)
    p_last = fit_path_b_classifier(x, y, ["small", "big"], X_current=x.iloc[[-1]])
    assert p_current is not None and p_last is not None
    assert p_current > 0.9
    assert abs(p_current - p_last) > 0.05


def test_path_b_is_invariant_to_feature_units() -> None:
    """With a scaler in the pipeline, re-expressing a feature in other units
    (x 1000) leaves the probability unchanged. Without one, the C = 0.5 L2
    penalty shrinks the small-unit feature and the probability moves."""
    x, y = _training_frame()
    current = pd.DataFrame({"small": [0.02], "big": [50.0]}, index=[pd.Timestamp("2020-06-30")])
    base = fit_path_b_classifier(x, y, ["small", "big"], X_current=current)
    x_scaled = x.assign(small=x["small"] * 1000.0)
    current_scaled = current.assign(small=current["small"] * 1000.0)
    rescaled = fit_path_b_classifier(x_scaled, y, ["small", "big"], X_current=current_scaled)
    assert base is not None and rescaled is not None
    assert rescaled == pytest.approx(base, abs=1e-6)


# ---------------------------------------------------------------------------
# Maturity
# ---------------------------------------------------------------------------


def _history() -> pd.DataFrame:
    # Rows are written with is_horizon_mature False at run time.
    return pd.DataFrame(
        {
            "as_of_date": ["2026-01-20", "2026-08-20"],
            "run_date": ["2026-01-20", "2026-08-20"],
            "feature_anchor_date": ["2025-12-31", "2026-07-31"],
            "mature_on_date": ["2026-06-30", "2027-01-31"],
            "is_horizon_mature": [False, False],
            "classifier_prob_actionable_sell": [0.30, 0.40],
        }
    )


def _patch_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.classification_monitoring as monitoring

    index = pd.date_range("2025-06-30", "2026-07-31", freq="ME")
    values = np.linspace(-0.10, 0.10, len(index))

    def fake_loader(conn, benchmark, horizon):  # noqa: ARG001
        return pd.Series(values, index=index)

    monkeypatch.setattr(monitoring.config, "PRIMARY_FORECAST_UNIVERSE", ["AAA", "BBB"])
    monkeypatch.setattr(monitoring, "load_relative_return_matrix", fake_loader)


def test_matured_rows_are_recomputed_and_get_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 2025-12-31 row's window ended 2026-06-30: by 2026-09-21 it is
    mature and gets its outcome. The old code kept the stored False, so no
    row ever matured (Matured observations 0)."""
    from src.models.classification_monitoring import (
        attach_matured_classifier_outcomes,
        summarize_matured_classifier_history,
    )

    _patch_targets(monkeypatch)
    out = attach_matured_classifier_outcomes(None, _history(), horizon_months=6)
    assert list(out["is_horizon_mature"]) == [True, False]
    index = pd.date_range("2025-06-30", "2026-07-31", freq="ME")
    expected = float(np.linspace(-0.10, 0.10, len(index))[list(index).index(pd.Timestamp("2025-12-31"))])
    assert out.loc[0, "actual_basket_relative_return"] == pytest.approx(expected)
    assert out.loc[0, "actual_actionable_sell"] == float(expected < -0.03)
    assert np.isnan(out.loc[1, "actual_basket_relative_return"])
    assert summarize_matured_classifier_history(out).matured_n == 1


def test_backdated_attach_does_not_see_later_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    """As of 2026-06-29 the 2025-12-31 window (ends 2026-06-30) is still open."""
    from src.models.classification_monitoring import attach_matured_classifier_outcomes

    _patch_targets(monkeypatch)
    out = attach_matured_classifier_outcomes(
        None, _history(), horizon_months=6, as_of=date(2026, 6, 29)
    )
    assert list(out["is_horizon_mature"]) == [False, False]
    assert out["actual_basket_relative_return"].isna().all()
    assert out.loc[0, "mature_on_date"] == "2026-06-30"


# ---------------------------------------------------------------------------
# Direction-aware veto and agreement
# ---------------------------------------------------------------------------


def _overlay(live_sell_pct: float, consensus: str, prob: float):
    return compute_shadow_gate_overlay(
        live_mode="ACTIONABLE",
        live_sell_pct=live_sell_pct,
        consensus=consensus,
        mean_predicted=0.18 if consensus == "OUTPERFORM" else -0.06,
        mean_ic=0.10,
        aggregate_oos_r2=0.05,
        classifier_prob_actionable_sell=prob,
        variant="veto_overlay",
        gate_style="veto_regression_sell",
        threshold=0.60,
    )


def test_veto_does_not_penalise_a_hold_leaning_month_the_classifier_supports() -> None:
    """Bullish ACTIONABLE month (sell 25 %), classifier P(sell) 10 %: the two
    agree. The old gate vetoed it to DEFER / 50 %, i.e. sold more."""
    overlay = _overlay(0.25, "OUTPERFORM", 0.10)
    assert overlay.would_change is False
    assert overlay.recommended_sell_pct == pytest.approx(0.25)
    assert overlay.recommendation_mode == "ACTIONABLE"


def test_veto_blocks_a_hold_the_classifier_contradicts() -> None:
    overlay = _overlay(0.25, "OUTPERFORM", 0.80)
    assert overlay.recommendation_mode == "DEFER-TO-TAX-DEFAULT"
    assert overlay.recommended_sell_pct == pytest.approx(0.50)
    assert overlay.reason == "classifier vetoed regression hold"


def test_veto_still_blocks_a_weak_regression_sell() -> None:
    overlay = _overlay(1.00, "UNDERPERFORM", 0.45)
    assert overlay.recommendation_mode == "DEFER-TO-TAX-DEFAULT"
    assert overlay.reason == "classifier vetoed weak regression sell"


def test_veto_confirms_a_sell_the_classifier_supports() -> None:
    overlay = _overlay(1.00, "UNDERPERFORM", 0.80)
    assert overlay.would_change is False
    assert overlay.reason == "classifier confirmed actionable sell"


@pytest.mark.parametrize(
    ("stance", "mode", "sell_pct", "aligned"),
    [
        ("ACTIONABLE-SELL", "ACTIONABLE", 1.00, True),
        ("NON-ACTIONABLE", "ACTIONABLE", 1.00, False),
        # A bullish ACTIONABLE month is not "aligned" with a sell stance.
        ("ACTIONABLE-SELL", "ACTIONABLE", 0.25, False),
        ("NON-ACTIONABLE", "ACTIONABLE", 0.25, True),
        ("NEUTRAL", "ACTIONABLE", 0.25, True),
        ("ACTIONABLE-SELL", "DEFER-TO-TAX-DEFAULT", 0.50, False),
        ("NON-ACTIONABLE", "DEFER-TO-TAX-DEFAULT", 0.50, True),
    ],
)
def test_agreement_reads_the_live_direction(stance: str, mode: str, sell_pct: float, aligned: bool) -> None:
    from src.models.classification_shadow import agreement_with_live_recommendation

    assert agreement_with_live_recommendation(stance, mode, sell_pct) is aligned
