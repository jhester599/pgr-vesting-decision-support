"""v138 -- Black-Litterman parameter replay proxy on matured shadow history.

This Target 8 harness is intentionally research-only. The historical repository
state does not preserve the month-by-month BL covariance matrices and view
payloads that were used when each live monthly decision was generated, so a
literal replay of the old BL calls is not possible from current artifacts.

Instead, this script evaluates a bounded Black-Litterman proxy on the longest
available matured decision frame: ``v118_prospective_shadow_replay_results``.
Each month is treated as a single-asset relative-return view where:
  - the equilibrium prior mean is neutral (0 excess return),
  - ``tau`` scales the prior variance,
  - the view mean comes from the archived ensemble prediction,
  - the view variance is a prequential residual MSE scaled by the candidate
    ``view_confidence_scalar`` and tightened when ``prob_sell`` is far from 0.5.

Theoretical priors used in this proxy:
  - lower ``tau`` keeps the posterior closer to the neutral prior,
  - higher ``tau`` lets the archived model view dominate more strongly,
  - lower ``view_confidence_scalar`` tightens Omega and strengthens the view,
  - higher ``view_confidence_scalar`` loosens Omega and makes the decision layer
    more conservative.

Usage:
    python results/research/v138_bl_param_eval.py --tau 0.05 --view-confidence 1.0
    python results/research/v138_bl_param_eval.py --params-file results/research/v138_bl_params_candidate.json

Outputs:
    recommendation_accuracy=X.XXXX
    coverage=X.XXXX
    mean_kelly_fraction=X.XXXX
    policy_uplift=X.XXXX
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config

DEFAULT_REPLAY_PATH = (
    PROJECT_ROOT / "results" / "research" / "v118_prospective_shadow_replay_results.csv"
)
DEFAULT_CANDIDATE_PATH = (
    PROJECT_ROOT / "results" / "research" / "v138_bl_params_candidate.json"
)
NEUTRAL_BAND_HALF_WIDTH = 0.015


def _validate_params(tau: float, view_confidence_scalar: float) -> tuple[float, float]:
    """Validate the tunable BL proxy parameters."""
    tau_value = float(tau)
    confidence_value = float(view_confidence_scalar)
    if not 0.01 <= tau_value <= 0.50:
        raise ValueError(f"tau must be in [0.01, 0.50], got {tau_value}")
    if not 0.25 <= confidence_value <= 4.0:
        raise ValueError(
            "view_confidence_scalar must be in [0.25, 4.0], "
            f"got {confidence_value}"
        )
    return tau_value, confidence_value


def _load_replay_frame(replay_path: Path = DEFAULT_REPLAY_PATH) -> pd.DataFrame:
    """Load the matured replay frame used by the BL proxy evaluator."""
    frame = pd.read_csv(replay_path, parse_dates=["date"])
    required_cols = {"date", "predicted", "realized", "prob_sell"}
    missing = required_cols.difference(frame.columns)
    if missing:
        raise KeyError(f"Replay frame missing required columns: {sorted(missing)}")

    matured = frame.dropna(subset=["predicted", "realized", "prob_sell"]).copy()
    matured = matured.sort_values("date").reset_index(drop=True)
    if matured.empty:
        raise ValueError("Replay frame contains no matured rows with realized outcomes.")
    return matured


def _build_proxy_frame(
    replay_df: pd.DataFrame,
    tau: float,
    view_confidence_scalar: float,
) -> pd.DataFrame:
    """Compute BL-proxy posterior diagnostics month by month."""
    residual_sq = (replay_df["realized"] - replay_df["predicted"]) ** 2
    baseline_view_mse = float(residual_sq.mean())
    prequential_view_mse = residual_sq.expanding(min_periods=12).mean().shift(1)
    prequential_view_mse = prequential_view_mse.fillna(baseline_view_mse)

    confidence_strength = (
        2.0 * (replay_df["prob_sell"] - 0.5).abs()
    ).clip(lower=0.05, upper=1.0)

    realized_var = float(replay_df["realized"].var(ddof=0))
    prior_var = max(realized_var * tau, 1e-6)
    omega = (
        prequential_view_mse.to_numpy(dtype=float)
        * view_confidence_scalar
        / confidence_strength.to_numpy(dtype=float)
    )
    omega = np.clip(omega, 1e-6, None)

    posterior_weight = prior_var / (prior_var + omega)
    posterior_mean = posterior_weight * replay_df["predicted"].to_numpy(dtype=float)
    posterior_var = 1.0 / ((1.0 / prior_var) + (1.0 / omega))

    confidence_z = posterior_mean / np.sqrt(posterior_var)
    hold_fraction = 0.5 + 0.5 * np.tanh(config.KELLY_FRACTION * confidence_z)
    hold_fraction = np.clip(hold_fraction, 0.0, 1.0)
    recommendation = np.where(
        hold_fraction >= 0.5 + NEUTRAL_BAND_HALF_WIDTH,
        "hold",
        np.where(
            hold_fraction <= 0.5 - NEUTRAL_BAND_HALF_WIDTH,
            "reduce",
            "neutral",
        ),
    )

    proxy_df = replay_df.copy()
    proxy_df["view_mse"] = prequential_view_mse.to_numpy(dtype=float)
    proxy_df["confidence_strength"] = confidence_strength.to_numpy(dtype=float)
    proxy_df["posterior_weight"] = posterior_weight
    proxy_df["posterior_mean"] = posterior_mean
    proxy_df["posterior_variance"] = posterior_var
    proxy_df["hold_fraction"] = hold_fraction
    proxy_df["recommendation"] = recommendation
    return proxy_df


def evaluate_bl_params(
    tau: float,
    view_confidence_scalar: float,
    risk_aversion: float = 2.5,
    use_bayesian_variance: bool = True,
) -> dict[str, float]:
    """Evaluate one BL proxy parameter pair on the matured replay frame."""
    del risk_aversion
    del use_bayesian_variance

    tau_value, confidence_value = _validate_params(tau, view_confidence_scalar)
    replay_df = _load_replay_frame()
    proxy_df = _build_proxy_frame(replay_df, tau_value, confidence_value)

    covered_mask = proxy_df["recommendation"] != "neutral"
    coverage = float(covered_mask.mean())

    if covered_mask.any():
        covered = proxy_df.loc[covered_mask].copy()
        correct = (
            ((covered["recommendation"] == "hold") & (covered["realized"] > 0.0))
            | ((covered["recommendation"] == "reduce") & (covered["realized"] <= 0.0))
        )
        recommendation_accuracy = float(correct.mean())
    else:
        recommendation_accuracy = 0.0

    mean_kelly_fraction = float((2.0 * np.abs(proxy_df["hold_fraction"] - 0.5)).mean())
    policy_uplift = float(
        ((proxy_df["hold_fraction"] - 0.5) * proxy_df["realized"]).mean()
    )
    sell_precision = float(
        (
            (
                (proxy_df["recommendation"] == "reduce")
                & (proxy_df["realized"] <= 0.0)
            ).sum()
        )
        / max((proxy_df["recommendation"] == "reduce").sum(), 1)
    )

    return {
        "recommendation_accuracy": recommendation_accuracy,
        "coverage": coverage,
        "mean_kelly_fraction": mean_kelly_fraction,
        "policy_uplift": policy_uplift,
        "sell_precision": sell_precision,
        "n_matured_months": float(len(proxy_df)),
    }


def _parse_params(
    tau: float | None,
    view_confidence: float | None,
    params_file: str | None,
) -> tuple[float, float]:
    """Parse the parameter pair from either flags or the candidate JSON file."""
    if params_file:
        payload = json.loads(Path(params_file).read_text(encoding="utf-8"))
        tau = float(payload["tau"])
        view_confidence = float(payload["view_confidence_scalar"])
    if tau is None or view_confidence is None:
        raise ValueError("Both tau and view confidence must be provided.")
    return float(tau), float(view_confidence)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate BL replay proxy parameters.")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--view-confidence", type=float, default=None)
    parser.add_argument("--params-file", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the BL replay proxy harness."""
    args = _parse_args(argv)
    tau, view_confidence = _parse_params(
        args.tau,
        args.view_confidence,
        args.params_file,
    )
    metrics = evaluate_bl_params(tau, view_confidence)
    print(f"recommendation_accuracy={metrics['recommendation_accuracy']:.4f}")
    print(f"coverage={metrics['coverage']:.4f}")
    print(f"mean_kelly_fraction={metrics['mean_kelly_fraction']:.4f}")
    print(f"policy_uplift={metrics['policy_uplift']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
