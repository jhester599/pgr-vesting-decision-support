"""v149 -- Kelly fraction / cap replay proxy on the v138 BL posterior frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v138_bl_param_eval import DEFAULT_CANDIDATE_PATH as DEFAULT_BL_PARAMS_PATH, _build_proxy_frame, _load_replay_frame, _parse_params

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v149_kelly_candidate.json"
DEFAULT_NEUTRAL_BAND = 0.015


def evaluate_kelly_params(
    fraction: float,
    cap: float,
    neutral_band: float = DEFAULT_NEUTRAL_BAND,
) -> dict[str, float]:
    """Evaluate Kelly-style sizing parameters on the preserved BL proxy frame."""
    fraction_value = float(fraction)
    cap_value = float(cap)
    if not 0.01 <= fraction_value <= 1.0:
        raise ValueError(f"fraction must be in [0.01, 1.0], got {fraction_value}")
    if not 0.05 <= cap_value <= 0.50:
        raise ValueError(f"cap must be in [0.05, 0.50], got {cap_value}")
    tau, confidence = _parse_params(None, None, str(DEFAULT_BL_PARAMS_PATH))
    replay_df = _load_replay_frame()
    proxy_df = _build_proxy_frame(replay_df, tau, confidence)

    confidence_z = proxy_df["posterior_mean"] / np.sqrt(proxy_df["posterior_variance"])
    signed_shift = 0.5 * np.tanh(fraction_value * confidence_z.to_numpy(dtype=float))
    signed_shift = np.clip(signed_shift, -cap_value, cap_value)
    hold_fraction = np.clip(0.5 + signed_shift, 0.0, 1.0)
    recommendation = np.where(
        hold_fraction >= 0.5 + neutral_band,
        "hold",
        np.where(hold_fraction <= 0.5 - neutral_band, "reduce", "neutral"),
    )
    covered_mask = recommendation != "neutral"
    if covered_mask.any():
        correct = (
            ((recommendation[covered_mask] == "hold") & (proxy_df.loc[covered_mask, "realized"] > 0.0))
            | ((recommendation[covered_mask] == "reduce") & (proxy_df.loc[covered_mask, "realized"] <= 0.0))
        )
        success_rate = float(np.mean(correct))
    else:
        success_rate = 0.0
    utility_score = float(np.mean((hold_fraction - 0.5) * proxy_df["realized"].to_numpy(dtype=float)))
    return {
        "utility_score": utility_score,
        "coverage": float(np.mean(covered_mask)),
        "success_rate": success_rate,
        "mean_hold_fraction": float(np.mean(hold_fraction)),
    }


def _parse_candidate(candidate_file: str) -> tuple[float, float]:
    payload = json.loads(Path(candidate_file).read_text(encoding="utf-8"))
    return float(payload["fraction"]), float(payload["cap"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Kelly fraction/cap replay proxy.")
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--cap", type=float, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fraction = args.fraction
    cap = args.cap
    if fraction is None or cap is None:
        fraction, cap = _parse_candidate(args.candidate_file)
    metrics = evaluate_kelly_params(float(fraction), float(cap))
    print(f"utility_score={metrics['utility_score']:.4f}")
    print(f"coverage={metrics['coverage']:.4f}")
    print(f"success_rate={metrics['success_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
