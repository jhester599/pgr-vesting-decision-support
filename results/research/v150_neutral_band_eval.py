"""v150 -- Neutral-band replay proxy on top of the v149 Kelly baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v149_kelly_eval import DEFAULT_CANDIDATE_PATH as DEFAULT_KELLY_CANDIDATE_PATH, evaluate_kelly_params

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v150_neutral_band_candidate.txt"


def evaluate_neutral_band(
    neutral_band: float,
) -> dict[str, float]:
    """Evaluate one neutral-band width using the v149 Kelly candidate."""
    band_value = float(neutral_band)
    if not 0.0 <= band_value <= 0.10:
        raise ValueError(f"neutral_band must be in [0.0, 0.10], got {band_value}")
    import json

    payload = json.loads(DEFAULT_KELLY_CANDIDATE_PATH.read_text(encoding="utf-8"))
    return evaluate_kelly_params(
        fraction=float(payload["fraction"]),
        cap=float(payload["cap"]),
        neutral_band=band_value,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate neutral-band replay proxy.")
    parser.add_argument("--band", type=float, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    band = args.band
    if band is None:
        band = float(Path(args.candidate_file).read_text(encoding="utf-8").strip())
    metrics = evaluate_neutral_band(float(band))
    print(f"success_rate={metrics['success_rate']:.4f}")
    print(f"coverage={metrics['coverage']:.4f}")
    print(f"utility_score={metrics['utility_score']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
