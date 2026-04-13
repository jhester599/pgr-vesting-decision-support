"""v131 -- Asymmetric abstention threshold evaluator for Path B temperature-scaled classifier.

Usage
-----
python results/research/v131_threshold_sweep_eval.py --low 0.30 --high 0.70

Loads the matched fold detail from v125, applies prequential temperature scaling
(warmup=24, verbatim from path_b_classifier.py), then evaluates the covered balanced
accuracy under the supplied abstention band.

Outputs (to stdout, one metric per line)
-----------------------------------------
covered_ba=X.XXXX
coverage=X.XXXX

Exit codes
----------
0 -- success
1 -- coverage < 0.20 (insufficient predictions to be useful)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
FOLD_DETAIL_PATH: Path = (
    PROJECT_ROOT / "results" / "research" / "v125_portfolio_target_fold_detail.csv"
)

# ---------------------------------------------------------------------------
# Temperature scaling (verbatim from path_b_classifier.py for reproducibility)
# ---------------------------------------------------------------------------
_GRID_TEMPERATURES: np.ndarray = np.concatenate(
    [
        np.linspace(0.50, 0.95, 10),
        np.linspace(1.0, 3.0, 41),
    ]
)

MIN_COVERAGE: float = 0.20
WARMUP: int = 24


def _clip_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(probs: np.ndarray) -> np.ndarray:
    clipped = _clip_probs(probs)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _apply_temperature(prob: float, temperature: float) -> float:
    temp = max(float(temperature), 1e-6)
    value = float(_sigmoid(np.array([_logit(np.array([prob]))[0] / temp]))[0])
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _fit_temperature_grid(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_hist = np.asarray(y_true, dtype=int)
    p_hist = _clip_probs(y_prob)
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in _GRID_TEMPERATURES:
        scaled = _sigmoid(_logit(p_hist) / float(temperature))
        eps = 1e-6
        scaled = np.clip(scaled, eps, 1.0 - eps)
        loss = -float(
            np.mean(y_hist * np.log(scaled) + (1 - y_hist) * np.log(1.0 - scaled))
        )
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


def apply_prequential_temperature_scaling(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    warmup: int = WARMUP,
) -> np.ndarray:
    """Apply prequential (walk-forward) temperature scaling.

    For each index t >= warmup, fits temperature on probs[:t] / labels[:t]
    and applies it to probs[t]. Observations before warmup are returned clipped
    but otherwise unchanged.

    Parameters
    ----------
    probs:
        Raw OOS probabilities in chronological order.
    labels:
        Integer binary labels (0/1) aligned with probs.
    warmup:
        Minimum prior observations required before calibration is applied.

    Returns
    -------
    np.ndarray
        Calibrated probabilities, same shape as probs.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    calibrated = _clip_probs(probs).copy()
    for idx in range(len(probs)):
        if idx < warmup or len(np.unique(labels[:idx])) < 2:
            continue
        temperature = _fit_temperature_grid(labels[:idx], probs[:idx])
        calibrated[idx] = _apply_temperature(float(probs[idx]), temperature)
    return np.clip(calibrated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low: float,
    high: float,
) -> dict[str, float]:
    """Compute covered balanced accuracy and coverage for given abstention band.

    Rows with ``low <= y_prob <= high`` are abstained (excluded from scoring).
    Rows with ``y_prob < low OR y_prob > high`` are the *covered* set.

    Parameters
    ----------
    y_true:
        Binary integer labels (0/1).
    y_prob:
        Temperature-scaled probabilities aligned with y_true.
    low:
        Lower abstention boundary (abstain if y_prob >= low).
    high:
        Upper abstention boundary (abstain if y_prob <= high).

    Returns
    -------
    dict with keys ``covered_ba`` and ``coverage``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    covered_mask = (y_prob < low) | (y_prob > high)
    n_covered = int(covered_mask.sum())
    n_total = len(y_true)
    coverage = n_covered / n_total if n_total > 0 else 0.0

    if n_covered == 0 or len(np.unique(y_true[covered_mask])) < 2:
        # Cannot compute balanced accuracy — return 0.5 (chance level)
        covered_ba = 0.5
    else:
        covered_ba = float(
            balanced_accuracy_score(y_true[covered_mask], (y_prob[covered_mask] >= 0.5).astype(int))
        )

    return {"covered_ba": covered_ba, "coverage": coverage}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate asymmetric abstention thresholds for Path B temp-scaled classifier."
    )
    parser.add_argument(
        "--low",
        type=float,
        required=True,
        help="Lower abstention boundary (abstain if low <= prob <= high).",
    )
    parser.add_argument(
        "--high",
        type=float,
        required=True,
        help="Upper abstention boundary (abstain if low <= prob <= high).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code."""
    args = _parse_args(argv)
    low: float = args.low
    high: float = args.high

    df = pd.read_csv(FOLD_DETAIL_PATH)
    df["test_date"] = pd.to_datetime(df["test_date"])
    df = df.sort_values("test_date").reset_index(drop=True)

    y_true = df["y_true"].to_numpy(dtype=int)
    path_b_raw = df["path_b_prob"].to_numpy(dtype=float)

    # Apply prequential temperature scaling (adopted method from v130/v131)
    y_prob = apply_prequential_temperature_scaling(path_b_raw, y_true, warmup=WARMUP)

    metrics = evaluate_thresholds(y_true, y_prob, low=low, high=high)

    covered_ba = metrics["covered_ba"]
    coverage = metrics["coverage"]

    print(f"covered_ba={covered_ba:.4f}")
    print(f"coverage={coverage:.4f}")

    if coverage < MIN_COVERAGE:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
