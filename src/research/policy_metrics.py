"""Policy-level utility helpers for v9 decision evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FIXED_POLICIES: tuple[str, ...] = (
    "always_sell_100",
    "always_sell_50",
    "always_hold_100",
)
SIGNAL_POLICIES: tuple[str, ...] = (
    "sign_hold_vs_sell",
    "tiered_25_50_100",
)


@dataclass(frozen=True)
class PolicySummary:
    """Summary of normalized decision utility against a benchmark baseline."""

    n_obs: int
    avg_hold_fraction: float
    mean_policy_return: float
    median_policy_return: float
    cumulative_policy_return: float
    positive_utility_rate: float
    regret_vs_oracle: float
    uplift_vs_sell_all: float
    uplift_vs_sell_50: float
    uplift_vs_hold_all: float
    capture_ratio: float


def hold_fraction_from_policy(
    predicted: pd.Series,
    policy_name: str,
) -> pd.Series:
    """Map predictions into a normalized hold fraction in [0, 1]."""
    if policy_name == "always_sell_100":
        return pd.Series(0.0, index=predicted.index, name="hold_fraction")
    if policy_name == "always_sell_50":
        return pd.Series(0.5, index=predicted.index, name="hold_fraction")
    if policy_name == "always_hold_100":
        return pd.Series(1.0, index=predicted.index, name="hold_fraction")
    if policy_name == "sign_hold_vs_sell":
        return (predicted > 0.0).astype(float).rename("hold_fraction")
    if policy_name == "tiered_25_50_100":
        hold_fraction = np.where(
            predicted > 0.15,
            0.75,
            np.where(predicted > 0.05, 0.50, 0.0),
        )
        return pd.Series(hold_fraction, index=predicted.index, name="hold_fraction")
    raise ValueError(f"Unknown policy_name '{policy_name}'.")


def evaluate_policy_series(
    predicted: pd.Series,
    realized_relative_return: pd.Series,
    policy_name: str,
) -> PolicySummary:
    """Evaluate decision utility for a given prediction series and policy."""
    aligned = pd.concat([predicted, realized_relative_return], axis=1).dropna()
    if aligned.empty:
        return PolicySummary(
            n_obs=0,
            avg_hold_fraction=float("nan"),
            mean_policy_return=float("nan"),
            median_policy_return=float("nan"),
            cumulative_policy_return=float("nan"),
            positive_utility_rate=float("nan"),
            regret_vs_oracle=float("nan"),
            uplift_vs_sell_all=float("nan"),
            uplift_vs_sell_50=float("nan"),
            uplift_vs_hold_all=float("nan"),
            capture_ratio=float("nan"),
        )

    predicted_aligned = aligned.iloc[:, 0]
    realized = aligned.iloc[:, 1]
    hold_fraction = hold_fraction_from_policy(predicted_aligned, policy_name)
    policy_return = hold_fraction * realized
    oracle_return = realized.clip(lower=0.0)
    sell_50_return = 0.5 * realized
    hold_all_return = realized

    oracle_positive_sum = float(oracle_return.sum())
    capture_ratio = (
        float(policy_return.sum()) / oracle_positive_sum
        if oracle_positive_sum > 1e-12
        else float("nan")
    )

    return PolicySummary(
        n_obs=int(len(aligned)),
        avg_hold_fraction=float(hold_fraction.mean()),
        mean_policy_return=float(policy_return.mean()),
        median_policy_return=float(policy_return.median()),
        cumulative_policy_return=float(policy_return.sum()),
        positive_utility_rate=float((policy_return > 0.0).mean()),
        regret_vs_oracle=float((oracle_return - policy_return).mean()),
        uplift_vs_sell_all=float(policy_return.mean()),
        uplift_vs_sell_50=float((policy_return - sell_50_return).mean()),
        uplift_vs_hold_all=float((policy_return - hold_all_return).mean()),
        capture_ratio=capture_ratio,
    )
