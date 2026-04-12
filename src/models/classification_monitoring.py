"""Historical monitoring helpers for shadow classifier outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import config
from src.models.calibration import compute_ece
from src.processing.multi_total_return import load_relative_return_matrix


@dataclass(frozen=True)
class ClassifierMaturitySummary:
    """Matured-horizon classifier monitoring summary."""

    matured_n: int
    brier_score: float | None
    log_loss: float | None
    ece_10: float | None

    def to_payload(self) -> dict[str, float | int | None]:
        """Return a JSON-serializable payload."""
        return asdict(self)


def attach_matured_classifier_outcomes(
    conn,
    history_df: pd.DataFrame,
    *,
    horizon_months: int = 6,
    threshold: float = 0.03,
) -> pd.DataFrame:
    """Fill matured classifier history rows with realized basket outcomes."""
    if history_df.empty or "feature_anchor_date" not in history_df.columns:
        return history_df

    rel_map: dict[str, pd.Series] = {}
    for benchmark in config.PRIMARY_FORECAST_UNIVERSE:
        rel_series = load_relative_return_matrix(conn, benchmark, horizon_months)
        if not rel_series.empty:
            rel_map[benchmark] = rel_series
    if not rel_map:
        return history_df

    basket_rel = pd.DataFrame(rel_map).mean(axis=1).rename("actual_basket_relative_return")
    df = history_df.copy()
    if "actual_actionable_sell" not in df.columns:
        df["actual_actionable_sell"] = np.nan
    if "actual_basket_relative_return" not in df.columns:
        df["actual_basket_relative_return"] = np.nan

    for idx, row in df.iterrows():
        if not bool(row.get("is_horizon_mature", False)):
            continue
        anchor = row.get("feature_anchor_date")
        if pd.isna(anchor):
            continue
        anchor_ts = pd.Timestamp(str(anchor))
        if anchor_ts not in basket_rel.index:
            continue
        realized = float(basket_rel.loc[anchor_ts])
        df.at[idx, "actual_basket_relative_return"] = realized
        df.at[idx, "actual_actionable_sell"] = float(realized < -threshold)
    return df


def summarize_matured_classifier_history(
    history_df: pd.DataFrame,
) -> ClassifierMaturitySummary:
    """Compute matured-horizon classifier diagnostics from history rows."""
    required = {
        "classifier_prob_actionable_sell",
        "actual_actionable_sell",
    }
    if history_df.empty or not required.issubset(history_df.columns):
        return ClassifierMaturitySummary(0, None, None, None)

    matured = history_df.dropna(
        subset=["classifier_prob_actionable_sell", "actual_actionable_sell"]
    ).copy()
    if matured.empty:
        return ClassifierMaturitySummary(0, None, None, None)

    y_prob = np.clip(
        matured["classifier_prob_actionable_sell"].to_numpy(dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    y_true = matured["actual_actionable_sell"].to_numpy(dtype=int)
    brier = float(np.mean((y_true - y_prob) ** 2))
    try:
        ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        ll = None
    ece = float(compute_ece(y_prob, y_true, n_bins=10))
    return ClassifierMaturitySummary(
        matured_n=int(len(matured)),
        brier_score=brier,
        log_loss=ll,
        ece_10=ece,
    )
