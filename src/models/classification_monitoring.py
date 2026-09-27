"""Historical monitoring helpers for shadow classifier outputs."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import config
from src.models.calibration import compute_ece
from src.processing.feature_engineering import (
    truncate_relative_target_for_asof,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.processing.total_return import forward_window_end


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
    conn: sqlite3.Connection,
    history_df: pd.DataFrame,
    *,
    horizon_months: int = 6,
    threshold: float = 0.03,
    as_of: date | None = None,
    require_complete_basket: bool = False,
) -> pd.DataFrame:
    """Recompute horizon maturity and attach realised basket outcomes.

    Review 2026-09-25, F24: ``is_horizon_mature`` was set once, when the row
    was written (always False then), and never updated, so no row ever got an
    outcome and "Matured observations" stayed 0. Maturity is now recomputed
    here: a row matures when its target window (``forward_window_end`` of the
    feature anchor) has ended by ``as_of`` (default: today). Targets are
    truncated to ``as_of`` as well, so a back-dated run cannot attach an
    outcome it could not have seen.
    """
    if history_df.empty or "feature_anchor_date" not in history_df.columns:
        return history_df
    reference = pd.Timestamp(as_of if as_of is not None else date.today())

    df = history_df.copy()
    for column in ("actual_actionable_sell", "actual_basket_relative_return"):
        if column not in df.columns:
            df[column] = np.nan
    if "mature_on_date" not in df.columns:
        df["mature_on_date"] = pd.NA
    df["mature_on_date"] = df["mature_on_date"].astype(object)
    df["is_horizon_mature"] = False

    baskets: dict[int, pd.Series] = {}
    df["actual_basket_relative_return"] = np.nan
    df["actual_actionable_sell"] = np.nan

    for idx, row in df.iterrows():
        anchor = row.get("feature_anchor_date")
        if anchor is None or pd.isna(anchor):
            continue
        row_horizon = row.get("forecast_horizon_months", horizon_months)
        if pd.isna(row_horizon):
            row_horizon = horizon_months
        horizon = int(row_horizon)
        if horizon <= 0 or horizon != float(row_horizon):
            raise ValueError(
                "forecast_horizon_months must be a positive integer"
            )
        anchor_ts = pd.Timestamp(str(anchor))
        mature_on = forward_window_end(anchor_ts, horizon)
        df.at[idx, "mature_on_date"] = mature_on.date().isoformat()
        is_mature = bool(mature_on <= reference)
        df.at[idx, "is_horizon_mature"] = is_mature
        if not is_mature:
            continue
        if horizon not in baskets:
            rel_map = {
                benchmark: truncate_relative_target_for_asof(
                    load_relative_return_matrix(conn, benchmark, horizon),
                    as_of=reference,
                    horizon_months=horizon,
                )
                for benchmark in config.PRIMARY_FORECAST_UNIVERSE
            }
            targets = pd.DataFrame(rel_map).replace([np.inf, -np.inf], np.nan)
            if require_complete_basket:
                targets = targets.dropna(how="any")
            baskets[horizon] = targets.mean(axis=1).dropna()
        basket_rel = baskets[horizon]
        if anchor_ts not in basket_rel.index:
            if require_complete_basket:
                df.at[idx, "is_horizon_mature"] = False
            continue
        realized = float(basket_rel.loc[anchor_ts])
        df.at[idx, "actual_basket_relative_return"] = realized
        df.at[idx, "actual_actionable_sell"] = float(realized < -threshold)
    return df


def attach_matured_ta_outcomes(
    conn: sqlite3.Connection,
    history_df: pd.DataFrame,
    *,
    as_of: date,
    threshold: float = 0.03,
) -> pd.DataFrame:
    """Attach as-of TA outcomes only when every basket constituent is known.

    Each issuance keeps its own anchor and horizon. The shared cached-target
    reader checks the BME endpoint and removes future outcomes, including
    previously stored outcomes on a backdated replay. No model is fitted.
    """
    return attach_matured_classifier_outcomes(
        conn,
        history_df,
        as_of=as_of,
        threshold=threshold,
        require_complete_basket=True,
    )


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
