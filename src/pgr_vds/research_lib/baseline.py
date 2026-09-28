"""Fixed incumbent procedure and non-searched endpoint controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pgr_vds.research_lib.adapters import nested_ridge
from pgr_vds.research_lib.temporal import chronological_splits


def strict_predictions(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    horizon: int,
    columns: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[dict]]:
    """Split monthly origins first; fit each model on past labels."""
    empty = pd.DataFrame(
        columns=[
            "date",
            "benchmark",
            "horizon",
            "available",
            "target_end",
            "fold",
            "y_true",
            "ridge",
            "gbt",
            "ridge_alpha",
        ]
    )
    if targets.empty:
        return empty, [
            {
                "kind": "outer",
                "horizon": horizon,
                "status": "unscorable",
                "n_train": 0,
                "n_test": 0,
                "n_origins": 0,
                "gap": 2 * horizon,
                "reason": "Empty target history",
            }
        ]
    dates = pd.DatetimeIndex(sorted(targets["date"].unique()))
    dates = pd.date_range(dates.min(), dates.max(), freq="BME")
    splits = chronological_splits(dates, horizon)
    if not splits:
        return empty, [
            {
                "kind": "outer",
                "horizon": horizon,
                "benchmark": benchmark,
                "status": "unscorable",
                "n_train": 0,
                "n_test": 0,
                "n_origins": len(dates),
                "gap": 2 * horizon,
                "reason": "Insufficient history for fixed chronological folds",
            }
            for benchmark in sorted(targets["benchmark"].unique())
        ]
    x = features.reindex(dates)
    predictions: list[dict] = []
    ledger: list[dict] = []
    minimum = 24 if horizon == 6 else 60
    for fold, (train, test) in enumerate(splits):
        for benchmark in sorted(targets["benchmark"].unique()):
            labels = targets.loc[targets["benchmark"] == benchmark].set_index(
                "date"
            )
            y = labels["y_true"].reindex(dates)
            available = labels["available"].reindex(dates)
            usable = (available.iloc[train] <= dates[test[0]]).to_numpy()
            y_train = y.iloc[train].where(usable)
            n_train = int(y_train.notna().sum())
            outer = {
                "kind": "outer",
                "fold": fold,
                "benchmark": benchmark,
                "horizon": horizon,
                "train_start": str(dates[train[0]].date()),
                "train_end": str(dates[train[-1]].date()),
                "test_start": str(dates[test[0]].date()),
                "test_end": str(dates[test[-1]].date()),
                "n_train": n_train,
                "purge": horizon,
                "embargo": horizon,
                "gap": 2 * horizon,
                "status": "unscorable",
            }
            if n_train < minimum:
                outer["reason"] = "Insufficient mature monthly labels"
                ledger.append(outer)
                continue
            try:
                ridge, alpha, inner = nested_ridge(
                    x.iloc[train][columns["ridge"]],
                    y_train,
                    x.iloc[test][columns["ridge"]],
                    horizon,
                    available=available.iloc[train],
                )
            except ValueError as error:
                outer["reason"] = str(error)
                ledger.append(outer)
                continue
            for inner_fold, entry in enumerate(inner):
                ledger.append(
                    {
                        **entry,
                        "kind": "inner",
                        "fold": fold,
                        "inner_fold": inner_fold,
                        "benchmark": benchmark,
                        "horizon": horizon,
                        "gap": 2 * horizon,
                        "status": "scorable",
                        "selected_alpha": alpha,
                    }
                )
            gbt = Pipeline(
                [
                    (
                        "impute",
                        SimpleImputer(
                            strategy="median", keep_empty_features=True
                        ),
                    ),
                    ("scale", StandardScaler()),
                    (
                        "gbt",
                        GradientBoostingRegressor(
                            max_depth=2,
                            n_estimators=50,
                            learning_rate=0.1,
                            subsample=0.8,
                            random_state=42,
                        ),
                    ),
                ]
            )
            finite = y_train.notna()
            gbt.fit(
                x.iloc[train].loc[finite, columns["gbt"]], y_train.loc[finite]
            )
            gbt_prediction = gbt.predict(x.iloc[test][columns["gbt"]])
            outer.update(status="scorable", selected_alpha=alpha)
            ledger.append(outer)
            for position, index in enumerate(test):
                if not np.isfinite(y.iloc[index]):
                    continue
                predictions.append(
                    {
                        "date": dates[index],
                        "benchmark": benchmark,
                        "horizon": horizon,
                        "available": available.iloc[index],
                        "target_end": (
                            labels.loc[dates[index], "target_end"]
                            if "target_end" in labels
                            else available.iloc[index]
                        ),
                        "fold": fold,
                        "y_true": float(y.iloc[index]),
                        "ridge": float(ridge[position]),
                        "gbt": float(gbt_prediction[position]),
                        "ridge_alpha": alpha,
                    }
                )
    return pd.DataFrame(predictions), ledger


def temperature_stream(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed production temperature grid selected from matured OOS labels."""
    from src.models.path_b_classifier import (
        _apply_temperature,
        _fit_temperature_grid,
    )

    output = frame.sort_values("date").copy()
    output["probability"] = np.nan
    output["temperature"] = np.nan
    output["calibration_warmup"] = True
    for index, row in output.iterrows():
        past = output.loc[
            (output["available"] <= row["date"])
            & (output["date"] < row["date"])
        ]
        if len(past) < 24 or past["y_true"].nunique() < 2:
            continue
        temperature = _fit_temperature_grid(
            past["y_true"].to_numpy(), past["raw_probability"].to_numpy()
        )
        output.loc[index, "probability"] = _apply_temperature(
            row["raw_probability"], temperature
        )
        output.loc[index, "temperature"] = temperature
        output.loc[index, "calibration_warmup"] = False
    return output


def calibration_summary(frame: pd.DataFrame) -> dict:
    """Score only genuine prequential probabilities and available intervals."""
    from src.models.calibration import compute_ece

    result: dict = {
        "brier": None,
        "log_loss": None,
        "ece": None,
        "coverage_80": None,
        "n_calibration": 0,
        "n_intervals": 0,
    }
    probabilities = frame.dropna(subset=["probability", "y_true"])
    if len(probabilities):
        labels = (probabilities["y_true"] > 0).astype(int).to_numpy()
        p = np.clip(probabilities["probability"].to_numpy(), 1e-6, 1 - 1e-6)
        result.update(
            brier=float(np.mean((p - labels) ** 2)),
            log_loss=float(
                -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
            ),
            ece=float(compute_ece(p, labels)),
            n_calibration=len(p),
        )
    intervals = frame.dropna(subset=["lower", "upper", "y_true"])
    if len(intervals):
        result.update(
            coverage_80=float(
                np.mean(
                    (intervals["y_true"] >= intervals["lower"])
                    & (intervals["y_true"] <= intervals["upper"])
                )
            ),
            n_intervals=len(intervals),
        )
    return result
