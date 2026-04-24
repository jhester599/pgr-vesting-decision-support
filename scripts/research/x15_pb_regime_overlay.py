"""Run x15 bounded P/B regime overlay experiments."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import db_client
from src.processing.feature_engineering import get_feature_columns
from src.research.x2_absolute_classification import (
    build_absolute_classifier_pipeline,
    iter_absolute_wfo_splits,
    _impute_fold,
)
from src.research.x4_bvps_forecasting import normalize_bvps_monthly
from src.research.x15_pb_regime_overlay import (
    apply_pb_overlay,
    build_pb_regime_targets,
    json_records,
    summarize_x15_results,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x15_pb_regime_overlay_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x15_pb_regime_overlay_summary.json"
MEMO_PATH = OUTPUT_DIR / "x15_research_memo.md"
HORIZONS: tuple[int, ...] = (3, 6)
HURDLE = 0.05
CONFIDENCE_THRESHOLD = 0.55
MODEL_NAMES: tuple[str, ...] = ("logistic_l2_balanced", "hist_gbt_depth2")


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x15 reads this cache "
            "without refreshing it to preserve the research-only boundary."
        )
    return pd.read_parquet(path)


def _month_end_close(prices: pd.DataFrame) -> pd.Series:
    close = prices["close"].copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index))
    result = close.resample("BME").last()
    result.name = "close_price"
    return result


def _build_pb_feature_matrix(
    feature_df: pd.DataFrame,
    current_pb: pd.Series,
) -> pd.DataFrame:
    available = set(get_feature_columns(feature_df))
    preferred = [
        "pb_ratio",
        "pgr_price_to_book_relative",
        "pgr_pe_vs_market_pe",
        "pe_ratio",
        "real_rate_10y",
        "yield_slope",
        "credit_spread_hy",
        "baa10y_spread",
        "pgr_vs_peers_6m",
        "pgr_vs_vfh_6m",
        "mom_6m",
        "high_52w",
        "vix",
    ]
    selected = [column for column in preferred if column in available]
    features = feature_df[selected].copy()
    features["current_pb"] = current_pb.reindex(features.index)
    return features


def _align_inputs(
    X: pd.DataFrame,
    regime_targets: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    selected = [column for column in feature_columns if column in X.columns]
    base = X[selected].drop(columns=["current_pb"], errors="ignore")
    aligned = base.join(regime_targets, how="inner")
    aligned = aligned.dropna(subset=["current_pb", "future_pb", "normalized_delta"])
    if aligned.empty:
        raise ValueError("No aligned non-null x15 observations.")
    return aligned


def _fit_fold_classifier(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.repeat(float(np.mean(y_train)), len(x_test))
    model = build_absolute_classifier_pipeline(model_name)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def _median_shift(
    normalized_delta: pd.Series,
    mask: pd.Series,
    *,
    fallback: float,
) -> float:
    observed = normalized_delta.loc[mask.astype(bool)]
    if observed.empty:
        return float(fallback)
    return float(observed.median())


def evaluate_x15_overlay(
    X: pd.DataFrame,
    regime_targets: pd.DataFrame,
    *,
    model_name: str,
    feature_columns: list[str],
    target_horizon_months: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one bounded x15 overlay model."""
    aligned = _align_inputs(X, regime_targets, feature_columns)
    _, splitter = iter_absolute_wfo_splits(
        len(aligned),
        target_horizon_months=target_horizon_months,
    )
    rows: list[dict[str, Any]] = []
    fold_count = 0
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(aligned)):
        x_train, x_test = _impute_fold(
            aligned.iloc[train_idx][feature_columns].to_numpy(dtype=float),
            aligned.iloc[test_idx][feature_columns].to_numpy(dtype=float),
        )
        up_train = aligned.iloc[train_idx]["target_up"].to_numpy(dtype=int)
        down_train = aligned.iloc[train_idx]["target_down"].to_numpy(dtype=int)
        up_prob = _fit_fold_classifier(model_name, x_train, up_train, x_test)
        down_prob = _fit_fold_classifier(model_name, x_train, down_train, x_test)
        positive_shift = _median_shift(
            aligned.iloc[train_idx]["normalized_delta"],
            aligned.iloc[train_idx]["target_up"],
            fallback=HURDLE,
        )
        negative_shift = _median_shift(
            aligned.iloc[train_idx]["normalized_delta"],
            aligned.iloc[train_idx]["target_down"],
            fallback=-HURDLE,
        )
        for offset, row_idx in enumerate(test_idx):
            row = aligned.iloc[row_idx]
            predicted_pb, action = apply_pb_overlay(
                current_pb=float(row["current_pb"]),
                up_prob=float(up_prob[offset]),
                down_prob=float(down_prob[offset]),
                positive_shift=positive_shift,
                negative_shift=negative_shift,
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )
            rows.append(
                {
                    "date": aligned.index[row_idx],
                    "fold_idx": int(fold_idx),
                    "model_name": model_name,
                    "horizon_months": int(target_horizon_months),
                    "current_pb": float(row["current_pb"]),
                    "future_pb": float(row["future_pb"]),
                    "normalized_delta": float(row["normalized_delta"]),
                    "target_regime": row["target_regime"],
                    "up_prob": float(up_prob[offset]),
                    "down_prob": float(down_prob[offset]),
                    "predicted_pb": predicted_pb,
                    "overlay_action": action,
                }
            )
        fold_count += 1
    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return predictions, _overlay_metrics(
        predictions,
        model_name=model_name,
        horizon_months=target_horizon_months,
        fold_count=fold_count,
        n_features=len(feature_columns),
    )


def evaluate_no_change_overlay(
    regime_targets: pd.DataFrame,
    *,
    target_horizon_months: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the no-change P/B overlay baseline."""
    _, splitter = iter_absolute_wfo_splits(
        len(regime_targets),
        target_horizon_months=target_horizon_months,
    )
    rows: list[dict[str, Any]] = []
    for fold_idx, (_, test_idx) in enumerate(splitter.split(regime_targets)):
        for row_idx in test_idx:
            row = regime_targets.iloc[row_idx]
            rows.append(
                {
                    "date": regime_targets.index[row_idx],
                    "fold_idx": int(fold_idx),
                    "model_name": "no_change_pb_overlay",
                    "horizon_months": int(target_horizon_months),
                    "current_pb": float(row["current_pb"]),
                    "future_pb": float(row["future_pb"]),
                    "normalized_delta": float(row["normalized_delta"]),
                    "target_regime": row["target_regime"],
                    "up_prob": float("nan"),
                    "down_prob": float("nan"),
                    "predicted_pb": float(row["current_pb"]),
                    "overlay_action": "neutral",
                }
            )
    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return predictions, _overlay_metrics(
        predictions,
        model_name="no_change_pb_overlay",
        horizon_months=target_horizon_months,
        fold_count=int(predictions["fold_idx"].nunique()),
        n_features=0,
    )


def _overlay_metrics(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    horizon_months: int,
    fold_count: int,
    n_features: int,
) -> dict[str, Any]:
    pb_error = predictions["predicted_pb"] - predictions["future_pb"]
    realized_regime = predictions["target_regime"].astype(str)
    predicted_regime = predictions["overlay_action"].astype(str)
    return {
        "horizon_months": int(horizon_months),
        "model_name": model_name,
        "n_obs": int(len(predictions)),
        "fold_count": int(fold_count),
        "n_features": int(n_features),
        "pb_mae": float(np.mean(np.abs(pb_error))),
        "pb_rmse": float(np.sqrt(np.mean(np.square(pb_error)))),
        "overlay_action_rate": float(np.mean(predicted_regime != "neutral")),
        "regime_hit_rate": float(np.mean(predicted_regime == realized_regime)),
        "mean_predicted_pb": float(predictions["predicted_pb"].mean()),
        "mean_realized_pb": float(predictions["future_pb"].mean()),
    }


def run_x15_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run x15 overlay experiments for medium horizons."""
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _load_feature_matrix_read_only()
        prices = db_client.get_prices(conn, "PGR")
        pgr_monthly = db_client.get_pgr_edgar_monthly(conn)
    finally:
        conn.close()

    current_bvps = normalize_bvps_monthly(
        pgr_monthly,
        filing_lag_months=config.EDGAR_FILING_LAG_MONTHS,
    )
    current_pb = (_month_end_close(prices) / current_bvps).rename("current_pb")
    X = _build_pb_feature_matrix(feature_df, current_pb)
    feature_columns = list(X.columns)

    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        future_pb = current_pb.shift(-horizon).rename("future_pb")
        regime_targets = build_pb_regime_targets(
            current_pb,
            future_pb,
            hurdle=HURDLE,
        )
        _, baseline_metrics = evaluate_no_change_overlay(
            regime_targets,
            target_horizon_months=horizon,
        )
        rows.append(baseline_metrics)
        for model_name in MODEL_NAMES:
            _, metrics = evaluate_x15_overlay(
                X,
                regime_targets,
                model_name=model_name,
                feature_columns=feature_columns,
                target_horizon_months=horizon,
            )
            rows.append(metrics)

    detail = pd.DataFrame(rows)
    summary = summarize_x15_results(detail)
    return detail, summary


def write_artifacts(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write x15 artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x15",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "horizons": list(HORIZONS),
        "hurdle": HURDLE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "models": list(MODEL_NAMES),
        "ranked_rows": json_records(summary),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(summary)


def write_memo(summary: pd.DataFrame) -> None:
    """Write a compact x15 memo."""
    lines = [
        "# x15 Research Memo",
        "",
        "## Scope",
        "",
        "x15 tests a bounded P/B regime overlay at the 3m and 6m horizons.",
        "It preserves the research-only boundary and compares against the",
        "existing no-change P/B anchor.",
        "",
        "## Results",
        "",
    ]
    for horizon in HORIZONS:
        rows = summary[summary["horizon_months"] == horizon]
        if rows.empty:
            continue
        best = rows.iloc[0]
        challenger_rows = rows[rows["model_name"] != "no_change_pb_overlay"]
        lines.append(
            f"- {horizon}m best row overall: `{best['model_name']}` "
            f"(P/B MAE {best['pb_mae']:.3f}, RMSE {best['pb_rmse']:.3f}, "
            f"action rate {best['overlay_action_rate']:.3f})."
        )
        if not challenger_rows.empty:
            challenger = challenger_rows.iloc[0]
            verdict = (
                "beat the no-change anchor"
                if bool(challenger["beats_no_change_pb"])
                else "did not beat the no-change anchor"
            )
            lines.append(
                f"- {horizon}m best overlay challenger: "
                f"`{challenger['model_name']}` "
                f"(P/B MAE {challenger['pb_mae']:.3f}, "
                f"RMSE {challenger['pb_rmse']:.3f}, "
                f"action rate {challenger['overlay_action_rate']:.3f}); "
                f"{verdict}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- x15 does not justify replacing `no_change_pb` with this bounded",
            "  regime overlay.",
            "- Treat any overlay improvement as provisional until it is",
            "  recombined with the structural BVPS path in a later x-series",
            "  step.",
            "- The x15 overlay is intentionally bounded and does not let the",
            "  classifier emit arbitrary P/B levels.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, summary = run_x15_experiments()
    write_artifacts(detail, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
