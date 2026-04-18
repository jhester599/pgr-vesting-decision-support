"""v163 survivor confirmation helpers for the TA research arc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path("results") / "research"
CONFIRM_PATH = OUTPUT_DIR / "v163_ta_survivor_confirm_summary.csv"
PAYLOAD_PATH = OUTPUT_DIR / "v163_ta_survivor_candidate.json"
DETAIL_PATH = OUTPUT_DIR / "v162_ta_broad_screen_detail.csv"


def select_survivors(
    screen_summary: pd.DataFrame,
    max_features: int = 6,
    max_groups: int = 3,
) -> pd.DataFrame:
    """Select a capped, group-diversified survivor set from screen results."""
    if screen_summary.empty:
        return screen_summary.copy()
    ranked_source = screen_summary.copy()
    if "experiment_mode" in ranked_source:
        ranked_source = ranked_source.loc[
            ranked_source["experiment_mode"].ne("baseline")
        ].copy()
    if "positive_benchmark_count" in ranked_source:
        ranked_source = ranked_source.loc[
            ranked_source["positive_benchmark_count"].fillna(0).ge(3)
        ].copy()
    if "mean_delta_score" in ranked_source:
        ranked_source = ranked_source.loc[
            ranked_source["mean_delta_score"].fillna(float("-inf")).gt(0.0)
        ].copy()
    if ranked_source.empty:
        return ranked_source
    sort_cols = ["positive_benchmark_count", "mean_delta_score", "feature"]
    ranked = ranked_source.sort_values(
        by=sort_cols,
        ascending=[False, False, True],
    )
    if "model_family" in ranked:
        pieces = [
            ranked.loc[ranked["model_family"].eq("classification")],
            ranked.loc[ranked["model_family"].eq("regression")],
            ranked.loc[
                ~ranked["model_family"].isin(["classification", "regression"])
            ],
        ]
        ranked = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
    selected_rows: list[pd.Series] = []
    groups: set[str] = set()
    features: set[str] = set()
    for _, row in ranked.iterrows():
        group = str(row.get("feature_group", ""))
        feature = str(row.get("feature", ""))
        if len(selected_rows) >= max_features:
            break
        if feature in features:
            continue
        if group not in groups and len(groups) >= max_groups:
            continue
        selected_rows.append(row)
        groups.add(group)
        features.add(feature)
    if not selected_rows:
        return ranked.head(0).copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _json_clean(value: Any) -> Any:
    """Return deterministic JSON-safe values without NaN tokens."""
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def prune_correlated_features(
    feature_frame: pd.DataFrame,
    candidate_features: list[str],
    baseline_features: list[str],
    threshold: float = 0.90,
) -> list[str]:
    """Drop TA candidates highly correlated with existing baseline features."""
    kept: list[str] = []
    baseline = [feature for feature in baseline_features if feature in feature_frame.columns]
    for candidate in candidate_features:
        if candidate not in feature_frame.columns:
            continue
        if not baseline:
            kept.append(candidate)
            continue
        corr = feature_frame[[candidate] + baseline].corr().loc[candidate, baseline]
        if corr.abs().max(skipna=True) < threshold:
            kept.append(candidate)
    return kept


def assign_regime_slice(date_like: object) -> str:
    """Map a timestamp to the pre-registered TA regime slice."""
    timestamp = pd.Timestamp(date_like)
    if timestamp < pd.Timestamp("2020-01-01"):
        return "pre_2020"
    if timestamp < pd.Timestamp("2022-01-01"):
        return "covid_2020_2021"
    return "post_2022"


def assign_benchmark_family(benchmark: str) -> str:
    """Return the pre-registered benchmark-family label."""
    if benchmark in {"VOO", "VXUS", "VWO", "VDE"}:
        return "equity"
    if benchmark in {"VMBS", "BND"}:
        return "defensive_rates"
    if benchmark in {"GLD", "DBC"}:
        return "real_assets"
    return "other"


def build_candidate_payload(
    survivors: pd.DataFrame,
    recommendation: str,
) -> dict[str, Any]:
    """Build deterministic candidate JSON for v163/v164 synthesis."""
    allowed = {
        "abandon_ta",
        "monitor_only",
        "shadow_candidate",
        "replacement_candidate",
    }
    if recommendation not in allowed:
        raise ValueError(f"Unsupported recommendation: {recommendation}")
    ordered = survivors.sort_values("feature").reset_index(drop=True)
    features = sorted(ordered["feature"].dropna().astype(str).unique().tolist()) if "feature" in ordered else []
    groups = sorted(ordered["feature_group"].dropna().astype(str).unique().tolist()) if "feature_group" in ordered else []
    rows = _json_clean(ordered.to_dict(orient="records"))
    return {
        "version": "v163",
        "recommendation": recommendation,
        "features": features,
        "feature_groups": groups,
        "rows": rows,
    }


def summarize_benchmark_family_slices(
    detail: pd.DataFrame,
    survivors: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected survivor rows by benchmark-family slices."""
    if detail.empty or survivors.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    key_cols = ["model_family", "model_type", "feature", "experiment_mode"]
    selected_keys = survivors[key_cols].drop_duplicates()
    selected_detail = detail.merge(selected_keys, on=key_cols, how="inner")
    if selected_detail.empty:
        return pd.DataFrame()
    selected_detail["benchmark_family"] = selected_detail["benchmark"].map(
        assign_benchmark_family
    )

    metric_cols = [
        "delta_ic",
        "delta_oos_r2",
        "delta_balanced_accuracy",
        "delta_brier_score",
    ]
    for keys, group in selected_detail.groupby(
        key_cols + ["benchmark_family"],
        dropna=False,
    ):
        model_family, model_type, feature, experiment_mode, benchmark_family = keys
        row: dict[str, Any] = {
            "slice_type": "benchmark_family",
            "slice_label": benchmark_family,
            "model_family": model_family,
            "model_type": model_type,
            "feature": feature,
            "experiment_mode": experiment_mode,
            "n_benchmarks": int(group["benchmark"].nunique()),
        }
        for col in metric_cols:
            if col in group:
                row[f"mean_{col}"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        by=["feature", "model_family", "model_type", "slice_label"],
    )


def infer_recommendation(survivors: pd.DataFrame) -> str:
    """Map the capped survivor set to a conservative v164 outcome label."""
    if survivors.empty:
        return "abandon_ta"
    strong = survivors.loc[
        survivors["positive_benchmark_count"].fillna(0).ge(4)
        & survivors["mean_delta_score"].fillna(0.0).gt(0.0)
    ]
    if strong.empty:
        return "monitor_only"
    replacement = strong["experiment_mode"].astype(str).str.startswith("replace_")
    if replacement.any():
        return "replacement_candidate"
    return "shadow_candidate"


def run_survivor_confirmation(
    screen_summary_path: Path = OUTPUT_DIR / "v162_ta_broad_screen_summary.csv",
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run capped survivor selection from the v162 summary artifact."""
    if not screen_summary_path.exists():
        empty = pd.DataFrame(columns=["feature", "feature_group", "mean_delta_score"])
        payload = build_candidate_payload(empty, recommendation="abandon_ta")
    else:
        screen = pd.read_csv(screen_summary_path)
        if "feature_group" not in screen.columns:
            screen["feature_group"] = screen["feature"].str.extract(r"^(ta_[^_]+_[^_]+)")
        if "mean_delta_score" not in screen.columns:
            screen["mean_delta_score"] = np.nanmean(
                screen[
                    [
                        col
                        for col in [
                            "mean_delta_ic",
                            "mean_delta_oos_r2",
                            "mean_delta_balanced_accuracy",
                        ]
                        if col in screen.columns
                    ]
                ],
                axis=1,
            )
        if "positive_benchmark_count" not in screen.columns:
            screen["positive_benchmark_count"] = 0
        survivors = select_survivors(screen)
        recommendation = infer_recommendation(survivors)
        payload = build_candidate_payload(survivors, recommendation=recommendation)
        output_dir.mkdir(parents=True, exist_ok=True)
        confirm_rows = [survivors.assign(slice_type="all_benchmarks", slice_label="all")]
        detail_path = output_dir / DETAIL_PATH.name
        if detail_path.exists() and not survivors.empty:
            detail = pd.read_csv(detail_path)
            family_slices = summarize_benchmark_family_slices(detail, survivors)
            if not family_slices.empty:
                confirm_rows.append(family_slices)
        pd.concat(confirm_rows, ignore_index=True, sort=False).to_csv(
            output_dir / CONFIRM_PATH.name,
            index=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PAYLOAD_PATH.name).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v163 TA survivor confirmation.")
    parser.add_argument("--screen-summary", default=str(OUTPUT_DIR / "v162_ta_broad_screen_summary.csv"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run_survivor_confirmation(
        screen_summary_path=Path(args.screen_summary),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
