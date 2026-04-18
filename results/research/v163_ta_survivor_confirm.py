"""v163 survivor confirmation helpers for the TA research arc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("results") / "research"
CONFIRM_PATH = OUTPUT_DIR / "v163_ta_survivor_confirm_summary.csv"
PAYLOAD_PATH = OUTPUT_DIR / "v163_ta_survivor_candidate.json"


def select_survivors(
    screen_summary: pd.DataFrame,
    max_features: int = 6,
    max_groups: int = 3,
) -> pd.DataFrame:
    """Select a capped, group-diversified survivor set from screen results."""
    if screen_summary.empty:
        return screen_summary.copy()
    ranked = screen_summary.sort_values(
        by=["positive_benchmark_count", "mean_delta_score", "feature"],
        ascending=[False, False, True],
    )
    selected_rows: list[pd.Series] = []
    groups: set[str] = set()
    for _, row in ranked.iterrows():
        group = str(row.get("feature_group", ""))
        if len(selected_rows) >= max_features:
            break
        if group not in groups and len(groups) >= max_groups:
            continue
        selected_rows.append(row)
        groups.add(group)
    if not selected_rows:
        return ranked.head(0).copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


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
    features = ordered["feature"].astype(str).tolist() if "feature" in ordered else []
    groups = sorted(ordered["feature_group"].dropna().astype(str).unique().tolist()) if "feature_group" in ordered else []
    rows = ordered.to_dict(orient="records")
    return {
        "version": "v163",
        "recommendation": recommendation,
        "features": features,
        "feature_groups": groups,
        "rows": rows,
    }


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
        recommendation = "monitor_only" if not survivors.empty else "abandon_ta"
        payload = build_candidate_payload(survivors, recommendation=recommendation)
        output_dir.mkdir(parents=True, exist_ok=True)
        survivors.to_csv(output_dir / CONFIRM_PATH.name, index=False)

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
