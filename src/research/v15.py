"""Shared helpers for the v15 fixed-budget feature-replacement cycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import config
from scripts.candidate_model_bakeoff import candidate_feature_sets


V15_FORECAST_UNIVERSE: list[str] = list(config.V13_REDEPLOY_UNIVERSE)
V15_TARGET_CANDIDATES: tuple[str, ...] = (
    "ridge_lean_v1",
    "gbt_lean_plus_two",
)
V15_DEPLOYED_MODELS: tuple[str, ...] = (
    "elasticnet",
    "ridge",
    "bayesian_ridge",
    "gbt",
)
V15_RESEARCH_STATUS: str = "phase_planned"

REQUIRED_INVENTORY_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "category",
    "replace_or_compete_with",
    "definition",
    "economic_rationale",
    "expected_direction",
    "likely_frequency",
    "likely_source",
    "implementation_difficulty",
    "likely_signal_quality",
    "why_it_might_outperform_existing_feature",
    "key_risks",
    "target_model",
    "priority_rank",
)

OPTIONAL_INVENTORY_COLUMNS: tuple[str, ...] = (
    "research_source",
    "status",
    "notes",
)


@dataclass(frozen=True)
class V15ModelSpec:
    """Canonical v15 model specification."""

    candidate_name: str
    model_type: str
    features: list[str]


def base_model_specs() -> dict[str, V15ModelSpec]:
    """Return the v15 starting-point model specifications."""
    specs = candidate_feature_sets()
    result = {
        "ridge_lean_v1": V15ModelSpec(
            candidate_name="ridge_lean_v1",
            model_type=str(specs["ridge_lean_v1"]["model_type"]),
            features=list(specs["ridge_lean_v1"]["features"]),
        ),
        "gbt_lean_plus_two": V15ModelSpec(
            candidate_name="gbt_lean_plus_two",
            model_type=str(specs["gbt_lean_plus_two"]["model_type"]),
            features=list(specs["gbt_lean_plus_two"]["features"]),
        ),
    }
    surgery_summary = _latest_v14_feature_surgery_summary()
    if surgery_summary is not None:
        for row in surgery_summary.itertuples(index=False):
            candidate_name = str(row.candidate_name)
            if candidate_name not in result:
                continue
            result[candidate_name] = V15ModelSpec(
                candidate_name=candidate_name,
                model_type=str(row.model_type),
                features=[value for value in str(row.feature_columns).split(",") if value],
            )
    return result


def ensemble_members() -> list[str]:
    """Return the ensemble members carried forward from v14."""
    return ["ridge_lean_v1", "gbt_lean_plus_two"]


def deployed_model_specs() -> dict[str, V15ModelSpec]:
    """Return the currently deployed model baselines for cross-model confirmation."""
    core_specs = base_model_specs()
    return {
        "elasticnet_current": V15ModelSpec(
            candidate_name="elasticnet_current",
            model_type="elasticnet",
            features=list(config.MODEL_FEATURE_OVERRIDES["elasticnet"]),
        ),
        "ridge_lean_v1": core_specs["ridge_lean_v1"],
        "bayesian_ridge_current": V15ModelSpec(
            candidate_name="bayesian_ridge_current",
            model_type="bayesian_ridge",
            features=list(config.MODEL_FEATURE_OVERRIDES["bayesian_ridge"]),
        ),
        "gbt_lean_plus_two": core_specs["gbt_lean_plus_two"],
    }


def build_inventory_template() -> pd.DataFrame:
    """Return an empty, schema-complete feature candidate inventory."""
    columns = list(REQUIRED_INVENTORY_COLUMNS) + list(OPTIONAL_INVENTORY_COLUMNS)
    return pd.DataFrame(columns=columns)


def validate_inventory(inventory_df: pd.DataFrame) -> None:
    """Validate the feature candidate inventory schema."""
    missing = [column for column in REQUIRED_INVENTORY_COLUMNS if column not in inventory_df.columns]
    if missing:
        raise ValueError(f"Inventory is missing required columns: {missing}")


def normalize_inventory(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize inventory values into a predictable internal format."""
    validate_inventory(inventory_df)
    normalized = inventory_df.copy()
    for column in REQUIRED_INVENTORY_COLUMNS + OPTIONAL_INVENTORY_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    normalized["feature_name"] = normalized["feature_name"].astype(str).str.strip()
    normalized["replace_or_compete_with"] = normalized["replace_or_compete_with"].fillna("").astype(str)
    normalized["target_model"] = (
        normalized["target_model"]
        .fillna("both")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"ridge+gbt": "both", "all": "both"})
    )
    normalized["research_source"] = normalized["research_source"].fillna("").astype(str).str.strip()
    normalized["status"] = normalized["status"].fillna("queued").astype(str).str.strip().str.lower()
    normalized["notes"] = normalized["notes"].fillna("").astype(str).str.strip()
    normalized["priority_rank"] = pd.to_numeric(normalized["priority_rank"], errors="coerce")
    normalized = normalized[normalized["feature_name"] != ""].reset_index(drop=True)
    return normalized


def build_existing_feature_inventory(specs: dict[str, V15ModelSpec]) -> pd.DataFrame:
    """Return a summary of the currently active v15 baseline features."""
    rows: list[dict[str, Any]] = []
    for spec in specs.values():
        for position, feature in enumerate(spec.features, start=1):
            rows.append(
                {
                    "candidate_name": spec.candidate_name,
                    "model_type": spec.model_type,
                    "feature_name": feature,
                    "feature_position": position,
                }
            )
    return pd.DataFrame(rows)


def build_swap_queue(
    inventory_df: pd.DataFrame,
    specs: dict[str, V15ModelSpec],
    available_features: set[str],
) -> pd.DataFrame:
    """Expand candidate inventory rows into model-specific one-for-one swap tests."""
    normalized = normalize_inventory(inventory_df)
    rows: list[dict[str, Any]] = []

    for row in normalized.itertuples(index=False):
        replace_tokens = [
            value.strip()
            for value in str(row.replace_or_compete_with).split(",")
            if value.strip()
        ]
        for spec_name, spec in specs.items():
            target_model = str(row.target_model)
            if target_model not in {"both", spec.model_type}:
                continue
            for replace_feature in replace_tokens:
                if replace_feature not in spec.features:
                    continue
                rows.append(
                    {
                        "candidate_name": spec_name,
                        "model_type": spec.model_type,
                        "candidate_feature": row.feature_name,
                        "replace_feature": replace_feature,
                        "candidate_available_now": row.feature_name in available_features,
                        "replacement_present_in_model": replace_feature in spec.features,
                        "priority_rank": row.priority_rank,
                        "category": row.category,
                        "likely_source": row.likely_source,
                        "implementation_difficulty": row.implementation_difficulty,
                        "likely_signal_quality": row.likely_signal_quality,
                        "research_source": row.research_source,
                        "status": row.status,
                        "notes": row.notes,
                    }
                )

    queue_df = pd.DataFrame(rows)
    if queue_df.empty:
        return queue_df
    return queue_df.sort_values(
        by=["priority_rank", "candidate_name", "replace_feature", "candidate_feature"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def apply_one_for_one_swap(
    spec: V15ModelSpec,
    replace_feature: str,
    candidate_feature: str,
) -> list[str]:
    """Return the feature list produced by one one-for-one swap."""
    if replace_feature not in spec.features:
        raise ValueError(f"Feature '{replace_feature}' is not in baseline spec '{spec.candidate_name}'.")
    swapped = [candidate_feature if feature == replace_feature else feature for feature in spec.features]
    deduped: list[str] = []
    seen: set[str] = set()
    for feature in swapped:
        if feature in seen:
            continue
        deduped.append(feature)
        seen.add(feature)
    return deduped


def choose_phase0_winners(
    summary_df: pd.DataFrame,
    *,
    max_per_model: int = 5,
    min_policy_uplift: float = 0.0,
    min_oos_r2_delta: float = -0.01,
) -> pd.DataFrame:
    """Select the most promising v15.0 swaps for cross-model confirmation."""
    if summary_df.empty:
        return pd.DataFrame()

    filtered = summary_df.copy()
    filtered = filtered[
        (filtered["candidate_available_now"] == True)  # noqa: E712
        & (filtered["mean_policy_return_sign_delta"] >= min_policy_uplift)
        & (filtered["mean_oos_r2_delta"] >= min_oos_r2_delta)
    ].copy()

    if filtered.empty:
        filtered = summary_df[summary_df["candidate_available_now"] == True].copy()  # noqa: E712

    ordered = filtered.sort_values(
        by=[
            "model_type",
            "mean_policy_return_sign_delta",
            "mean_oos_r2_delta",
            "mean_ic_delta",
            "priority_rank",
        ],
        ascending=[True, False, False, False, True],
    )
    return (
        ordered.groupby("model_type", dropna=False, group_keys=False)
        .head(max_per_model)
        .reset_index(drop=True)
    )


def build_confirmation_queue(
    phase0_winners_df: pd.DataFrame,
    specs: dict[str, V15ModelSpec],
) -> pd.DataFrame:
    """Expand v15.0 winners into the v15.1 cross-model confirmation queue."""
    if phase0_winners_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for row in phase0_winners_df.itertuples(index=False):
        for spec_name, spec in specs.items():
            if str(row.replace_feature) not in spec.features:
                continue
            rows.append(
                {
                    "candidate_name": spec_name,
                    "model_type": spec.model_type,
                    "candidate_feature": row.candidate_feature,
                    "replace_feature": row.replace_feature,
                    "priority_rank": row.priority_rank,
                    "research_source": getattr(row, "research_source", ""),
                    "source_phase0_model": row.model_type,
                }
            )

    queue_df = pd.DataFrame(rows)
    if queue_df.empty:
        return queue_df
    return queue_df.sort_values(
        by=["priority_rank", "candidate_name", "replace_feature", "candidate_feature"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def choose_best_confirmed_swaps(
    summary_df: pd.DataFrame,
    *,
    min_policy_uplift: float = 0.0,
    min_oos_r2_delta: float = -0.01,
) -> pd.DataFrame:
    """Pick one best confirmed swap per deployed model for v15.2."""
    if summary_df.empty:
        return pd.DataFrame()

    filtered = summary_df[
        (summary_df["mean_policy_return_sign_delta"] >= min_policy_uplift)
        & (summary_df["mean_oos_r2_delta"] >= min_oos_r2_delta)
    ].copy()
    if filtered.empty:
        filtered = summary_df.copy()

    ordered = filtered.sort_values(
        by=[
            "model_type",
            "mean_policy_return_sign_delta",
            "mean_oos_r2_delta",
            "mean_ic_delta",
            "priority_rank",
        ],
        ascending=[True, False, False, False, True],
    )
    return (
        ordered.groupby("model_type", dropna=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def _latest_v14_feature_surgery_summary() -> pd.DataFrame | None:
    """Load the most recent v14 feature-surgery summary when available."""
    v14_dir = Path("results") / "v14"
    files = sorted(v14_dir.glob("v14_feature_surgery_summary_*.csv"))
    if not files:
        return None
    return pd.read_csv(files[-1])
