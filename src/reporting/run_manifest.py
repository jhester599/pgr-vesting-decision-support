"""Helpers for machine-readable run manifests."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def resolve_git_sha(cwd: str | None = None) -> str:
    """Return the current git SHA from env or local git."""
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        head_sha = result.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if status.stdout.strip():
            return f"{head_sha}-dirty"
        return head_sha
    except Exception as exc:
        logger.exception(
            "Could not resolve git SHA for run manifest; using 'unknown'. Error=%r",
            exc,
        )
        return "unknown"


def build_run_manifest(
    *,
    workflow_name: str,
    script_name: str,
    as_of_date: date | None,
    schema_version: str | None,
    latest_dates: dict[str, Any] | None = None,
    row_counts: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    outputs: list[str] | None = None,
    artifact_classification: str = "production",
    cwd: str | None = None,
) -> dict[str, Any]:
    """Build a serializable manifest for a major production run."""
    return {
        "run_timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "workflow_name": workflow_name,
        "script_name": script_name,
        "git_sha": resolve_git_sha(cwd=cwd),
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
        "schema_version": schema_version,
        "artifact_classification": artifact_classification,
        "latest_dates": latest_dates or {},
        "row_counts": row_counts or {},
        "warnings": warnings or [],
        "outputs": outputs or [],
    }


def build_run_manifest_gates(
    readiness: dict[str, Any] | None,
    recommendation_mode: dict[str, Any] | None,
) -> dict[str, Any]:
    """The ``decision_gates`` block of a monthly run manifest (R3).

    Records the gate contract version, the readiness contract behind the
    ``wfo_completed`` and ``data_ready`` gates, and the recommendation mode
    with its failed gates and reasons. Unknown values stay ``None``; they are
    never filled with a passing default.
    """
    import config

    readiness = readiness or {}
    mode = recommendation_mode or {}
    return {
        "gate_contract_version": config.DECISION_GATE_CONTRACT_VERSION,
        "metrics_version": config.MODEL_HEALTH_METRICS_VERSION,
        "recommendation_mode": mode.get("label"),
        "recommended_sell_pct": mode.get("sell_pct"),
        "failed_gates": list(mode.get("failed_gates") or []),
        "deferral_reasons": list(mode.get("deferral_reasons") or []),
        "wfo_completed": readiness.get("wfo_completed"),
        "wfo_required_pairs": readiness.get("wfo_required_pairs"),
        "wfo_failed_pairs": readiness.get("wfo_failed_pairs"),
        "wfo_optional_excluded": readiness.get("wfo_optional_excluded"),
        "data_ready": readiness.get("data_ready"),
        "missing_live_features": readiness.get("missing_live_features"),
        "stale_required_feeds": readiness.get("stale_required_feeds"),
        "decision_row_date": readiness.get("decision_row_date"),
        "readiness_basis": readiness.get("readiness_basis"),
        "readiness_note": readiness.get("readiness_note"),
    }


def write_run_manifest(out_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write a manifest JSON file and return the path."""
    path = Path(out_dir) / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
