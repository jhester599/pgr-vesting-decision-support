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


def write_run_manifest(out_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write a manifest JSON file and return the path."""
    path = Path(out_dir) / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
