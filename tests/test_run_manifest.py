from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.reporting.run_manifest import build_run_manifest, write_run_manifest
from src.reporting.run_manifest import resolve_git_sha


def test_build_run_manifest_includes_expected_fields() -> None:
    manifest = build_run_manifest(
        workflow_name="monthly_decision",
        script_name="scripts/monthly_decision.py",
        as_of_date=date(2026, 4, 2),
        schema_version="001_initial",
        latest_dates={"daily_prices.date": "2026-04-02"},
        row_counts={"daily_prices": 123},
        warnings=["sample warning"],
        outputs=["results/monthly_decisions/2026-04/recommendation.md"],
        artifact_classification="production",
    )

    assert manifest["workflow_name"] == "monthly_decision"
    assert manifest["schema_version"] == "001_initial"
    assert manifest["warnings"] == ["sample warning"]
    assert manifest["as_of_date"] == "2026-04-02"


def test_write_run_manifest_writes_json_file(tmp_path: Path) -> None:
    manifest = build_run_manifest(
        workflow_name="monthly_decision",
        script_name="scripts/monthly_decision.py",
        as_of_date=date(2026, 4, 2),
        schema_version="001_initial",
    )
    path = write_run_manifest(tmp_path, manifest)
    written = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "run_manifest.json"
    assert written["workflow_name"] == "monthly_decision"


def test_resolve_git_sha_logs_and_returns_unknown_on_git_failure(caplog) -> None:
    with caplog.at_level("ERROR"), patch.dict(
        "os.environ",
        {},
        clear=True,
    ), patch(
        "src.reporting.run_manifest.subprocess.run",
        side_effect=RuntimeError("synthetic git failure"),
    ):
        sha = resolve_git_sha()

    assert sha == "unknown"
    assert "Could not resolve git SHA for run manifest" in caplog.text
    assert "synthetic git failure" in caplog.text
