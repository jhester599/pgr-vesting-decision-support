"""Fail-closed study entry points on isolated artifact directories."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shlex
from types import ModuleType

import pandas as pd
import pytest
import yaml


def isolated_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    """Load the runner without changing import paths or reading the DB."""
    source = (
        Path(__file__).resolve().parents[2]
        / "research/studies/v200_clean_baseline/run.py"
    )
    spec = importlib.util.spec_from_file_location("v200_runner", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/pgr_financials.db").write_bytes(b"isolated fixture")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "STUDY", tmp_path)
    monkeypatch.setattr(module, "OUTPUTS", tmp_path / "outputs")
    return module


def test_failed_preflight_writes_plain_blocked_readme_without_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = isolated_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "export_git_blob", lambda *args: tmp_path)

    def failed_preflight(*args: object) -> None:
        raise ValueError("independent availability gate failed")

    monkeypatch.setattr(module, "preflight", failed_preflight)
    with pytest.raises(ValueError, match="availability gate failed"):
        module.prepare(tmp_path / "scratch")
    assert (tmp_path / "README.md").read_text().startswith("v200 is blocked")
    lock = json.loads((tmp_path / "outputs/baseline_lock.json").read_text())
    assert lock["status"] == "blocked"
    assert not (tmp_path / "outputs/predictions.csv").exists()


def test_absent_lock_stops_before_fit_and_records_zero_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = isolated_runner(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        module.run(tmp_path / "scratch", tmp_path / "outputs", False)
    attempt = json.loads(
        (tmp_path / "outputs/blocked_attempt.json").read_text()
    )
    assert attempt["fit_count"] == 0
    assert (tmp_path / "README.md").read_text().startswith("v200 is blocked")


def test_ci_monthly_smoke_label_ends_before_quarantine() -> None:
    """Even the smoke's longest horizon stays inside development history."""
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["test"]["steps"]
    smoke = next(
        step for step in steps
        if step.get("name") == (
            "Smoke test production entrypoints (network mocked)"
        )
    )
    command = next(
        line for line in smoke["run"].splitlines()
        if "cli/monthly_decision.py" in line
    )
    arguments = shlex.split(command)
    assert "--dry-run" in arguments
    assert "--skip-fred" in arguments
    as_of = pd.Timestamp(arguments[arguments.index("--as-of") + 1])
    latest_label_end = as_of + pd.offsets.BMonthEnd(12)
    assert latest_label_end < pd.Timestamp("2023-09-29"), (
        f"CI smoke reaches the research quarantine: {latest_label_end.date()}"
    )
