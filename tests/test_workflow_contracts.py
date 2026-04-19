from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_production_workflows_use_current_action_versions_and_concurrency() -> None:
    workflow_paths = [
        ".github/workflows/weekly_data_fetch.yml",
        ".github/workflows/peer_data_fetch.yml",
        ".github/workflows/monthly_8k_fetch.yml",
        ".github/workflows/monthly_decision.yml",
    ]

    for workflow_path in workflow_paths:
        text = _read(workflow_path)
        assert "uses: actions/checkout@v5" in text
        assert "uses: actions/setup-python@v6" in text
        assert "concurrency:" in text


def test_ci_workflow_runs_lint_tests_and_smokes() -> None:
    text = _read(".github/workflows/ci.yml")
    assert "ruff check ." in text
    assert "python -m pytest -q" in text
    assert "python scripts/weekly_fetch.py --dry-run --skip-fred" in text
    assert "python scripts/monthly_decision.py --as-of 2026-04-02 --dry-run --skip-fred" in text


def test_monthly_decision_workflow_verifies_manifest() -> None:
    text = _read(".github/workflows/monthly_decision.yml")
    assert "scripts/verify_monthly_outputs.py" in text
    assert "--summary-path workflow_summary.md" in text
    assert "$GITHUB_STEP_SUMMARY" in text


def test_monthly_8k_workflow_verifies_calendar_aware_freshness() -> None:
    text = _read(".github/workflows/monthly_8k_fetch.yml")

    assert "check_data_freshness" in text
    assert "PGR monthly EDGAR" in text
    assert "expected_month_end" in text
