from __future__ import annotations

import ast
import inspect
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import yaml

from scripts import initial_fetch


def _workflow(name: str) -> dict[str, Any]:
    return yaml.safe_load(_read(f".github/workflows/{name}.yml"))


def test_peer_bootstrap_summary_uses_existing_date_column(
    tmp_path: Path,
) -> None:
    """Execute the actual summary heredoc against the production schema."""
    db_path = tmp_path / "data" / "pgr_financials.db"
    db_path.parent.mkdir()
    schema = Path("src/database/schema.sql").read_text(encoding="utf-8")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(schema)
        for ticker in ("ALL", "TRV", "CB", "HIG"):
            conn.executemany(
                "INSERT INTO daily_prices (ticker, date, close) "
                "VALUES (?, ?, ?)",
                [(ticker, "2020-02-07", 12.0), (ticker, "2020-01-03", 10.0)],
            )
            conn.execute(
                "INSERT INTO daily_dividends (ticker, ex_date, amount) "
                "VALUES (?, '2020-01-10', 0.5)",
                (ticker,),
            )
        conn.commit()
    steps = _workflow("peer_bootstrap")["jobs"]["peer-bootstrap"]["steps"]
    run = next(
        step["run"] for step in steps if step["name"] == "Bootstrap summary"
    )
    lines = run.splitlines()
    assert lines[0] == "python - <<'EOF'" and lines[-1] == "EOF"
    code = "\n".join(lines[1:-1])
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for ticker in ("ALL", "TRV", "CB", "HIG"):
        assert (
            f"{ticker}: 2 prices (from 2020-01-03), 1 dividends" in proc.stdout
        )


def test_ci_has_read_only_permissions() -> None:
    assert _workflow("ci")["permissions"] == {"contents": "read"}


def test_ci_entrypoint_smokes_use_an_external_checkout_copy() -> None:
    steps = _workflow("ci")["jobs"]["test"]["steps"]
    run = next(
        s["run"]
        for s in steps
        if s["name"] == "Smoke test production entrypoints (network mocked)"
    )
    copy = run.index('cp -a . "$SMOKE_DIR/repo"')
    chdir = run.index('cd "$SMOKE_DIR/repo"')
    install = run.index("python -m pip install --no-deps")
    smoke = run.index("python scripts/ci_offline_smoke.py")
    assert 'SMOKE_DIR="$(mktemp -d)"' in run
    assert copy < chdir < install < smoke


def test_ci_runs_windows_safety_regressions_on_python312() -> None:
    jobs = _workflow("ci")["jobs"]
    job = jobs["windows-regressions"]
    assert job["runs-on"] == "windows-latest"
    setup = next(
        s for s in job["steps"] if s.get("uses") == "actions/setup-python@v6"
    )
    assert setup["with"]["python-version"] == "3.12"
    commands = "\n".join(s.get("run", "") for s in job["steps"])
    for filename in (
        "tests/integration/repo/test_test_suite_hygiene.py",
        "tests/integration/repo/test_restructure_phase1.py",
        "tests/integration/repo/test_workflow_contracts.py",
        "tests/unit/scripts/test_capital_return_charts.py",
    ):
        assert filename in commands
    assert "pytest" in commands and "skip" not in commands
    for name in ("test", "research", "artifacts"):
        assert jobs[name]["runs-on"] == "ubuntu-latest"


def test_initial_fetch_removes_ignored_force_option(tmp_path: Path) -> None:
    """Reject the old no-op flag before any fetch code can run."""
    assert "force" not in inspect.signature(initial_fetch.main).parameters
    proc = subprocess.run(
        [sys.executable, str(Path(initial_fetch.__file__)), "--force"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "unrecognized arguments: --force" in proc.stderr
    tree = ast.parse(_read("scripts/initial_fetch.py"))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "force"
        for node in ast.walk(tree)
    )
    text = _read(".github/workflows/initial_fetch_prices.yml")
    assert "--force" not in text and "inputs.force" not in text
    # PyYAML's YAML 1.1 resolver treats the unquoted GitHub 'on' key as True.
    workflow = _workflow("initial_fetch_prices")
    triggers = workflow.get("on", workflow.get(True))
    assert "force" not in triggers["workflow_dispatch"]["inputs"]


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
    # Smoke runs go through the network-blocking wrapper (review 2026-09-25, F26).
    assert "python scripts/ci_offline_smoke.py scripts/weekly_fetch.py --dry-run --skip-fred" in text
    assert (
        "python scripts/ci_offline_smoke.py cli/monthly_decision.py "
        "--as-of 2026-04-02 --dry-run --skip-fred"
    ) in text


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


def test_weekly_workflow_refreshes_dividends_and_checks_integrity() -> None:
    """Review F08: a budget-aware dividend refresh runs on its own cron, and
    the integrity check (split jumps, duplicate week bars, dividend
    freshness) runs after the DB commit."""
    text = _read(".github/workflows/weekly_data_fetch.yml")
    assert "- cron: '0 16 * * 3'" in text
    assert 'MODE_FLAG="--dividend-refresh"' in text
    assert "python scripts/weekly_fetch.py $MODE_FLAG $DRY_FLAG" in text
    assert "python scripts/check_data_integrity.py $STALE_FLAG" in text
    assert text.index("Commit updated database") < text.index(
        "python scripts/check_data_integrity.py"
    )
