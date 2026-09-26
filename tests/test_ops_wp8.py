"""Review 2026-09-25, step 6 (WP8, F26): workflows, mode and EDGAR access."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import config

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
DB_WRITERS = (
    "initial_fetch_dividends",
    "initial_fetch_prices",
    "monthly_8k_fetch",
    "monthly_decision",
    "peer_bootstrap",
    "peer_data_fetch",
    "post_initial_bootstrap",
    "weekly_data_fetch",
)


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # PyYAML reads the bare key `on` as True.
    return workflow.get("on", workflow.get(True)) or {}


def _steps(workflow: dict) -> list[dict]:
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def _step(workflow: dict, step_id: str) -> dict:
    return next(step for step in _steps(workflow) if step.get("id") == step_id)


def test_every_workflow_that_commits_the_db_shares_one_concurrency_group() -> None:
    writers = {
        path.stem
        for path in WORKFLOWS.glob("*.yml")
        if "data/pgr_financials.db" in path.read_text(encoding="utf-8")
    }
    assert writers == set(DB_WRITERS)
    for name in DB_WRITERS:
        concurrency = _load(name)["concurrency"]
        assert concurrency["group"] == "db-writer", name
        assert concurrency["cancel-in-progress"] is False, name


def test_monthly_decision_runs_after_the_8k_fetch() -> None:
    decision = _triggers(_load("monthly_decision"))
    fetch_name = _load("monthly_8k_fetch")["name"]
    assert decision["workflow_run"]["workflows"] == [fetch_name]
    assert decision["workflow_run"]["types"] == ["completed"]
    # No cron that could start the decision before the 8-K job on the 20th.
    crons = [entry["cron"] for entry in decision.get("schedule", [])]
    assert all(cron.split()[2] not in {"20", "*"} for cron in crons), crons


@pytest.mark.parametrize("name", ["initial_fetch_prices", "initial_fetch_dividends", "post_initial_bootstrap", "peer_bootstrap"])
def test_bootstrap_workflows_are_dispatch_only(name: str) -> None:
    assert set(_triggers(_load(name))) == {"workflow_dispatch"}


def test_monthly_email_charts_and_commit_are_gated_on_generated() -> None:
    workflow = _load("monthly_decision")
    for step_id in ("verify", "charts", "commit"):
        assert "steps.decision.outputs.generated == 'true'" in _step(workflow, step_id)["if"], step_id
    email = next(step for step in _steps(workflow) if step.get("name") == "Send monthly decision email")
    assert "steps.decision.outputs.generated == 'true'" in email["if"]


@pytest.mark.parametrize(
    ("name", "step_name"),
    [
        ("monthly_8k_fetch", "Fetch PGR 8-K operating metrics"),
        ("weekly_data_fetch", "Run weekly fetch"),
    ],
)
def test_edgar_workflows_set_a_real_user_agent(name: str, step_name: str) -> None:
    step = next(s for s in _steps(_load(name)) if s.get("name") == step_name)
    agent = step["env"]["EDGAR_USER_AGENT"]
    assert "@" in agent
    assert agent != config.EDGAR_USER_AGENT_FALLBACK


def test_ci_smoke_tests_run_with_the_network_mocked() -> None:
    step = next(s for s in _steps(_load("ci")) if str(s.get("name", "")).startswith("Smoke test"))
    commands = [line.strip() for line in step["run"].splitlines() if line.strip()]
    assert commands
    assert all(line.startswith("python scripts/ci_offline_smoke.py ") for line in commands)


def test_offline_guard_blocks_requests_and_serves_the_edgar_index(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    import requests

    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    monkeypatch.setattr(requests.Session, "send", requests.Session.send)

    from scripts.ci_offline_smoke import NetworkBlockedError, install_network_guard

    requested = install_network_guard()
    response = requests.get("https://data.sec.gov/submissions/CIK0000080661.json", timeout=5)
    assert response.json()["filings"]["recent"]["form"] == []
    with pytest.raises(NetworkBlockedError):
        requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY", timeout=5)
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("api.stlouisfed.org", 443))
    assert requested[0].startswith("https://data.sec.gov/submissions/")


# ---------------------------------------------------------------------------
# Recommendation-layer mode and the generated output
# ---------------------------------------------------------------------------


def test_unknown_recommendation_layer_mode_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """The typo `live-only` used to fall back to the retired shadow_promoted mode."""
    import scripts.monthly_decision as md

    monkeypatch.setattr(config, "RECOMMENDATION_LAYER_MODE", "live-only")
    monkeypatch.setattr(md, "_already_ran", lambda as_of: pytest.fail("must fail before any work"))
    with pytest.raises(ValueError, match="Unknown RECOMMENDATION_LAYER_MODE 'live-only'"):
        md.main(as_of_date_str="2026-04-02", dry_run=True, skip_fred=True)


@pytest.mark.parametrize("mode", list(config.RECOMMENDATION_LAYER_VALID_MODES))
def test_valid_modes_pass_validation(mode: str) -> None:
    import scripts.monthly_decision as md

    assert md._validate_layer_mode(mode) == mode


def test_skipped_run_writes_generated_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.monthly_decision as md

    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(md, "_already_ran", lambda as_of: True)
    md.main(as_of_date_str="2026-04-02", dry_run=False, skip_fred=True)
    assert output.read_text(encoding="utf-8").strip() == "generated=false"


def test_step_output_is_a_no_op_outside_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.monthly_decision as md

    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    md._write_step_output("generated", "true")  # must not raise


def test_weekly_fetch_reraises_a_missing_edgar_user_agent() -> None:
    """The weekly job logs and continues on EDGAR errors, but a missing
    User-Agent is a configuration error and must fail the run."""
    import scripts.weekly_fetch as wf

    source = Path(wf.__file__).read_text(encoding="utf-8")
    reraise = source.index("except config.EdgarUserAgentError:")
    generic = source.index("except Exception as exc:", reraise)
    assert "raise" in source[reraise:generic]
