"""Review 2026-09-25, step 6 (WP8, F26): workflows, mode and EDGAR access.

The workflow files are read as text (PyYAML is not a project dependency),
as in ``test_workflow_contracts.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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


def _text(name: str) -> str:
    return (WORKFLOWS / f"{name}.yml").read_text(encoding="utf-8")


def _on_block(text: str) -> str:
    """The top-level ``on:`` block (up to the next top-level key)."""
    match = re.search(r"^on:\n((?:[ #].*\n|\n)*)", text, re.MULTILINE)
    assert match, "no top-level on: block"
    return match.group(1)


def _top_level_keys(block: str) -> set[str]:
    return set(re.findall(r"^  ([A-Za-z_]+):", block, re.MULTILINE))


def _steps(text: str) -> list[str]:
    """Each ``- name:`` step of the workflow, as text."""
    parts = re.split(r"\n(?=      - name: )", text)
    # A job's last step ends where the next job (a two-space key) starts.
    return [
        re.split(r"\n(?=  \S)", part, maxsplit=1)[0]
        for part in parts
        if part.startswith("      - name: ")
    ]


def _step(text: str, *, step_id: str | None = None, name: str | None = None) -> str:
    for step in _steps(text):
        if step_id is not None and re.search(rf"^        id: {re.escape(step_id)}$", step, re.MULTILINE):
            return step
        if name is not None and step.startswith(f"      - name: {name}"):
            return step
    raise AssertionError(f"step not found: id={step_id} name={name}")


def test_every_workflow_that_commits_the_db_shares_one_concurrency_group() -> None:
    writers = {
        path.stem
        for path in WORKFLOWS.glob("*.yml")
        if "data/pgr_financials.db" in path.read_text(encoding="utf-8")
    }
    assert writers == set(DB_WRITERS)
    for name in DB_WRITERS:
        match = re.search(
            r"^concurrency:\n(?:  #.*\n)*  group: (\S+)\n  cancel-in-progress: (\S+)$",
            _text(name),
            re.MULTILINE,
        )
        assert match, name
        assert match.group(1) == "db-writer", name
        assert match.group(2) == "false", name


def test_monthly_decision_runs_after_the_8k_fetch() -> None:
    fetch_name = re.search(r"^name: (.+)$", _text("monthly_8k_fetch"), re.MULTILINE).group(1)
    block = _on_block(_text("monthly_decision"))
    assert f'  workflow_run:\n    workflows: ["{fetch_name}"]\n    types: [completed]' in block
    # No cron that could start the decision before the 8-K job on the 20th.
    crons = re.findall(r"cron: '([^']+)'", block)
    assert crons
    assert all(cron.split()[2] not in {"20", "*"} for cron in crons), crons


@pytest.mark.parametrize("name", ["initial_fetch_prices", "initial_fetch_dividends", "post_initial_bootstrap", "peer_bootstrap"])
def test_bootstrap_workflows_are_dispatch_only(name: str) -> None:
    assert _top_level_keys(_on_block(_text(name))) == {"workflow_dispatch"}


def test_monthly_email_charts_and_commit_are_gated_on_generated() -> None:
    text = _text("monthly_decision")
    gate = "steps.decision.outputs.generated == 'true'"
    for step_id in ("verify", "charts", "commit"):
        assert gate in _step(text, step_id=step_id), step_id
    assert gate in _step(text, name="Send monthly decision email")


@pytest.mark.parametrize(
    ("name", "step_name"),
    [
        ("monthly_8k_fetch", "Fetch PGR 8-K operating metrics"),
        ("weekly_data_fetch", "Run weekly fetch"),
    ],
)
def test_edgar_workflows_set_a_real_user_agent(name: str, step_name: str) -> None:
    step = _step(_text(name), name=step_name)
    match = re.search(r'^          EDGAR_USER_AGENT: "([^"]+)"$', step, re.MULTILINE)
    assert match, step_name
    agent = match.group(1)
    assert "@" in agent
    assert agent != config.EDGAR_USER_AGENT_FALLBACK


def test_ci_smoke_tests_run_with_the_network_mocked() -> None:
    step = _step(_text("ci"), name="Smoke test")
    run = step.split("        run: |\n", 1)[1]
    commands = [line.strip() for line in run.splitlines() if line.strip()]
    assert len(commands) == 4
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
