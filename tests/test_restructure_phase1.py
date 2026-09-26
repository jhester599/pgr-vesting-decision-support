"""Review 2026-09-25, section 5, phase 1: production artifacts live under ``artifacts/``.

- The monthly decision folders, the shadow reviews, the monthly ``pgr_*.png``
  charts and the fetch-status log moved (``git mv``) out of ``results/`` and
  ``data/``.
- Their paths are constants in ``config`` (``config/paths.py``), and every
  production reader and writer uses them.
- Every workflow ``git add`` names an exact production path; nothing stages
  ``results/``.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

import config
from src.reporting import classification_artifacts, email_sender

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

MONTHLY_CHARTS: tuple[str, ...] = (
    "pgr_book_value_per_share.png",
    "pgr_book_value_per_share_split_adjusted.png",
    "pgr_share_repurchase_volume.png",
    "pgr_repurchase_dollar_amount.png",
    "pgr_repurchase_dollar_amount_capped.png",
    "pgr_share_price.png",
    "pgr_share_price_split_adjusted.png",
    "pgr_price_to_book.png",
    "pgr_price_to_book_split_adjusted.png",
    "pgr_repurchase_dividend_annual.png",
    "pgr_capital_return_pct_marketcap.png",
    "pgr_cr_vs_capital_return.png",
)


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"phase1_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tracked(prefix: str) -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--", prefix],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if (REPO_ROOT / line).exists()]


def _posix(path: str | Path) -> str:
    return Path(path).as_posix()


# ---------------------------------------------------------------------------
# Config constants and the files themselves
# ---------------------------------------------------------------------------


def test_artifact_paths_are_config_constants() -> None:
    assert _posix(config.ARTIFACTS_DIR) == "artifacts"
    assert _posix(config.MONTHLY_DECISIONS_DIR) == "artifacts/monthly_decisions"
    assert _posix(config.DECISION_LOG_PATH) == "artifacts/monthly_decisions/decision_log.md"
    assert _posix(config.SHADOW_REVIEWS_DIR) == "artifacts/shadow_reviews"
    assert _posix(config.CHARTS_DIR) == "artifacts/charts"
    assert _posix(config.FETCH_STATUS_PATH) == "artifacts/ops/fetch_status.md"
    # Dry runs stay out of the committed tree (gitignored).
    assert _posix(config.DRY_RUN_MONTHLY_DECISIONS_DIR) == "results/dry_run/monthly_decisions"


def test_production_artifacts_moved_out_of_results_and_data() -> None:
    assert _tracked("results/monthly_decisions") == []
    assert _tracked("results/v14/shadow_reviews") == []
    assert _tracked("data/fetch_status.md") == []
    assert [p for p in _tracked("results") if Path(p).name in MONTHLY_CHARTS] == []

    decisions = _tracked(config.MONTHLY_DECISIONS_DIR)
    assert f"{_posix(config.MONTHLY_DECISIONS_DIR)}/2026-09/recommendation.md" in decisions
    assert _posix(config.DECISION_LOG_PATH) in decisions
    memos = [p for p in _tracked(config.SHADOW_REVIEWS_DIR) if Path(p).name != "README.md"]
    assert len(memos) == 6
    charts = [Path(p).name for p in _tracked(config.CHARTS_DIR) if Path(p).name != "README.md"]
    assert sorted(charts) == sorted(MONTHLY_CHARTS)
    assert _tracked(config.FETCH_STATUS_PATH) == [_posix(config.FETCH_STATUS_PATH)]


def test_every_artifact_folder_has_a_readme() -> None:
    for folder in ("", "monthly_decisions", "shadow_reviews", "charts", "ops"):
        readme = REPO_ROOT / "artifacts" / folder / "README.md"
        assert readme.is_file(), readme


# ---------------------------------------------------------------------------
# Readers and writers use the constants
# ---------------------------------------------------------------------------


def test_monthly_decision_output_dirs() -> None:
    module = _script("monthly_decision")
    as_of = date(2026, 9, 21)
    assert _posix(module._output_dir(as_of)) == "artifacts/monthly_decisions/2026-09"
    assert _posix(module._dry_run_output_dir(as_of)) == "results/dry_run/monthly_decisions/2026-09"
    source = inspect.getsource(module)
    assert 'Path("results") / "monthly_decisions"' not in source
    assert "results/monthly_decisions/decision_log.md" not in source


def test_classification_history_paths_default_to_artifacts() -> None:
    assert _posix(classification_artifacts.classification_history_path()) == (
        "artifacts/monthly_decisions/classification_shadow_history.csv"
    )
    assert _posix(classification_artifacts.ta_shadow_variant_history_path()) == (
        "artifacts/monthly_decisions/ta_shadow_variant_history.csv"
    )


def test_email_reads_report_and_charts_from_artifacts(tmp_path: Path, monkeypatch) -> None:
    assert _posix(email_sender._CHARTS_DIR) == "artifacts/charts"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="artifacts/monthly_decisions/2026-09/recommendation.md"):
        email_sender.send_monthly_email(month_label="2026-09", dry_run=True)


def test_chart_scripts_write_to_artifacts_charts() -> None:
    for name in ("capital_return_charts", "repurchase_timeseries_charts"):
        module = _script(name)
        assert Path(module.OUT_DIR).resolve() == (REPO_ROOT / config.CHARTS_DIR).resolve(), name


def test_initial_fetch_default_status_file() -> None:
    module = _script("initial_fetch")
    assert _posix(module._DEFAULT_STATUS_FILE) == "artifacts/ops/fetch_status.md"


def test_verify_and_replay_read_artifacts() -> None:
    verify = _script("verify_monthly_outputs")
    default = inspect.signature(verify.verify_monthly_outputs).parameters["base_dir"].default
    assert _posix(default) == "artifacts/monthly_decisions"
    replay = _script("replay_monthly_decisions")
    assert replay.COMMITTED_DIR == REPO_ROOT / config.MONTHLY_DECISIONS_DIR
    assert replay.DRY_RUN_DIR == REPO_ROOT / config.DRY_RUN_MONTHLY_DECISIONS_DIR


def test_dashboard_reads_artifacts() -> None:
    """dashboard/data.py cannot import config under `streamlit run`; it mirrors it."""
    from dashboard import data as dashboard_data

    expected = (REPO_ROOT / config.MONTHLY_DECISIONS_DIR).resolve()
    assert dashboard_data.DECISIONS_DIR.resolve() == expected
    assert dashboard_data.DECISION_LOG.resolve() == (REPO_ROOT / config.DECISION_LOG_PATH).resolve()


# ---------------------------------------------------------------------------
# Workflows stage exact paths
# ---------------------------------------------------------------------------

ALLOWED_GIT_ADD_PATHS: frozenset[str] = frozenset(
    {
        "data/pgr_financials.db",
        "artifacts/monthly_decisions/",
        '"artifacts/charts/$chart"',
        "artifacts/ops/fetch_status.md",
    }
)


def _git_add_lines() -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("git add"):
                lines.append((path.name, stripped))
    return lines


def test_workflows_stage_only_exact_production_paths() -> None:
    lines = _git_add_lines()
    assert lines
    offenders: list[str] = []
    for name, line in lines:
        args = line.split("#", 1)[0].split()[2:]
        if "||" in args or not args or any(a not in ALLOWED_GIT_ADD_PATHS for a in args):
            offenders.append(f"{name}: {line}")
    assert offenders == []


def test_workflows_write_to_artifact_paths() -> None:
    text = {p.name: p.read_text(encoding="utf-8") for p in WORKFLOWS.glob("*.yml")}
    for name in ("initial_fetch_prices.yml", "initial_fetch_dividends.yml"):
        assert "--status-file artifacts/ops/fetch_status.md" in text[name]
        assert "data/fetch_status.md" not in text[name]
    monthly = text["monthly_decision.yml"]
    assert re.search(r'"artifacts/charts/\$chart" -nt', monthly)
    assert "results/" not in monthly
    assert "git add results" not in "".join(text.values())
