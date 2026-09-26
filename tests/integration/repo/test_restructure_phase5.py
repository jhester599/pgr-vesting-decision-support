"""Repository layout after phase 5 of the restructure (review 2026-09-25, section 5).

Phase 5 split the two monoliths: ``scripts/monthly_decision.py`` into
``src/pgr_vds/decision/`` behind ``cli/monthly_decision.py`` (and out of its
mypy exemption), and the two ``edgar_8k_fetcher.py`` files into
``src/pgr_vds/ingestion/edgar_monthly/`` behind ``cli/edgar_monthly_fetch.py``.
The golden-output comparison that proves the split changed no output is
``scripts/checks/golden_replay.py``; its comparison is tested here too.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PGR_VDS = REPO_ROOT / "src" / "pgr_vds"
DECISION = PGR_VDS / "decision"
EDGAR = PGR_VDS / "ingestion" / "edgar_monthly"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

DECISION_MODULES = (
    "schedule",
    "refresh",
    "signal_generation",
    "health",
    "tax_lots",
    "portfolio",
    "rendering",
    "recommendation_report",
    "diagnostic_report",
    "artifacts",
    "pipeline",
)
EDGAR_MODULES = ("fetch", "parse", "derive", "load")
CLI_ENTRY_POINTS = ("cli/monthly_decision.py", "cli/edgar_monthly_fetch.py")
RETIRED = (
    "scripts/monthly_decision.py",
    "scripts/edgar_8k_fetcher.py",
    "src/ingestion/edgar_8k_fetcher.py",
    "src/ingestion/pgr_monthly_loader.py",
)


def _tracked_python() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if (REPO_ROOT / line).is_file()]


def _golden() -> ModuleType:
    path = REPO_ROOT / "scripts" / "checks" / "golden_replay.py"
    spec = importlib.util.spec_from_file_location("golden_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The monthly decision
# ---------------------------------------------------------------------------


def test_monthly_decision_monolith_is_split() -> None:
    assert not (REPO_ROOT / "scripts" / "monthly_decision.py").exists()
    for name in DECISION_MODULES:
        assert (DECISION / f"{name}.py").is_file(), name
    assert (DECISION / "__init__.py").is_file()


def test_no_decision_module_is_a_monolith() -> None:
    sizes = {p.name: len(p.read_text(encoding="utf-8").splitlines()) for p in DECISION.glob("*.py")}
    assert max(sizes.values()) < 1_000, sizes


def test_cli_is_thin() -> None:
    source = (REPO_ROOT / "cli" / "monthly_decision.py").read_text(encoding="utf-8")
    assert len(source.splitlines()) < 100
    assert "from pgr_vds.decision.pipeline import main" in source
    tree = ast.parse(source)
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert functions == {"parse_args", "main"}


def test_mypy_exemption_is_gone() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    overrides = pyproject["tool"]["mypy"].get("overrides", [])
    assert not [o for o in overrides if o.get("ignore_errors")], overrides
    assert "scripts.monthly_decision" not in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_ci_type_checks_the_new_package() -> None:
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "python -m mypy --follow-imports=silent src/pgr_vds cli" in ci


# ---------------------------------------------------------------------------
# The EDGAR monthly pipeline
# ---------------------------------------------------------------------------


def test_edgar_fetchers_are_merged() -> None:
    for name in EDGAR_MODULES:
        assert (EDGAR / f"{name}.py").is_file(), name
    for rel in RETIRED:
        assert not (REPO_ROOT / rel).exists(), rel


def test_one_csv_loader() -> None:
    """``load_from_csv`` is the only function that seeds pgr_edgar_monthly from the CSV."""
    definitions = []
    for path in _tracked_python():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "load_from_csv"
            for node in ast.walk(tree)
        ):
            definitions.append(path.relative_to(REPO_ROOT).as_posix())
    assert definitions == ["src/pgr_vds/ingestion/edgar_monthly/load.py"]


def test_nothing_imports_the_retired_modules() -> None:
    pattern = re.compile(
        r"^\s*(?:from|import)\s+(?:scripts\.monthly_decision|scripts\.edgar_8k_fetcher"
        r"|src\.ingestion\.edgar_8k_fetcher|src\.ingestion\.pgr_monthly_loader)\b"
        r"|^\s*from\s+scripts\s+import\s+(?:monthly_decision|edgar_8k_fetcher)\b"
        r"|^\s*from\s+src\.ingestion\s+import\s+(?:edgar_8k_fetcher|pgr_monthly_loader)\b",
        re.MULTILINE,
    )
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _tracked_python()
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# Package, entry points and workflows
# ---------------------------------------------------------------------------


def test_pgr_vds_has_one_import_name() -> None:
    import pgr_vds.decision

    # Installed from this checkout's src/pgr_vds/ (pip install -e .).
    assert Path(pgr_vds.decision.__file__).resolve().is_relative_to(DECISION)
    with pytest.raises(ImportError, match="import pgr_vds, not src.pgr_vds"):
        importlib.import_module("src.pgr_vds")


@pytest.mark.parametrize("rel_path", CLI_ENTRY_POINTS)
def test_cli_entry_point_runs_main(rel_path: str, tmp_path: Path) -> None:
    """``--help`` exits 0 with usage text: the file has a ``__main__`` block
    that calls ``main``, and it imports without ``PYTHONPATH`` (installed package)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / rel_path), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "usage:" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_workflows_run_the_cli_entry_points() -> None:
    texts = {p.name: p.read_text(encoding="utf-8") for p in WORKFLOWS.glob("*.yml")}
    assert "python cli/monthly_decision.py $AS_OF_FLAG" in texts["monthly_decision.yml"]
    assert "python cli/edgar_monthly_fetch.py --backfill-years" in texts["monthly_8k_fetch.yml"]
    assert "ci_offline_smoke.py cli/monthly_decision.py" in texts["ci.yml"]
    assert "ci_offline_smoke.py cli/edgar_monthly_fetch.py --dry-run" in texts["ci.yml"]
    for name, text in texts.items():
        for rel in RETIRED:
            assert rel not in text, (name, rel)


def test_tests_follow_the_code() -> None:
    unit = REPO_ROOT / "tests" / "unit"
    assert list((unit / "decision").glob("test_*.py"))
    assert list((unit / "ingestion").glob("test_edgar_monthly_*.py"))
    assert not list((unit / "scripts").glob("test_monthly_decision_*.py"))
    assert not list((unit / "scripts").glob("test_edgar_*.py"))


# ---------------------------------------------------------------------------
# Golden-output comparison (scripts/checks/golden_replay.py)
# ---------------------------------------------------------------------------


def _month(folder: Path, manifest: dict[str, object], report: str = "report\n") -> Path:
    (folder / "plots").mkdir(parents=True)
    (folder / "recommendation.md").write_text(report, encoding="utf-8")
    (folder / "plots" / "calibration_curve.png").write_bytes(b"\x89PNG")
    (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


_MANIFEST = {"git_sha": "a", "run_timestamp_utc": "t1", "script_name": "scripts/x.py", "warnings": []}


def test_golden_comparison_ignores_only_run_fields(tmp_path: Path) -> None:
    golden = _golden()
    before = _month(tmp_path / "before", _MANIFEST)
    after = _month(tmp_path / "after", _MANIFEST | {"git_sha": "b", "run_timestamp_utc": "t2"})
    identical, problems = golden.compare_month_folders(before, after)
    assert problems == []
    assert identical == ["plots/calibration_curve.png", "recommendation.md"]


def test_golden_comparison_flags_output_and_manifest_changes(tmp_path: Path) -> None:
    golden = _golden()
    before = _month(tmp_path / "before", _MANIFEST)
    after = _month(
        tmp_path / "after",
        _MANIFEST | {"script_name": "cli/x.py", "warnings": ["new"]},
        report="report changed\n",
    )
    (after / "extra.csv").write_text("x\n", encoding="utf-8")
    (before / "plots" / "calibration_curve.png").unlink()
    _, problems = golden.compare_month_folders(before, after)
    assert problems == [
        "new after: extra.csv",
        "new after: plots/calibration_curve.png",
        "differs: recommendation.md",
        "differs: run_manifest.json fields ['script_name', 'warnings']",
    ]
    _, allowed = golden.compare_month_folders(
        before, after, golden.RUN_FIELDS | {"script_name", "warnings"}
    )
    assert "differs: run_manifest.json fields ['script_name', 'warnings']" not in allowed


def test_golden_replay_picks_the_commits_entry_point(tmp_path: Path) -> None:
    golden = _golden()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "monthly_decision.py").write_text("", encoding="utf-8")
    assert golden.entry_point(tmp_path) == "scripts/monthly_decision.py"
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "monthly_decision.py").write_text("", encoding="utf-8")
    assert golden.entry_point(tmp_path) == "cli/monthly_decision.py"
    assert golden.entry_point(REPO_ROOT) == "cli/monthly_decision.py"


def test_golden_replay_refuses_the_committed_db() -> None:
    golden = _golden()
    with pytest.raises(SystemExit):
        golden.main(
            ["--base", "HEAD", "--db", str(REPO_ROOT / "data" / "pgr_financials.db"), "--as-of", "2026-09-21"]
        )
