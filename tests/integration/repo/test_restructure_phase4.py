"""Review 2026-09-25, section 5, phase 4: docs and tests layout (WP13, F32).

- ``docs/plans``, ``docs/superpowers``, ``docs/closeouts``, ``docs/results``
  and ``docs/archive`` are merged, sub-folders intact, into ``docs/history/``
  behind one index, and the link checker covers all of it.
- Promotion decisions are one file each in ``docs/decisions/``, with a summary
  table left in ``docs/model-governance.md``.
- ``CLAUDE.md`` points to ``AGENTS.md``; ``pyproject.toml`` is the only
  dependency and tool-config file; root ``archive/`` is retired.
- ``tests/`` is split into ``unit/`` (mirroring the code), ``integration/``
  and ``research/``; no version-numbered test files outside
  ``tests/research/``; research tests run in their own CI job.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORY = REPO_ROOT / "docs" / "history"
DECISIONS = REPO_ROOT / "docs" / "decisions"
TESTS = REPO_ROOT / "tests"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# Tracked files per history sub-folder before the move (master at 282a6b3).
HISTORY_TREES: dict[str, int] = {
    "plans": 28,
    "superpowers": 49,
    "closeouts": 32,
    "results": 24,
    "archive": 44,
}
SRC_PACKAGES = (
    "backtest", "database", "ingestion", "models", "portfolio", "processing",
    "reporting", "tax",
)


def _tracked(*patterns: str) -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *patterns],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return [line for line in output.splitlines() if (REPO_ROOT / line).exists()]


def _load_check(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / "checks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"phase4_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. docs/history/
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tree", sorted(HISTORY_TREES))
def test_history_tree_moved_under_docs_history(tree: str) -> None:
    assert not (REPO_ROOT / "docs" / tree).exists()
    moved = _tracked(f"docs/history/{tree}/")
    assert len(moved) >= HISTORY_TREES[tree]


def test_history_keeps_sub_folders() -> None:
    for sub in (
        "superpowers/plans", "superpowers/specs",
        "archive/history/peer-reviews", "archive/history/repo-peer-reviews",
        "archive/history/v160-ta-research-reports",
    ):
        assert (HISTORY / sub).is_dir(), sub


def test_history_index_links_every_sub_folder() -> None:
    text = (HISTORY / "README.md").read_text(encoding="utf-8")
    for target in (
        "](plans/)", "](superpowers/plans/)", "](superpowers/specs/)", "](closeouts/)",
        "](results/)", "](archive/)", "](retired-code/README.md)",
    ):
        assert target in text, target
    docs_map = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "](history/README.md)" in docs_map
    assert "](decisions/README.md)" in docs_map


def test_link_checker_covers_history_and_decisions_and_passes() -> None:
    checker = _load_check("check_doc_links")
    docs = checker.active_docs()
    rel = {p.relative_to(REPO_ROOT).as_posix() for p in docs}
    history_md = set(_tracked("docs/history/*.md"))
    assert history_md and history_md <= rel
    assert {"docs/history/README.md", "docs/decisions/README.md"} <= rel
    broken = [str(item) for path in docs for item in checker.check_file(path)]
    assert broken == []


def test_no_live_file_points_at_the_old_history_paths() -> None:
    old = re.compile(r"(?<![\w/])docs/(plans|superpowers|closeouts|results|archive)\b")
    offenders = []
    for rel in _tracked("*.md", "*.py", "*.yml", "*.yaml", "*.toml"):
        if rel.startswith(("docs/history/", "docs/reviews/", "research/legacy/", "CHANGELOG.md")):
            continue  # dated records keep their original wording
        if "/outputs/" in rel or rel == "scripts/checks/check_doc_links.py":
            continue
        if rel.startswith("tests/integration/repo/test_restructure_phase"):
            continue
        for number, line in enumerate((REPO_ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
            if old.search(line):
                offenders.append(f"{rel}:{number}: {line.strip()[:90]}")
    assert offenders == []


# ---------------------------------------------------------------------------
# 1b. docs/decisions/
# ---------------------------------------------------------------------------


def _decision_files() -> list[Path]:
    return sorted(p for p in DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md"))


def test_one_decision_file_per_promotion() -> None:
    files = _decision_files()
    assert len(files) >= 7
    numbers = [int(p.name[:4]) for p in files]
    assert numbers == list(range(1, len(files) + 1))
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert text.startswith(f"# {path.name[:4]} — "), path.name
        for heading in ("**Status**", "**Date**", "## Decision", "## Evidence", "## Consequences"):
            assert heading in text, f"{path.name}: {heading}"


def test_governance_keeps_a_summary_table_of_every_decision() -> None:
    text = (REPO_ROOT / "docs" / "model-governance.md").read_text(encoding="utf-8")
    assert "## Decision Record" in text
    assert "## Recent Promotion Record" not in text
    for path in _decision_files():
        assert f"](decisions/{path.name})" in text, path.name
    index = (DECISIONS / "README.md").read_text(encoding="utf-8")
    for path in _decision_files():
        assert f"]({path.name})" in index, path.name


# ---------------------------------------------------------------------------
# 2. Root-level cleanup
# ---------------------------------------------------------------------------


def test_claude_md_points_to_agents_md() -> None:
    tracked = _tracked("CLAUDE.md", "claude.md")
    assert tracked == ["CLAUDE.md"]
    text = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in text
    assert len(text.splitlines()) < 20  # a pointer, not a copy of the directives


def test_pyproject_is_the_only_dependency_and_tool_config() -> None:
    for name in (
        "requirements.txt", "requirements-dev.txt", "requirements-dashboard.txt",
        "constraints-dev.txt", "pytest.ini", "mypy.ini", "ruff.toml",
    ):
        assert not (REPO_ROOT / name).exists(), name
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    dev = {re.split(r"[<>=]", spec, maxsplit=1)[0]: spec for spec in extras["dev"]}
    # The constraints file's exact pins now live in the dev extra.
    assert dev["pytest"] == "pytest==9.0.2"
    assert dev["ruff"] == "ruff==0.13.2"
    assert dev["mypy"] == "mypy==1.18.2"
    assert dev["hypothesis"] == "hypothesis==6.135.7"
    assert dev["pyyaml"] == "pyyaml==6.0.2"
    assert any(spec.startswith("streamlit") for spec in extras["dashboard"])


def test_every_workflow_installs_from_pyproject() -> None:
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert "requirements" not in text and "constraints-dev" not in text, path.name
        if "pip install" in text:
            assert re.search(r'pip install -e \.|pip install -e "\.\[dev\]"', text), path.name
        assert text.count("cache: 'pip'") == text.count("cache-dependency-path: pyproject.toml"), path.name


def test_root_archive_is_retired() -> None:
    assert not (REPO_ROOT / "archive").exists()
    assert _tracked("archive/") == []
    index = (HISTORY / "retired-code" / "README.md").read_text(encoding="utf-8")
    for name in ("v11_autonomous_loop.py", "v24_vti_replacement_study.py", "test_v11_research.py"):
        assert name in index
    allowlist = (REPO_ROOT / "scripts" / "checks" / "sys_path_allowlist.txt").read_text(encoding="utf-8")
    assert "archive/" not in allowlist


# ---------------------------------------------------------------------------
# 3. tests/ layout
# ---------------------------------------------------------------------------


def test_no_test_files_left_at_the_top_of_tests() -> None:
    top = [p.name for p in TESTS.glob("test_*.py")]
    assert top == []
    helpers = {p.name for p in TESTS.glob("*.py")}
    assert helpers == {
        "__init__.py", "conftest.py", "repo_guard.py", "capital_return_fixture.py", "guard_probe.py",
    }


def test_unit_tests_mirror_the_code() -> None:
    unit = TESTS / "unit"
    for package in SRC_PACKAGES:
        assert (REPO_ROOT / "src" / package).is_dir()
        assert list((unit / package).glob("test_*.py")), package
    for extra in ("config", "dashboard", "scripts"):
        assert list((unit / extra).glob("test_*.py")), extra
    for folder in (unit, *[p for p in unit.iterdir() if p.is_dir() and p.name != "__pycache__"]):
        assert (
            folder.name == "unit"
            or (REPO_ROOT / "src" / folder.name).is_dir()
            # Phase 5: packages that moved under src/pgr_vds/ (e.g. decision).
            or (REPO_ROOT / "src" / "pgr_vds" / folder.name).is_dir()
            or folder.name in ("config", "dashboard", "scripts")
        ), folder.name


def test_integration_and_research_folders() -> None:
    for sub in ("pipeline", "data", "repo"):
        assert list((TESTS / "integration" / sub).glob("test_*.py")), sub
    assert len(list((TESTS / "research").glob("test_*.py"))) >= 100


def test_every_test_folder_is_a_package() -> None:
    packages = [p.parent.relative_to(TESTS).as_posix() for p in TESTS.rglob("__init__.py")]
    assert {"unit", "unit/models", "integration", "integration/repo", "research"} <= set(packages)
    missing = [
        str(folder.relative_to(REPO_ROOT))
        for folder in [TESTS, *TESTS.rglob("*")]
        if folder.is_dir()
        and folder.name not in ("__pycache__", "fixtures")
        and list(folder.glob("*.py"))
        and not (folder / "__init__.py").exists()
    ]
    assert missing == []


def test_no_version_numbered_test_files_outside_research() -> None:
    numbered = re.compile(r"^test_v\d+_|_wp\d+\b|^test_wp\d+_|_p2\d")
    offenders = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in TESTS.rglob("*.py")
        if "research" not in p.relative_to(TESTS).parts
        and numbered.search(p.stem)
        and not p.stem.startswith("test_v129_feature_map")  # named after src/models/v129_feature_map.py
    ]
    assert offenders == []


def test_moved_tests_resolve_the_repo_root() -> None:
    wrong = []
    moved = [*(TESTS / "unit").rglob("test_*.py"), *(TESTS / "integration").rglob("test_*.py")]
    assert len(moved) >= 140
    for path in moved:
        text = path.read_text(encoding="utf-8")
        for depth in re.findall(r"Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]", text):
            if depth not in ("2", "3"):  # parents[2] is tests/, for tests/fixtures
                wrong.append(f"{path.name}: parents[{depth}]")
        if re.search(r"Path\(__file__\)(\.resolve\(\))?\.parent\.parent\b", text):
            wrong.append(f"{path.name}: .parent.parent")
    assert wrong == []


def test_layer_markers_are_applied_by_folder() -> None:
    from tests import conftest

    assert conftest.layer_marker(TESTS / "unit" / "models" / "test_x.py") == "unit"
    assert conftest.layer_marker(TESTS / "integration" / "repo" / "test_x.py") == "integration"
    assert conftest.layer_marker(TESTS / "research" / "test_x.py") == "research"
    assert conftest.layer_marker(TESTS / "conftest.py") is None
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = " ".join(data["tool"]["pytest"]["ini_options"]["markers"])
    for name in ("unit:", "integration:", "research:", "artifact:"):
        assert name in markers


def test_this_file_carries_the_integration_marker(request: pytest.FixtureRequest) -> None:
    assert request.node.get_closest_marker("integration") is not None
    assert request.node.get_closest_marker("research") is None


def test_ci_runs_research_tests_in_their_own_job() -> None:
    ci = CI.read_text(encoding="utf-8")
    jobs = re.findall(r"^  ([a-z_-]+):\s*$", ci, flags=re.MULTILINE)
    assert {"test", "research", "artifacts"} <= set(jobs)
    assert 'python -m pytest -q -m "not artifact and not research"' in ci
    assert 'python -m pytest -q -m "research and not artifact"' in ci
    assert "python -m pytest -q -m artifact" in ci


def test_mutation_study_names_existing_test_files() -> None:
    study = _load_check("mutation_study")
    listed = {t for *_, tests in study.MUTATIONS for t in tests}
    assert len(listed) >= 40
    missing = sorted(t for t in listed if not (TESTS / t).is_file())
    assert missing == []


# ---------------------------------------------------------------------------
# 4. Docs describe the new layout
# ---------------------------------------------------------------------------


def test_contributing_and_docs_map_describe_the_new_layout() -> None:
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for needle in (
        "tests/unit/", "tests/integration/", "tests/research/", 'pip install -e ".[dev]"',
        "docs/decisions/", "docs/history/",
    ):
        assert needle in contributing, needle
    assert "requirements-dev.txt" not in contributing
    assert "requirements.txt" not in contributing
