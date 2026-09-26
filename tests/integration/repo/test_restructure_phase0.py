"""Review 2026-09-25, section 5, phase 0: packaging and repository safety nets.

- ``pyproject.toml`` holds the package metadata (``pgr_vds``) and the pytest,
  mypy and ruff config; the old ``pytest.ini``, ``mypy.ini`` and ``ruff.toml``
  are gone, and pandas is pinned to the tested major version.
- ``scripts/checks/check_doc_links.py`` finds no broken link in the active
  docs, and does find a broken one.
- ``scripts/checks/check_sys_path_edits.py`` passes on the repository and
  fails on a new ``sys.path`` edit.
- CI installs the package and runs both checks.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKS = REPO_ROOT / "scripts" / "checks"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, CHECKS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _requirement_names(lines: list[str]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for line in lines:
        entry = line.split("#", 1)[0].strip()
        if not entry or entry.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[ ;]", entry, maxsplit=1)[0]
        specs[name.lower()] = entry.replace(" ", "")
    return specs


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------


def test_pyproject_declares_installable_package() -> None:
    data = _pyproject()
    assert data["project"]["name"] == "pgr_vds"
    assert data["project"]["requires-python"] == ">=3.11"
    include = data["tool"]["setuptools"]["packages"]["find"]["include"]
    assert {"src", "src.*", "config", "config.*"} <= set(include)


def test_pyproject_pins_pandas_to_tested_major() -> None:
    deps = _requirement_names(_pyproject()["project"]["dependencies"])
    assert deps["pandas"] == "pandas>=3.0,<4"


def test_requirements_txt_matches_pyproject_dependencies() -> None:
    """The workflows install requirements.txt; it must equal the package deps."""
    pyproject = _requirement_names(_pyproject()["project"]["dependencies"])
    requirements = _requirement_names(
        (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    )
    assert requirements == pyproject


def test_tool_config_lives_in_pyproject_only() -> None:
    tool = _pyproject()["tool"]
    assert tool["pytest"]["ini_options"]["testpaths"] == ["tests"]
    assert "-q" not in tool["pytest"]["ini_options"]["addopts"].split()
    assert tool["mypy"]["python_version"] == "3.11"
    assert tool["ruff"]["line-length"] == 100
    for old in ("pytest.ini", "mypy.ini", "ruff.toml", "setup.cfg", "tox.ini"):
        assert not (REPO_ROOT / old).exists(), f"{old} would shadow pyproject.toml"


# ---------------------------------------------------------------------------
# Markdown link checker
# ---------------------------------------------------------------------------


def test_active_docs_have_no_broken_links() -> None:
    checker = _load("check_doc_links")
    docs = checker.active_docs()
    rel = {p.relative_to(REPO_ROOT).as_posix() for p in docs}
    assert {"README.md", "docs/workflows.md", "docs/artifact-policy.md"} <= rel
    assert not any(r.startswith(("docs/plans/", "docs/archive/", "results/")) for r in rel)
    broken = [str(item) for path in docs for item in checker.check_file(path)]
    assert broken == []


def test_link_checker_reports_missing_files_and_anchors(tmp_path: Path) -> None:
    checker = _load("check_doc_links")
    (tmp_path / "other.md").write_text("# Real Heading\n\n## Step 2: the fix\n", encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_text(
        "\n".join(
            [
                "# Title",
                "[ok](other.md) [ok anchor](other.md#real-heading) [ok2](other.md#step-2-the-fix)",
                "[self](#title) [web](https://example.com/missing)",
                "[gone](missing.md) [bad anchor](other.md#nope)",
                "`[not a link](missing-in-code.md)`",
                "```",
                "[fenced](missing-in-fence.md)",
                "```",
                "[ref]: ./also-missing.md",
            ]
        ),
        encoding="utf-8",
    )
    broken = checker.check_file(doc, root=tmp_path)
    assert sorted((b.target, b.reason) for b in broken) == [
        ("./also-missing.md", "missing file"),
        ("missing.md", "missing file"),
        ("other.md#nope", "missing anchor"),
    ]


# ---------------------------------------------------------------------------
# sys.path rule
# ---------------------------------------------------------------------------


def test_no_new_sys_path_edits_outside_conftest() -> None:
    checker = _load("check_sys_path_edits")
    new, stale = checker.find_violations(checker.tracked_files(), checker.load_allowlist())
    assert new == [], "new sys.path edits; import via `pip install -e .` instead"
    assert stale == [], "remove these from scripts/checks/sys_path_allowlist.txt"


# Sources are written with "SP" for "sys.path" so this file does not itself
# trip the rule it tests.
SP = "sys" + ".path"


@pytest.mark.parametrize(
    ("source", "edits"),
    [
        (f"{SP}.insert(0, str(ROOT))", True),
        (f"{SP}.append('x')", True),
        (f"{SP} = ['x'] + {SP}", True),
        (f"{SP} += ['x']", True),
        (f"{SP}[0:0] = ['x']", True),
        ("site" + ".addsitedir('x')", True),
        (f"print({SP})", False),
        (f"if 'x' in {SP}: pass", False),
        (f"# {SP}.insert(0, 'x')", False),
        (f"assert {SP} == before", False),
    ],
)
def test_sys_path_edit_detection(source: str, edits: bool) -> None:
    assert _load("check_sys_path_edits").edits_sys_path(source) is edits


def test_sys_path_rule_flags_new_file_and_stale_entry(tmp_path: Path) -> None:
    checker = _load("check_sys_path_edits")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text(f"import sys\n{SP}.insert(0, '.')\n")
    (tmp_path / "old.py").write_text(f"import sys\n{SP}.insert(0, '.')\n")
    (tmp_path / "new.py").write_text(f"import sys\n{SP}.append('.')\n")
    (tmp_path / "fixed.py").write_text("import src\n")
    files = ["tests/conftest.py", "old.py", "new.py", "fixed.py"]
    new, stale = checker.find_violations(files, {"old.py", "fixed.py"}, root=tmp_path)
    assert new == ["new.py"]
    assert stale == ["fixed.py"]


# ---------------------------------------------------------------------------
# CI wiring
# ---------------------------------------------------------------------------


def test_ci_installs_package_and_runs_repository_checks() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r"pip install (--no-deps )?-e \.", ci)
    assert "python scripts/checks/check_doc_links.py" in ci
    assert "python scripts/checks/check_sys_path_edits.py" in ci
