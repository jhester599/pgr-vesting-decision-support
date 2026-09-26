"""Import smoke test for every production entry point (review 2026-09-25, section 5).

Phase 0 safety net before any file move: every script and module that a GitHub
workflow runs must import cleanly in a fresh interpreter started outside the
repository, so a move that breaks an import fails here rather than in a
scheduled run. The ``src`` and ``config`` modules must import without a
``sys.path`` edit, which needs the package installed (``pip install -e .``).

Phase 2 (F30): importing the production code must not load anything from
``results/`` and must not change the warning filters, ``sys.path`` or the
stdout encoding.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Every script a workflow runs (directly or through ci_offline_smoke.py).
PRODUCTION_SCRIPTS: tuple[str, ...] = (
    "scripts/bootstrap.py",
    "scripts/capital_return_charts.py",
    "scripts/check_data_integrity.py",
    "scripts/checks/check_doc_links.py",
    "scripts/checks/check_sys_path_edits.py",
    "scripts/ci_offline_smoke.py",
    "scripts/edgar_8k_fetcher.py",
    "scripts/finalize_db.py",
    "scripts/initial_fetch.py",
    "scripts/monthly_decision.py",
    "scripts/peer_fetch.py",
    "scripts/repurchase_timeseries_charts.py",
    "scripts/verify_monthly_outputs.py",
    "scripts/weekly_fetch.py",
)
# Every module a workflow's inline Python imports, plus the config package.
PRODUCTION_MODULES: tuple[str, ...] = (
    "config",
    "src.database.db_client",
    "src.models.drift_monitor",
    "src.models.retrain_trigger",
    "src.reporting.email_sender",
)

_WORKFLOW_SCRIPT = re.compile(r"\bpython3?\s+(?:-u\s+)?(scripts/[\w/]+\.py)")
_WORKFLOW_MODULE = re.compile(
    r"^\s*from\s+((?:src|config)(?:\.\w+)*)\s+import\s+(\w+)", re.MULTILINE
)


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.setdefault("EDGAR_USER_AGENT", "pytest suite pytest@example.invalid")
    return env


def _run(code: str, cwd: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = _clean_env()
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _workflow_entry_points() -> tuple[set[str], set[str]]:
    scripts: set[str] = set()
    modules: set[str] = set()
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        scripts.update(_WORKFLOW_SCRIPT.findall(text))
        for package, name in _WORKFLOW_MODULE.findall(text):
            # `from src.database import db_client` imports the module db_client.
            as_module = REPO_ROOT / package.replace(".", "/") / f"{name}.py"
            modules.add(f"{package}.{name}" if as_module.is_file() else package)
    return scripts, modules


def test_smoke_list_covers_every_workflow_entry_point() -> None:
    scripts, modules = _workflow_entry_points()
    assert scripts, "no workflow scripts found; the pattern is stale"
    assert sorted(scripts - set(PRODUCTION_SCRIPTS)) == []
    assert sorted(modules - set(PRODUCTION_MODULES)) == []
    for rel in PRODUCTION_SCRIPTS:
        assert (REPO_ROOT / rel).is_file(), rel


@pytest.mark.parametrize("rel_path", PRODUCTION_SCRIPTS)
def test_production_script_imports(rel_path: str, tmp_path: Path) -> None:
    """Load the script as a module (its ``__main__`` block does not run)."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('entry_smoke', {str(REPO_ROOT / rel_path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['entry_smoke'] = module\n"
        "spec.loader.exec_module(module)\n"
    )
    result = _run(code, cwd=tmp_path)
    assert result.returncode == 0, result.stderr[-4000:]
    assert list(tmp_path.iterdir()) == [], "importing the script wrote to the working directory"


@pytest.mark.parametrize("module", PRODUCTION_MODULES)
def test_production_module_imports_without_sys_path_edit(module: str, tmp_path: Path) -> None:
    """``src``/``config`` resolve from the installed package, not from the cwd."""
    code = (
        "import importlib, pathlib\n"
        f"mod = importlib.import_module({module!r})\n"
        "print(pathlib.Path(mod.__file__).resolve())\n"
    )
    result = _run(code, cwd=tmp_path)
    assert result.returncode == 0, (
        f"{module} does not import from outside the repo; run `pip install -e .`\n"
        + result.stderr[-4000:]
    )
    assert Path(result.stdout.strip()).is_relative_to(REPO_ROOT)


# Third-party imports (numpy, scipy) add narrow filters of their own; the
# probe reports the blanket "ignore" filters that results/research/
# v46_classification.py installed at import (ConvergenceWarning, sklearn
# FutureWarning, "All-NaN slice").
_SIDE_EFFECT_PROBE = """
import json, sys, warnings
filters = list(warnings.filters)
path = list(sys.path)
{imports}
added = [f for f in warnings.filters if f not in filters]
blanket = [
    repr(f) for f in added
    if f[0] == "ignore" and (
        f[2].__name__ in ("ConvergenceWarning", "FutureWarning")
        or (f[1] is not None and "All-NaN" in f[1].pattern)
    )
]
print(json.dumps({{
    "results_modules": sorted(m for m in sys.modules if m == "results" or m.startswith("results.")),
    "blanket_filters": blanket,
    "sys_path_changed": list(sys.path) != path,
    "stdout_encoding": sys.stdout.encoding,
}}))
"""


def test_production_model_imports_have_no_side_effects(tmp_path: Path) -> None:
    """F30: the classification shadow chain used to import results/research/v46_classification.py."""
    code = _SIDE_EFFECT_PROBE.format(
        imports="import src.models.classification_shadow\nimport src.research.v66_utils"
    )
    result = _run(code, cwd=tmp_path, extra_env={"PYTHONIOENCODING": "ascii"})
    assert result.returncode == 0, result.stderr[-4000:]
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert probe["results_modules"] == []
    assert probe["blanket_filters"] == []
    assert probe["sys_path_changed"] is False
    assert probe["stdout_encoding"] == "ascii"


def test_monthly_decision_import_loads_nothing_from_results(tmp_path: Path) -> None:
    code = _SIDE_EFFECT_PROBE.format(
        imports=(
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('entry_smoke', "
            f"{str(REPO_ROOT / 'scripts' / 'monthly_decision.py')!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "sys.modules['entry_smoke'] = module\n"
            "spec.loader.exec_module(module)"
        )
    )
    result = _run(code, cwd=tmp_path, extra_env={"PYTHONIOENCODING": "ascii"})
    assert result.returncode == 0, result.stderr[-4000:]
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert probe["results_modules"] == []
    assert probe["blanket_filters"] == []
    # Reconfiguring stdout is the entry point's job, in its __main__ block.
    assert probe["stdout_encoding"] == "ascii"
