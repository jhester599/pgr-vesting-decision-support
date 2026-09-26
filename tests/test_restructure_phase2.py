"""Review 2026-09-25, section 5, phase 2: de-version the library.

- The nine ``from X import *`` shims in ``src/research`` are gone
  (``v11`` had no importers; the other 8 had 39 importing files, which now
  import the real modules).
- ``results/research/v46_classification.py``, which production imported
  through ``src.research.v66_utils`` (F30), moved to
  ``src/research/binary_classification.py`` without its import-time side
  effects (``sys.path`` edit, ``sys.stdout.reconfigure``, global warning
  filters).
- No library or production module imports from ``results/``.
"""

from __future__ import annotations

import ast
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SHIMS: dict[str, str] = {
    "benchmark_sets": "src.portfolio.benchmark_sets",
    "diversification": "src.portfolio.diversification",
    "evaluation": "src.models.evaluation",
    "policy_metrics": "src.models.policy_metrics",
    "v11": "src.portfolio.redeploy_buckets",
    "v12": "src.reporting.snapshot_summary",
    "v22": "src.reporting.cross_check",
    "v27": "src.portfolio.redeploy_portfolio",
    "v29": "src.reporting.confidence",
}
_SHIM_REFERENCE = re.compile(r"\bsrc\.research\.(" + "|".join(SHIMS) + r")\b")


def _tracked_python() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py", "*.yml"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if (REPO_ROOT / line).is_file()]


@pytest.mark.parametrize("name", sorted(SHIMS))
def test_research_shim_deleted(name: str) -> None:
    assert not (REPO_ROOT / "src" / "research" / f"{name}.py").exists()


def test_nothing_references_the_deleted_shims() -> None:
    offenders = []
    for rel in _tracked_python():
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if _SHIM_REFERENCE.search(line):
                offenders.append(f"{rel}:{number}: {line.strip()}")
    assert offenders == []


def _module_level_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


def test_library_and_production_code_never_import_results() -> None:
    roots = ["src", "config", "dashboard"]
    offenders = []
    for rel in _tracked_python():
        in_scope = rel.split("/", 1)[0] in roots or (
            rel.startswith("scripts/") and not rel.startswith("scripts/research/")
        )
        if not in_scope or not rel.endswith(".py"):
            continue
        for module in _module_level_imports(REPO_ROOT / rel):
            if module == "results" or module.startswith("results."):
                offenders.append(f"{rel}: {module}")
    assert offenders == []


# ---------------------------------------------------------------------------
# v46_classification -> src/research/binary_classification.py
# ---------------------------------------------------------------------------

NEW_MODULE = REPO_ROOT / "src" / "research" / "binary_classification.py"


def test_v46_classification_moved_into_src() -> None:
    assert not (REPO_ROOT / "results" / "research" / "v46_classification.py").exists()
    assert NEW_MODULE.is_file()
    assert "src.research.binary_classification" in _module_level_imports(
        REPO_ROOT / "src" / "research" / "v66_utils.py"
    )


def _module_level_side_effects(path: Path) -> list[str]:
    """Module-level calls to sys.path, sys.stdout or warnings (not inside a def)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                text = ast.unparse(sub.func)
                if text.startswith(("sys.path", "sys.stdout", "warnings.", "os.chdir")):
                    found.append(text)
    return found


def test_binary_classification_has_no_import_time_side_effects() -> None:
    assert _module_level_side_effects(NEW_MODULE) == []


def test_compute_binary_metrics_values() -> None:
    """Hand-computed: y = [1, 0, 1, 1], p = [0.8, 0.4, 0.3, 0.9], threshold 0.5."""
    from src.research.binary_classification import compute_binary_metrics

    metrics = compute_binary_metrics(np.array([1, 0, 1, 1]), np.array([0.8, 0.4, 0.3, 0.9]))
    expected_log_loss = -(math.log(0.8) + math.log(0.6) + math.log(0.3) + math.log(0.9)) / 4
    assert metrics == pytest.approx(
        {
            "n": 4.0,
            "accuracy": 0.75,
            "balanced_accuracy": (2 / 3 + 1.0) / 2,
            "brier_score": (0.04 + 0.16 + 0.49 + 0.01) / 4,
            "log_loss": expected_log_loss,
            "precision": 1.0,
            "recall": 2 / 3,
            "base_rate": 0.75,
            "predicted_positive_rate": 0.5,
        }
    )


def test_classification_wfo_uses_time_series_split_only() -> None:
    """AGENTS.md: walk-forward folds with a gap; no K-fold, no shuffling."""
    source = NEW_MODULE.read_text(encoding="utf-8")
    assert "TimeSeriesSplit(" in source and "gap=GAP_MONTHS" in source
    assert not re.search(r"\bKFold\b|shuffle\s*=\s*True|StandardScaler", source)
