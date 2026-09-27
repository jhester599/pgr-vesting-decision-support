"""R3 (pre-v200 remediation): CPCV is retired and no active code runs K-fold.

The combinatorial purged cross-validation diagnostic trained on folds after
its test folds, i.e. it was a combinatorial K-fold, which AGENTS.md
prohibits (review 2026-09-25, F02). These tests replace the old CPCV unit
tests (``test_cpcv.py``): the compatibility entry point refuses to run, and
no production module (``src`` outside ``src/research``, ``cli``, ``config``,
``scripts``) imports a K-fold style splitter, calls a K-fold scorer, shuffles
a split, or tunes a ``*CV`` estimator with anything but an explicit
chronological splitter.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import wfo_engine

REPO_ROOT = Path(__file__).resolve().parents[3]
ACTIVE_ROOTS = ("src", "cli", "config", "scripts")
EXCLUDED = ("src/research",)
FORBIDDEN_NAMES = {
    "CombinatorialPurgedCV",
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "LeaveOneOut",
    "LeavePOut",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "cross_val_score",
    "cross_val_predict",
    "cross_validate",
}
TUNED_ESTIMATORS = {"RidgeCV", "LassoCV", "ElasticNetCV", "LogisticRegressionCV"}


def _active_python_files() -> list[Path]:
    files: list[Path] = []
    for root in ACTIVE_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if any(rel.startswith(prefix + "/") for prefix in EXCLUDED):
                continue
            files.append(path)
    return files


def _violations(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.extend(f"imports {a.name}" for a in node.names if a.name in FORBIDDEN_NAMES)
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            found.append(f"uses {node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.append(f"uses {node.id}")
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if (
                    keyword.arg == "shuffle"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    found.append("shuffle=True")
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in TUNED_ESTIMATORS:
                cv = next((k.value for k in node.keywords if k.arg == "cv"), None)
                # No cv (leave-one-out / K-fold default), cv=None or an int
                # (K-fold) are all prohibited; a splitter object is required.
                if cv is None or isinstance(cv, ast.Constant):
                    found.append(f"{name} without an explicit chronological splitter")
    return found


def test_no_active_module_uses_k_fold_validation() -> None:
    files = _active_python_files()
    assert len(files) > 50
    problems = {
        file.relative_to(REPO_ROOT).as_posix(): v
        for file in files
        if (v := _violations(ast.parse(file.read_text(encoding="utf-8"))))
    }
    assert problems == {}


@pytest.mark.parametrize(
    "source",
    [
        "from skfolio.model_selection import CombinatorialPurgedCV",
        "from sklearn.model_selection import KFold",
        "import sklearn.model_selection as ms\nms.LeaveOneOut()",
        "TimeSeriesSplit(n_splits=3, shuffle=True)",
        "RidgeCV(alphas=[1.0])",
        "RidgeCV(alphas=[1.0], cv=5)",
        "cross_val_score(model, X, y)",
    ],
)
def test_the_scan_catches_each_forbidden_pattern(source: str) -> None:
    """Counterfactual for the scan above: each pattern is reported."""
    assert _violations(ast.parse(source))


def test_the_scan_accepts_the_production_inner_splitter() -> None:
    source = "RidgeCV(alphas=alphas, cv=_make_inner_cv(target_horizon_months=6))"
    assert _violations(ast.parse(source)) == []


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"model_type": "ridge"}, {"n_folds": 8, "n_test_folds": 2, "embargo_size": 2}],
)
def test_run_cpcv_is_an_unsupported_method(kwargs: dict[str, object]) -> None:
    idx = pd.date_range("2010-01-31", periods=96, freq="ME")
    X = pd.DataFrame({"a": np.linspace(0.0, 1.0, 96)}, index=idx)
    y = pd.Series(np.linspace(0.0, 1.0, 96), index=idx, name="y")
    with pytest.raises(wfo_engine.UnsupportedValidationMethodError, match="K-fold"):
        wfo_engine.run_cpcv(X, y, **kwargs)
    assert issubclass(wfo_engine.UnsupportedValidationMethodError, ValueError)


def test_cpcv_result_types_are_gone() -> None:
    """Nothing can build or read a CPCV result in active code any more."""
    for name in ("CPCVResult", "cpcv_path_thresholds", "_recombined_path_members"):
        assert not hasattr(wfo_engine, name), name
