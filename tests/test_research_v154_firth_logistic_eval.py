"""Tests for the v154 Firth logistic utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_hat_diag_sums_to_rank() -> None:
    from src.research.v154_utils import _hat_diag
    rng = np.random.default_rng(0)
    n, p = 30, 4
    X = rng.normal(size=(n, p))
    W_diag = np.ones(n) * 0.25
    XW_half = X * np.sqrt(W_diag)[:, None]
    h = _hat_diag(XW_half)
    assert h.shape == (n,)
    assert abs(h.sum() - p) < 1e-6
    assert np.all(h >= -1e-9)


def test_firth_logistic_separable_data_converges() -> None:
    """Firth should not diverge on perfectly separable data (unlike MLE)."""
    from src.research.v154_utils import fit_firth_logistic, predict_firth_proba
    X = np.array([[1.0], [2.0], [3.0], [-1.0], [-2.0], [-3.0]])
    X_aug = np.column_stack([np.ones(6), X])
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    beta = fit_firth_logistic(X_aug, y, max_iter=50)
    assert beta.shape == (2,)
    assert np.all(np.isfinite(beta))
    assert abs(beta[1]) < 20.0
    proba = predict_firth_proba(X_aug, beta)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_firth_logistic_recovers_direction() -> None:
    """Firth should assign higher probability to positive-class observations."""
    from src.research.v154_utils import fit_firth_logistic, predict_firth_proba
    rng = np.random.default_rng(42)
    n = 50
    X = rng.normal(size=(n, 3))
    beta_true = np.array([0.0, 1.5, -1.0, 0.5])
    X_aug = np.column_stack([np.ones(n), X])
    logit = X_aug @ beta_true
    prob_true = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, prob_true).astype(float)
    beta_hat = fit_firth_logistic(X_aug, y)
    proba = predict_firth_proba(X_aug, beta_hat)
    assert float(np.corrcoef(proba, prob_true)[0, 1]) > 0.50


def test_count_training_positives() -> None:
    from src.research.v154_utils import count_training_positives
    y = np.array([0, 1, 0, 1, 1, 0])
    assert count_training_positives(y) == 3


def test_thin_benchmarks_threshold() -> None:
    from src.research.v154_utils import FIRTH_THIN_THRESHOLD
    assert FIRTH_THIN_THRESHOLD == 30


@pytest.mark.slow
def test_evaluate_classifier_wfo_returns_expected_keys() -> None:
    from src.research.v154_utils import (
        evaluate_classifier_wfo,
        load_research_inputs_for_classification,
    )
    feature_df, rel_map = load_research_inputs_for_classification()
    benchmark = "VOO"
    from src.research.v37_utils import RIDGE_FEATURES_12
    result = evaluate_classifier_wfo(
        feature_df=feature_df,
        rel_series=rel_map[benchmark],
        feature_cols=list(RIDGE_FEATURES_12),
        benchmark=benchmark,
        use_firth=False,
    )
    assert "ba_covered" in result
    assert "coverage" in result
    assert "avg_train_positives" in result


@pytest.mark.slow
def test_run_firth_evaluation_writes_candidate(tmp_path: Path) -> None:
    from results.research.v154_firth_logistic_eval import run_firth_evaluation
    candidate_path = tmp_path / "v154_firth_candidate.json"
    result = run_firth_evaluation(
        benchmarks=["DBC", "VDE"],
        candidate_path=candidate_path,
    )
    assert candidate_path.exists()
    assert "firth_winners" in result
    assert "rows" in result
    assert len(result["rows"]) == 2


def test_candidate_file_schema() -> None:
    import json
    try:
        from results.research.v154_firth_logistic_eval import DEFAULT_CANDIDATE_PATH
    except ModuleNotFoundError:
        pytest.skip("harness module not yet created — run Task 3 first")
    if not DEFAULT_CANDIDATE_PATH.exists():
        pytest.skip("candidate file not yet generated — run harness first")
    data = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert "firth_winners" in data
    assert isinstance(data["firth_winners"], list)
    assert "recommendation" in data
    assert data["recommendation"] in ("adopt_firth_for_thin_benchmarks", "no_benefit")
