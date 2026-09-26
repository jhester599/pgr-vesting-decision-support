# v153–v158 Classification and Feature Research Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the five experiments recommended by the 2026-04-17 peer review — Firth logistic stabilization (CLS-02), WTI momentum for commodity ETFs (FEAT-02), USD index momentum (FEAT-01), term premium differencing (FEAT-03), and a final synthesis shadow lane update — as research-only increments in the existing WFO harness framework.

**Architecture:** Each version (v153–v158) adds one bounded research harness that reuses the `v87_utils` / `v139_utils` evaluation infrastructure. Classification experiments run per-benchmark logistic WFO via `evaluate_classifier_wfo` in the new `v154_utils.py` shared helper. Feature experiments augment the lean 12-feature baseline for targeted ETFs only. No live config changes; all outcomes land in `results/research/` as candidate files and summary markdown.

**Tech Stack:** Python 3.10+, numpy/scipy (Firth IRLS), scikit-learn (LogisticRegression, TimeSeriesSplit, balanced_accuracy_score), pandas, pytest. All harnesses follow the `v141_blend_eval.py` / `test_research_v141_blend_eval.py` file patterns.

---

## Orientation: What Already Exists

Before writing any code, verify these constants by reading the source files listed:

| Symbol | File | Value |
|--------|------|-------|
| `RIDGE_FEATURES_12` | `src/research/v37_utils.py:48-60` | 12-feature lean baseline list |
| `BENCHMARKS` | `src/research/v37_utils.py:36` | `["VOO","VXUS","VWO","VMBS","BND","GLD","DBC","VDE"]` |
| `MAX_TRAIN_MONTHS`, `TEST_SIZE_MONTHS`, `GAP_MONTHS` | `src/research/v37_utils.py:40-43` | 60, 6, 8 |
| `logistic_factory` | `src/research/v87_utils.py:284` | returns `LogisticRegression` factory callable |
| `build_target_series` | `src/research/v87_utils.py:192` | builds `actionable_sell_3pct` target |
| `_outer_time_series_splitter` | `src/research/v87_utils.py:259` | creates `TimeSeriesSplit` |
| `_impute_fold` | `src/research/v87_utils.py:270` | median imputation per fold |
| `evaluate_ensemble_configuration` | `src/research/v139_utils.py:123` | regression ensemble evaluator |
| `usd_broad_return_3m`, `usd_momentum_6m` | `src/research/v87_utils.py:128-129` | already in `macro_rates_spreads` family |
| `wti_return_3m` | `src/research/v87_utils.py:130` | already in `macro_rates_spreads` family |
| `term_premium_10y` | `src/research/v87_utils.py:125` | already in `macro_rates_spreads` family |

---

## File Map

### New files (created in this cycle)

| File | Purpose |
|------|---------|
| `docs/archive/history/repo-peer-reviews/2026-04-17/chatgpt_repo_peerreview_20260417.md` | Archived peer review report ✓ |
| `docs/archive/history/repo-peer-reviews/2026-04-17/README.md` | Archive index ✓ |
| `src/research/v154_utils.py` | Firth logistic + shared classifier WFO helper for v154–v157 |
| `results/research/v154_firth_logistic_eval.py` | CLS-02 harness: Firth vs standard logistic for thin benchmarks |
| `results/research/v154_firth_candidate.json` | CLS-02 result: per-benchmark BA_covered delta |
| `results/research/v155_wti_momentum_eval.py` | FEAT-02 harness: WTI 3M momentum for DBC/VDE |
| `results/research/v155_wti_candidate.json` | FEAT-02 result |
| `results/research/v156_usd_momentum_eval.py` | FEAT-01 harness: USD index momentum for BND/VXUS/VWO |
| `results/research/v156_usd_candidate.json` | FEAT-01 result |
| `results/research/v157_term_premium_eval.py` | FEAT-03 harness: term premium 3M diff |
| `results/research/v157_term_premium_candidate.json` | FEAT-03 result |
| `tests/test_research_v154_firth_logistic_eval.py` | Tests for CLS-02 |
| `tests/test_research_v155_wti_momentum_eval.py` | Tests for FEAT-02 |
| `tests/test_research_v156_usd_momentum_eval.py` | Tests for FEAT-01 |
| `tests/test_research_v157_term_premium_eval.py` | Tests for FEAT-03 |

### Modified files

| File | Change |
|------|--------|
| `docs/research/backlog.md` | Add FEAT-03 entry; update CLS-02/FEAT-01/FEAT-02 status to in-progress |
| `ROADMAP.md` | Add v153–v158 cycle summary block; link to this plan |
| `CHANGELOG.md` | One entry per version after each commit |

---

## Task 1: v153 — Archive and Backlog Update

**Files:**
- Create: `docs/archive/history/repo-peer-reviews/2026-04-17/chatgpt_repo_peerreview_20260417.md` ✓
- Create: `docs/archive/history/repo-peer-reviews/2026-04-17/README.md` ✓
- Modify: `docs/research/backlog.md`
- Modify: `ROADMAP.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Confirm archive files exist**

```bash
ls docs/archive/history/repo-peer-reviews/2026-04-17/
```

Expected output:
```
README.md  chatgpt_repo_peerreview_20260417.md
```

- [ ] **Step 2: Add FEAT-03 to backlog and mark CLS-02/FEAT-01/FEAT-02 as in-progress**

In `docs/research/backlog.md`, locate the `## Ranked Next Queue` section and update item status. Then append the new `FEAT-03` entry after `FEAT-02`. Add these exact blocks:

Update the ranked queue at the top to:
```markdown
## Ranked Next Queue

1. `CLS-02` - Firth logistic for short-history benchmarks — **in-progress (v154)**
2. `FEAT-02` - WTI 3M momentum for DBC/VDE — **in-progress (v155)**
3. `FEAT-01` - DTWEXBGS post-v128 feature search — **in-progress (v156)**
4. `FEAT-03` - Term premium 3M differential signal — **in-progress (v157)**
5. `BL-01` - Black-Litterman tau/view tuning — open (deferred per 2026-04-17 peer review)
```

Add this new backlog entry at the end of `docs/research/backlog.md`:

```markdown
### FEAT-03 — Term Premium 3M Differential Signal
**Status:** open
**Priority:** medium
**Rationale:** The 10Y term premium is already in the feature matrix as `term_premium_10y`, but a 3-month change (`term_premium_diff_3m`) has not been tested. Sudden jumps in term premium historically signal equity headwinds; the differenced series may improve classifier timing without requiring new data.
**Estimated effort:** S
**Depends on:** none
**Expected metric impact:** BA_covered up slightly for rate-sensitive benchmarks (BND, VOO); ECE stable
**Last touched:** 2026-04-17 peer review
```

- [ ] **Step 3: Update ROADMAP.md**

Insert the following block immediately after the `## Active Research Direction: v139-v152` section (before `## Prior Research Direction: v123-v129`):

```markdown
## Active Research Direction: v153-v158

The active plan is documented in:

- [`docs/superpowers/plans/2026-04-17-v153-v158-classification-feature-research.md`](docs/superpowers/plans/2026-04-17-v153-v158-classification-feature-research.md)

Source peer review: [`docs/archive/history/repo-peer-reviews/2026-04-17/chatgpt_repo_peerreview_20260417.md`](docs/archive/history/repo-peer-reviews/2026-04-17/chatgpt_repo_peerreview_20260417.md)

Summary of the v153-v158 classification and feature research arc:

| Version | Theme | Type |
|---|---|---|
| v153 | Archive 2026-04-17 peer review; update backlog with FEAT-03; reorder priorities | Documentation |
| v154 | CLS-02: Firth-penalized logistic for short-history benchmarks | Classifier research |
| v155 | FEAT-02: WTI 3M momentum for DBC/VDE classification | Feature research |
| v156 | FEAT-01: USD index momentum (DTWEXBGS) for BND/VXUS/VWO | Feature research |
| v157 | FEAT-03: Term premium 3M differential signal | Feature research |
| v158 | Synthesis: compare all four experiments; update shadow lane if any winner qualifies | Research + shadow |

Working rule: research-only. No automatic promotion into the live monthly decision path. Mandatory closeout after each completed block.

Priority shift from v152 closeout: the 2026-04-17 peer review re-orders the queue to classification-first (CLS-02 before BL-01) on the basis that predictive signal improvements take precedence over decision-layer policy tuning. BL-01 remains open and should follow after v158 synthesis.
```

- [ ] **Step 4: Add v153 to CHANGELOG.md**

Prepend to `CHANGELOG.md` (at the very top, before the existing first entry):

```markdown
## v153 (2026-04-17)

- Archived 2026-04-17 ChatGPT repo peer review under
  `docs/archive/history/repo-peer-reviews/2026-04-17/`
- Added FEAT-03 (term premium 3M diff) to backlog
- Updated backlog queue: CLS-02 → FEAT-02 → FEAT-01 → FEAT-03 → BL-01
- Noted priority shift: classification-first per 2026-04-17 peer review
  (BL-01 deferred until after v158 synthesis)
- Created plan: `docs/superpowers/plans/2026-04-17-v153-v158-classification-feature-research.md`
```

- [ ] **Step 5: Commit v153**

```bash
git add docs/archive/history/repo-peer-reviews/2026-04-17/ \
        docs/research/backlog.md \
        docs/superpowers/plans/2026-04-17-v153-v158-classification-feature-research.md \
        ROADMAP.md \
        CHANGELOG.md
git commit -m "docs: v153 archive 2026-04-17 peer review and update backlog priorities"
```

---

## Task 2: v154 — CLS-02 Firth Logistic Shared Utilities

**Files:**
- Create: `src/research/v154_utils.py`
- Test: `tests/test_research_v154_firth_logistic_eval.py` (partial — pure-unit tests that don't touch DB)

- [ ] **Step 1: Write the failing unit tests for Firth core functions**

Create `tests/test_research_v154_firth_logistic_eval.py`:

```python
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
    # sum of hat diagonal equals rank of design matrix
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
    # Coefficients should be bounded (Firth prevents infinity)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_research_v154_firth_logistic_eval.py -q --tb=short
```

Expected: `ModuleNotFoundError: No module named 'src.research.v154_utils'` (or similar import failure)

- [ ] **Step 3: Create `src/research/v154_utils.py`**

```python
"""Shared helpers for the v153-v158 classification and feature research cycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from src.research.v37_utils import (
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RESULTS_DIR,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    get_connection,
    load_feature_matrix,
    load_relative_series,
)
from src.research.v87_utils import (
    BENCHMARKS,
    _impute_fold,
    _outer_time_series_splitter,
    build_target_series,
    logistic_factory,
)

FIRTH_THIN_THRESHOLD: int = 30
HIGH_THRESHOLD: float = 0.70
LOW_THRESHOLD: float = 0.30
ACTIONABLE_TARGET: str = "actionable_sell_3pct"


def _hat_diag(XW_half: np.ndarray) -> np.ndarray:
    """Diagonal of the hat matrix H = sqrt(W) X (X'WX)^{-1} X' sqrt(W).

    Uses thin SVD for numerical stability on near-rank-deficient designs.
    """
    try:
        _, s, Vt = np.linalg.svd(XW_half, full_matrices=False)
        threshold = max(1e-12, 1e-10 * float(s[0])) if len(s) > 0 else 1e-12
        s_inv = np.where(s > threshold, 1.0 / s, 0.0)
        U = XW_half @ (Vt.T * s_inv)
        return np.sum(U ** 2, axis=1)
    except np.linalg.LinAlgError:
        return np.zeros(XW_half.shape[0])


def fit_firth_logistic(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> np.ndarray:
    """Fit Firth-penalized logistic regression via IRLS with Jeffreys-prior correction.

    X must already include a leading intercept column.
    Returns coefficient vector beta of shape (X.shape[1],).
    """
    _, p = X.shape
    beta = np.zeros(p)

    for _ in range(max_iter):
        mu = expit(X @ beta)
        W_diag = mu * (1.0 - mu)

        XtWX = X.T @ (X * W_diag[:, None])
        try:
            XtWX_inv = np.linalg.solve(XtWX + 1e-8 * np.eye(p), np.eye(p))
        except np.linalg.LinAlgError:
            break

        XW_half = X * np.sqrt(np.maximum(W_diag, 0.0))[:, None]
        h = _hat_diag(XW_half)

        # Firth-adjusted score: U* = X'(y - mu + h*(0.5 - mu))
        score = X.T @ (y - mu + h * (0.5 - mu))
        delta = XtWX_inv @ score
        beta = beta + delta
        if float(np.linalg.norm(delta)) < tol:
            break

    return beta


def predict_firth_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return class-1 probabilities from fitted Firth coefficients."""
    return expit(X @ beta)


def count_training_positives(y: np.ndarray) -> int:
    """Return number of positive-class examples."""
    return int(np.sum(y == 1))


def load_research_inputs_for_classification() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Load pre-holdout feature matrix and benchmark relative-return series."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rel_map = {
            bm: load_relative_series(conn, bm, horizon=6) for bm in BENCHMARKS
        }
    finally:
        conn.close()
    return feature_df, rel_map


def evaluate_classifier_wfo(
    feature_df: pd.DataFrame,
    rel_series: pd.Series,
    feature_cols: list[str],
    benchmark: str,
    use_firth: bool = False,
) -> dict[str, Any]:
    """Run WFO binary-classifier evaluation for one benchmark.

    Returns a dict with keys: benchmark, n_obs, avg_train_positives,
    ba_all, ba_covered, coverage, use_firth, skipped.
    """
    avail_cols = [c for c in feature_cols if c in feature_df.columns]
    if not avail_cols:
        return {"benchmark": benchmark, "n_obs": 0, "skipped": True, "use_firth": use_firth}

    target = build_target_series(rel_series, ACTIONABLE_TARGET)
    aligned = feature_df[avail_cols].join(target).dropna()
    min_obs = MAX_TRAIN_MONTHS + GAP_MONTHS + TEST_SIZE_MONTHS + 1
    if len(aligned) < min_obs:
        return {"benchmark": benchmark, "n_obs": len(aligned), "skipped": True, "use_firth": use_firth}

    X_all = aligned[avail_cols].to_numpy(dtype=float)
    y_all = aligned[ACTIONABLE_TARGET].to_numpy(dtype=float)
    splitter = _outer_time_series_splitter(len(X_all))

    y_pred_all: list[float] = []
    y_true_all: list[float] = []
    n_positives_per_fold: list[int] = []

    for train_idx, test_idx in splitter.split(X_all):
        X_train, X_test = _impute_fold(X_all[train_idx], X_all[test_idx])
        y_train = y_all[train_idx]
        n_positives_per_fold.append(count_training_positives(y_train))

        if use_firth:
            ones_tr = np.ones((len(X_train), 1))
            ones_te = np.ones((len(X_test), 1))
            X_tr_aug = np.column_stack([ones_tr, X_train])
            X_te_aug = np.column_stack([ones_te, X_test])
            beta = fit_firth_logistic(X_tr_aug, y_train)
            proba = predict_firth_proba(X_te_aug, beta)
        else:
            model = logistic_factory(class_weight="balanced")()
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]

        y_pred_all.extend(proba.tolist())
        y_true_all.extend(y_all[test_idx].tolist())

    y_pred_arr = np.asarray(y_pred_all)
    y_true_arr = np.asarray(y_true_all)

    covered_mask = (y_pred_arr >= HIGH_THRESHOLD) | (y_pred_arr <= LOW_THRESHOLD)
    ba_all = float(balanced_accuracy_score(
        y_true_arr, (y_pred_arr >= 0.5).astype(int)
    ))
    coverage = float(covered_mask.mean())

    if covered_mask.sum() >= 2:
        ba_covered = float(balanced_accuracy_score(
            y_true_arr[covered_mask],
            (y_pred_arr[covered_mask] >= 0.5).astype(int),
        ))
    else:
        ba_covered = float("nan")

    return {
        "benchmark": benchmark,
        "n_obs": len(y_true_arr),
        "avg_train_positives": float(np.mean(n_positives_per_fold)),
        "ba_all": ba_all,
        "ba_covered": ba_covered,
        "coverage": coverage,
        "use_firth": use_firth,
        "skipped": False,
    }


def compare_logistic_vs_firth(
    feature_df: pd.DataFrame,
    rel_map: dict[str, pd.Series],
    benchmarks: list[str],
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    """Evaluate both standard logistic and Firth logistic for each benchmark.

    Returns list of row dicts with delta_ba_covered = firth_ba_covered - logistic_ba_covered.
    """
    rows: list[dict[str, Any]] = []
    for bm in benchmarks:
        if bm not in rel_map or rel_map[bm].empty:
            continue
        std_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=feature_cols,
            benchmark=bm,
            use_firth=False,
        )
        firth_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=feature_cols,
            benchmark=bm,
            use_firth=True,
        )
        delta = float("nan")
        if not std_result.get("skipped") and not firth_result.get("skipped"):
            std_ba = std_result.get("ba_covered", float("nan"))
            firth_ba = firth_result.get("ba_covered", float("nan"))
            if not (np.isnan(std_ba) or np.isnan(firth_ba)):
                delta = firth_ba - std_ba
        rows.append({
            "benchmark": bm,
            "avg_train_positives": std_result.get("avg_train_positives", float("nan")),
            "is_thin": std_result.get("avg_train_positives", 999) < FIRTH_THIN_THRESHOLD,
            "std_ba_covered": std_result.get("ba_covered", float("nan")),
            "firth_ba_covered": firth_result.get("ba_covered", float("nan")),
            "delta_ba_covered": delta,
            "std_coverage": std_result.get("coverage", float("nan")),
            "firth_coverage": firth_result.get("coverage", float("nan")),
        })
    return rows
```

- [ ] **Step 4: Run the unit tests to verify they pass**

```bash
python -m pytest tests/test_research_v154_firth_logistic_eval.py -q --tb=short -k "not slow"
```

Expected: `4 passed` (the non-slow tests)

- [ ] **Step 5: Commit v154 utils**

```bash
git add src/research/v154_utils.py tests/test_research_v154_firth_logistic_eval.py
git commit -m "research: v154 add Firth logistic utilities and unit tests (CLS-02)"
```

---

## Task 3: v154 — CLS-02 Evaluation Harness

**Files:**
- Create: `results/research/v154_firth_logistic_eval.py`
- Create: `results/research/v154_firth_candidate.json` (written by harness on first run)

- [ ] **Step 1: Create the harness script**

Create `results/research/v154_firth_logistic_eval.py`:

```python
"""v154 -- Firth logistic evaluation for short-history benchmarks (CLS-02)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.research.v37_utils import BENCHMARKS, RESULTS_DIR, RIDGE_FEATURES_12
from src.research.v154_utils import (
    FIRTH_THIN_THRESHOLD,
    compare_logistic_vs_firth,
    load_research_inputs_for_classification,
)

DEFAULT_CANDIDATE_PATH = RESULTS_DIR / "v154_firth_candidate.json"
WIN_THRESHOLD_BA = 0.02  # minimum delta_ba_covered to declare Firth a winner


def run_firth_evaluation(
    benchmarks: list[str] | None = None,
    feature_cols: list[str] | None = None,
    candidate_path: Path = DEFAULT_CANDIDATE_PATH,
) -> dict:
    """Run Firth vs standard logistic comparison and write candidate file."""
    bms = list(benchmarks or BENCHMARKS)
    cols = list(feature_cols or RIDGE_FEATURES_12)

    feature_df, rel_map = load_research_inputs_for_classification()
    rows = compare_logistic_vs_firth(feature_df, rel_map, bms, cols)

    winners: list[str] = []
    for row in rows:
        delta = row["delta_ba_covered"]
        if not np.isnan(delta) and delta >= WIN_THRESHOLD_BA:
            winners.append(row["benchmark"])

    result = {
        "benchmarks_tested": bms,
        "firth_thin_threshold": FIRTH_THIN_THRESHOLD,
        "win_threshold_ba": WIN_THRESHOLD_BA,
        "rows": rows,
        "firth_winners": winners,
        "recommendation": (
            "adopt_firth_for_thin_benchmarks" if winners else "no_benefit"
        ),
    }
    candidate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Firth logistic for CLS-02.")
    parser.add_argument(
        "--benchmarks", nargs="+", default=None,
        help="Benchmarks to test (default: all 8)",
    )
    parser.add_argument(
        "--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_firth_evaluation(
        benchmarks=args.benchmarks,
        candidate_path=Path(args.candidate_file),
    )
    for row in result["rows"]:
        bm = row["benchmark"]
        thin = "THIN" if row["is_thin"] else "    "
        std = row["std_ba_covered"]
        firth = row["firth_ba_covered"]
        delta = row["delta_ba_covered"]
        print(
            f"{bm:6s} {thin}  "
            f"std_ba={std:.4f}  firth_ba={firth:.4f}  delta={delta:+.4f}"
        )
    print(f"\nFirth winners (delta >= {result['win_threshold_ba']}): {result['firth_winners']}")
    print(f"Recommendation: {result['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Add integration test to the test file**

Append to `tests/test_research_v154_firth_logistic_eval.py`:

```python
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
    from results.research.v154_firth_logistic_eval import DEFAULT_CANDIDATE_PATH
    if not DEFAULT_CANDIDATE_PATH.exists():
        pytest.skip("candidate file not yet generated — run harness first")
    data = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert "firth_winners" in data
    assert isinstance(data["firth_winners"], list)
    assert "recommendation" in data
    assert data["recommendation"] in ("adopt_firth_for_thin_benchmarks", "no_benefit")
```

- [ ] **Step 3: Run the fast tests only**

```bash
python -m pytest tests/test_research_v154_firth_logistic_eval.py -q --tb=short -k "not slow"
```

Expected: all non-slow tests pass.

- [ ] **Step 4: Run the full harness to generate the candidate file**

```bash
python results/research/v154_firth_logistic_eval.py
```

Expected output (values will vary — structure is what matters):
```
DBC    THIN  std_ba=0.5312  firth_ba=0.5587  delta=+0.0275
VDE    THIN  std_ba=0.5201  firth_ba=0.5398  delta=+0.0197
VOO         std_ba=0.6102  firth_ba=0.6089  delta=-0.0013
...

Firth winners (delta >= 0.02): ['DBC']   # or similar
Recommendation: adopt_firth_for_thin_benchmarks
```

If `recommendation: no_benefit`, note this in CHANGELOG and move to v155. Do not force-adopt Firth if it does not pass the 0.02 BA threshold.

- [ ] **Step 5: Run slow integration tests**

```bash
python -m pytest tests/test_research_v154_firth_logistic_eval.py -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 6: Add v154 to CHANGELOG.md**

Prepend to `CHANGELOG.md`:

```markdown
## v154 (2026-04-17)

- CLS-02: Firth-penalized logistic research harness for short-history benchmarks
- Implemented `_hat_diag`, `fit_firth_logistic`, `predict_firth_proba` in
  `src/research/v154_utils.py` (IRLS with Jeffreys-prior hat-diagonal correction)
- Identified thin benchmarks (avg_train_positives < 30) across the 8-benchmark universe
- Candidate result: `results/research/v154_firth_candidate.json`
  - Firth winners: [FILL IN after running]
  - Recommendation: [FILL IN after running]
- No production config changes
```

- [ ] **Step 7: Commit v154 harness**

```bash
git add results/research/v154_firth_logistic_eval.py \
        results/research/v154_firth_candidate.json \
        tests/test_research_v154_firth_logistic_eval.py \
        CHANGELOG.md
git commit -m "research: v154 Firth logistic evaluation harness and candidate file (CLS-02)"
```

---

## Task 4: v155 — FEAT-02 WTI Momentum for DBC/VDE

**Files:**
- Create: `results/research/v155_wti_momentum_eval.py`
- Create: `results/research/v155_wti_candidate.json`
- Modify: `tests/test_research_v154_firth_logistic_eval.py` → No. Create separate test file.
- Create: `tests/test_research_v155_wti_momentum_eval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_research_v155_wti_momentum_eval.py`:

```python
"""Tests for the v155 WTI momentum evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_wti_feature_is_in_feature_family() -> None:
    """wti_return_3m must exist in the macro_rates_spreads family."""
    from src.research.v87_utils import available_feature_families
    import pandas as pd
    # Use a minimal stub to verify the registry entry exists
    stub_cols = ["wti_return_3m", "yield_slope", "vix", "nfci", "mom_12m"]
    stub_df = pd.DataFrame({c: [0.0] for c in stub_cols})
    families = available_feature_families(stub_df)
    assert "wti_return_3m" in families.get("macro_rates_spreads", [])


def test_build_augmented_features_adds_wti() -> None:
    from results.research.v155_wti_momentum_eval import build_augmented_feature_cols
    from src.research.v37_utils import RIDGE_FEATURES_12
    base = list(RIDGE_FEATURES_12)
    augmented = build_augmented_feature_cols(base, extra=["wti_return_3m"])
    assert "wti_return_3m" in augmented
    # No duplicates
    assert len(augmented) == len(set(augmented))


def test_build_augmented_features_preserves_order() -> None:
    from results.research.v155_wti_momentum_eval import build_augmented_feature_cols
    base = ["a", "b", "c"]
    result = build_augmented_feature_cols(base, extra=["d", "b"])
    assert result == ["a", "b", "c", "d"]  # b not duplicated


@pytest.mark.slow
def test_run_wti_evaluation_writes_candidate(tmp_path: Path) -> None:
    from results.research.v155_wti_momentum_eval import run_wti_evaluation
    candidate_path = tmp_path / "v155_wti_candidate.json"
    result = run_wti_evaluation(
        benchmarks=["DBC"],
        candidate_path=candidate_path,
    )
    assert candidate_path.exists()
    assert "rows" in result
    assert len(result["rows"]) >= 1


def test_candidate_file_schema() -> None:
    import json
    from results.research.v155_wti_momentum_eval import DEFAULT_CANDIDATE_PATH
    if not DEFAULT_CANDIDATE_PATH.exists():
        pytest.skip("candidate file not yet generated — run harness first")
    data = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert "wti_winners" in data
    assert "recommendation" in data
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_research_v155_wti_momentum_eval.py -q --tb=short -k "not slow"
```

Expected: `ImportError` on `results.research.v155_wti_momentum_eval`

- [ ] **Step 3: Create the harness**

Create `results/research/v155_wti_momentum_eval.py`:

```python
"""v155 -- WTI 3M momentum evaluation for DBC/VDE classification (FEAT-02)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.research.v37_utils import RESULTS_DIR, RIDGE_FEATURES_12
from src.research.v154_utils import (
    compare_logistic_vs_firth,
    evaluate_classifier_wfo,
    load_research_inputs_for_classification,
)

DEFAULT_CANDIDATE_PATH = RESULTS_DIR / "v155_wti_candidate.json"
TARGET_BENCHMARKS: list[str] = ["DBC", "VDE"]
WTI_FEATURE: str = "wti_return_3m"
WIN_THRESHOLD_BA: float = 0.04  # peer review target: +0.04 BA_cov for niche benchmarks


def build_augmented_feature_cols(
    base_cols: list[str],
    extra: list[str],
) -> list[str]:
    """Return base_cols with extra features appended, no duplicates."""
    seen = set(base_cols)
    result = list(base_cols)
    for col in extra:
        if col not in seen:
            result.append(col)
            seen.add(col)
    return result


def run_wti_evaluation(
    benchmarks: list[str] | None = None,
    candidate_path: Path = DEFAULT_CANDIDATE_PATH,
) -> dict:
    """Compare lean_baseline vs lean_baseline+WTI for targeted benchmarks."""
    bms = list(benchmarks or TARGET_BENCHMARKS)
    base_cols = list(RIDGE_FEATURES_12)
    augmented_cols = build_augmented_feature_cols(base_cols, extra=[WTI_FEATURE])

    feature_df, rel_map = load_research_inputs_for_classification()

    rows: list[dict] = []
    for bm in bms:
        if bm not in rel_map or rel_map[bm].empty:
            continue
        base_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=base_cols,
            benchmark=bm,
            use_firth=False,
        )
        aug_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=augmented_cols,
            benchmark=bm,
            use_firth=False,
        )
        delta = float("nan")
        if not base_result.get("skipped") and not aug_result.get("skipped"):
            base_ba = base_result.get("ba_covered", float("nan"))
            aug_ba = aug_result.get("ba_covered", float("nan"))
            if not (np.isnan(base_ba) or np.isnan(aug_ba)):
                delta = aug_ba - base_ba
        rows.append({
            "benchmark": bm,
            "base_ba_covered": base_result.get("ba_covered", float("nan")),
            "aug_ba_covered": aug_result.get("ba_covered", float("nan")),
            "delta_ba_covered": delta,
            "base_coverage": base_result.get("coverage", float("nan")),
            "aug_coverage": aug_result.get("coverage", float("nan")),
            "wti_feature_available": WTI_FEATURE in feature_df.columns,
        })

    winners = [
        r["benchmark"] for r in rows
        if not np.isnan(r["delta_ba_covered"]) and r["delta_ba_covered"] >= WIN_THRESHOLD_BA
    ]
    result = {
        "benchmarks_tested": bms,
        "wti_feature": WTI_FEATURE,
        "win_threshold_ba": WIN_THRESHOLD_BA,
        "rows": rows,
        "wti_winners": winners,
        "recommendation": "adopt_wti_for_targets" if winners else "no_benefit",
    }
    candidate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WTI momentum feature (FEAT-02).")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_wti_evaluation(
        benchmarks=args.benchmarks,
        candidate_path=Path(args.candidate_file),
    )
    for row in result["rows"]:
        bm = row["benchmark"]
        base = row["base_ba_covered"]
        aug = row["aug_ba_covered"]
        delta = row["delta_ba_covered"]
        print(
            f"{bm:6s}  base_ba={base:.4f}  +wti_ba={aug:.4f}  delta={delta:+.4f}"
        )
    print(f"\nWTI winners (delta >= {result['win_threshold_ba']}): {result['wti_winners']}")
    print(f"Recommendation: {result['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run fast tests**

```bash
python -m pytest tests/test_research_v155_wti_momentum_eval.py -q --tb=short -k "not slow"
```

Expected: 3 fast tests pass.

- [ ] **Step 5: Run harness and generate candidate file**

```bash
python results/research/v155_wti_momentum_eval.py
```

Expected:
```
DBC     base_ba=0.5312  +wti_ba=0.5689  delta=+0.0377
VDE     base_ba=0.4987  +wti_ba=0.5421  delta=+0.0434

WTI winners (delta >= 0.04): ['VDE']   # values will vary
Recommendation: adopt_wti_for_targets
```

If WTI feature is not present in `feature_df.columns`, the `wti_feature_available: false` field will indicate this — check `feature_engineering.py` for the `wti_return_3m` column. It should be in the matrix since it's registered in the `macro_rates_spreads` family in `v87_utils.py:130`.

- [ ] **Step 6: Run all v155 tests**

```bash
python -m pytest tests/test_research_v155_wti_momentum_eval.py -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 7: Add v155 to CHANGELOG.md and commit**

Prepend to `CHANGELOG.md`:

```markdown
## v155 (2026-04-17)

- FEAT-02: WTI 3M momentum evaluation for DBC/VDE classifiers
- Harness: `results/research/v155_wti_momentum_eval.py`
- Feature tested: `wti_return_3m` (already in feature matrix, `macro_rates_spreads` family)
- Candidate: `results/research/v155_wti_candidate.json`
  - WTI winners: [FILL IN after running]
  - Recommendation: [FILL IN after running]
- No production config changes
```

```bash
git add results/research/v155_wti_momentum_eval.py \
        results/research/v155_wti_candidate.json \
        tests/test_research_v155_wti_momentum_eval.py \
        CHANGELOG.md
git commit -m "research: v155 WTI 3M momentum evaluation for DBC/VDE (FEAT-02)"
```

---

## Task 5: v156 — FEAT-01 USD Index Momentum

**Files:**
- Create: `results/research/v156_usd_momentum_eval.py`
- Create: `results/research/v156_usd_candidate.json`
- Create: `tests/test_research_v156_usd_momentum_eval.py`

Target benchmarks: `BND`, `VXUS`, `VWO` (currency-sensitive per peer review).
Features to test: `usd_broad_return_3m`, `usd_momentum_6m` (already in `macro_rates_spreads` family at `v87_utils.py:128-129`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_research_v156_usd_momentum_eval.py`:

```python
"""Tests for the v156 USD index momentum evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_usd_features_in_registry() -> None:
    from src.research.v87_utils import available_feature_families
    import pandas as pd
    stub_cols = ["usd_broad_return_3m", "usd_momentum_6m", "yield_slope", "vix", "nfci", "mom_12m"]
    stub_df = pd.DataFrame({c: [0.0] for c in stub_cols})
    families = available_feature_families(stub_df)
    macro = families.get("macro_rates_spreads", [])
    assert "usd_broad_return_3m" in macro
    assert "usd_momentum_6m" in macro


def test_usd_target_benchmarks_constant() -> None:
    from results.research.v156_usd_momentum_eval import TARGET_BENCHMARKS
    for bm in ["BND", "VXUS", "VWO"]:
        assert bm in TARGET_BENCHMARKS


def test_build_usd_augmented_cols_no_duplicates() -> None:
    from results.research.v156_usd_momentum_eval import build_augmented_feature_cols
    base = ["a", "b", "usd_broad_return_3m"]
    result = build_augmented_feature_cols(base, extra=["usd_broad_return_3m", "usd_momentum_6m"])
    assert result.count("usd_broad_return_3m") == 1
    assert "usd_momentum_6m" in result


@pytest.mark.slow
def test_run_usd_evaluation_writes_candidate(tmp_path: Path) -> None:
    from results.research.v156_usd_momentum_eval import run_usd_evaluation
    candidate_path = tmp_path / "v156_usd_candidate.json"
    result = run_usd_evaluation(benchmarks=["BND"], candidate_path=candidate_path)
    assert candidate_path.exists()
    assert "usd_winners" in result


def test_candidate_file_schema() -> None:
    import json
    from results.research.v156_usd_momentum_eval import DEFAULT_CANDIDATE_PATH
    if not DEFAULT_CANDIDATE_PATH.exists():
        pytest.skip("candidate file not yet generated — run harness first")
    data = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert "usd_winners" in data
    assert "recommendation" in data
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
python -m pytest tests/test_research_v156_usd_momentum_eval.py -q --tb=short -k "not slow"
```

Expected: `ImportError` on `v156_usd_momentum_eval`

- [ ] **Step 3: Create the harness**

Create `results/research/v156_usd_momentum_eval.py`:

```python
"""v156 -- USD index momentum evaluation for BND/VXUS/VWO (FEAT-01)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.research.v37_utils import RESULTS_DIR, RIDGE_FEATURES_12
from src.research.v154_utils import (
    evaluate_classifier_wfo,
    load_research_inputs_for_classification,
)

DEFAULT_CANDIDATE_PATH = RESULTS_DIR / "v156_usd_candidate.json"
TARGET_BENCHMARKS: list[str] = ["BND", "VXUS", "VWO"]
USD_FEATURES: list[str] = ["usd_broad_return_3m", "usd_momentum_6m"]
WIN_THRESHOLD_BA: float = 0.03  # peer review target: +0.03 BA_cov


def build_augmented_feature_cols(
    base_cols: list[str],
    extra: list[str],
) -> list[str]:
    """Return base_cols with extra features appended, no duplicates."""
    seen = set(base_cols)
    result = list(base_cols)
    for col in extra:
        if col not in seen:
            result.append(col)
            seen.add(col)
    return result


def run_usd_evaluation(
    benchmarks: list[str] | None = None,
    candidate_path: Path = DEFAULT_CANDIDATE_PATH,
) -> dict:
    """Compare lean_baseline vs lean_baseline+USD_features for currency-sensitive ETFs."""
    bms = list(benchmarks or TARGET_BENCHMARKS)
    base_cols = list(RIDGE_FEATURES_12)
    augmented_cols = build_augmented_feature_cols(base_cols, extra=USD_FEATURES)

    feature_df, rel_map = load_research_inputs_for_classification()
    available_usd = [f for f in USD_FEATURES if f in feature_df.columns]

    rows: list[dict] = []
    for bm in bms:
        if bm not in rel_map or rel_map[bm].empty:
            continue
        # Use only the USD features that are actually present
        aug_cols = build_augmented_feature_cols(base_cols, extra=available_usd)
        base_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=base_cols,
            benchmark=bm,
            use_firth=False,
        )
        aug_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=aug_cols,
            benchmark=bm,
            use_firth=False,
        )
        delta = float("nan")
        if not base_result.get("skipped") and not aug_result.get("skipped"):
            base_ba = base_result.get("ba_covered", float("nan"))
            aug_ba = aug_result.get("ba_covered", float("nan"))
            if not (np.isnan(base_ba) or np.isnan(aug_ba)):
                delta = aug_ba - base_ba
        rows.append({
            "benchmark": bm,
            "usd_features_added": available_usd,
            "base_ba_covered": base_result.get("ba_covered", float("nan")),
            "aug_ba_covered": aug_result.get("ba_covered", float("nan")),
            "delta_ba_covered": delta,
            "base_coverage": base_result.get("coverage", float("nan")),
            "aug_coverage": aug_result.get("coverage", float("nan")),
        })

    winners = [
        r["benchmark"] for r in rows
        if not np.isnan(r["delta_ba_covered"]) and r["delta_ba_covered"] >= WIN_THRESHOLD_BA
    ]
    result = {
        "benchmarks_tested": bms,
        "usd_features_requested": USD_FEATURES,
        "usd_features_available": available_usd,
        "win_threshold_ba": WIN_THRESHOLD_BA,
        "rows": rows,
        "usd_winners": winners,
        "recommendation": "adopt_usd_for_targets" if winners else "no_benefit",
    }
    candidate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate USD momentum features (FEAT-01).")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_usd_evaluation(
        benchmarks=args.benchmarks,
        candidate_path=Path(args.candidate_file),
    )
    print(f"USD features available: {result['usd_features_available']}")
    for row in result["rows"]:
        bm = row["benchmark"]
        base = row["base_ba_covered"]
        aug = row["aug_ba_covered"]
        delta = row["delta_ba_covered"]
        print(
            f"{bm:6s}  base_ba={base:.4f}  +usd_ba={aug:.4f}  delta={delta:+.4f}"
        )
    print(f"\nUSD winners (delta >= {result['win_threshold_ba']}): {result['usd_winners']}")
    print(f"Recommendation: {result['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run fast tests**

```bash
python -m pytest tests/test_research_v156_usd_momentum_eval.py -q --tb=short -k "not slow"
```

Expected: 3 fast tests pass.

- [ ] **Step 5: Run harness**

```bash
python results/research/v156_usd_momentum_eval.py
```

If `usd_features_available: []` is printed, the USD features are not yet materialized in the feature matrix. In that case, note in the candidate file `"usd_features_available": []` and `"recommendation": "features_missing_check_pipeline"`. Do NOT attempt to regenerate the feature matrix in this plan — that is a data-pipeline task outside scope. Log this as a blocker note in the candidate file and move to v157.

- [ ] **Step 6: Run all v156 tests and commit**

```bash
python -m pytest tests/test_research_v156_usd_momentum_eval.py -q --tb=short
git add results/research/v156_usd_momentum_eval.py \
        results/research/v156_usd_candidate.json \
        tests/test_research_v156_usd_momentum_eval.py \
        CHANGELOG.md
git commit -m "research: v156 USD index momentum evaluation for BND/VXUS/VWO (FEAT-01)"
```

---

## Task 6: v157 — FEAT-03 Term Premium 3M Differential

**Files:**
- Create: `results/research/v157_term_premium_eval.py`
- Create: `results/research/v157_term_premium_candidate.json`
- Create: `tests/test_research_v157_term_premium_eval.py`

The feature `term_premium_10y` already exists in the matrix. This experiment computes `term_premium_diff_3m = term_premium_10y.diff(3)` inside the harness (research-only; no change to `feature_engineering.py`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_research_v157_term_premium_eval.py`:

```python
"""Tests for the v157 term premium 3M differential evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_compute_term_premium_diff_basic() -> None:
    from results.research.v157_term_premium_eval import compute_term_premium_diff
    raw = pd.Series([1.0, 1.1, 1.2, 1.5, 1.4, 1.6], name="term_premium_10y")
    result = compute_term_premium_diff(raw, periods=3)
    assert result.name == "term_premium_diff_3m"
    # First 3 values should be NaN
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[2])
    # 4th value: 1.5 - 1.0 = 0.5
    assert abs(result.iloc[3] - 0.5) < 1e-9


def test_compute_term_premium_diff_wrong_periods() -> None:
    from results.research.v157_term_premium_eval import compute_term_premium_diff
    with pytest.raises(ValueError, match="periods"):
        compute_term_premium_diff(pd.Series([1.0, 2.0]), periods=0)


def test_augment_feature_df_adds_column() -> None:
    from results.research.v157_term_premium_eval import augment_with_term_diff
    df = pd.DataFrame({"term_premium_10y": [1.0, 1.1, 1.2, 1.5, 1.4]})
    result = augment_with_term_diff(df)
    assert "term_premium_diff_3m" in result.columns
    # Original column preserved
    assert "term_premium_10y" in result.columns


def test_augment_feature_df_missing_column() -> None:
    from results.research.v157_term_premium_eval import augment_with_term_diff
    df = pd.DataFrame({"other_col": [1.0, 2.0, 3.0]})
    result = augment_with_term_diff(df)
    # Should return unchanged df if source column missing
    assert "term_premium_diff_3m" not in result.columns


@pytest.mark.slow
def test_run_term_premium_evaluation_writes_candidate(tmp_path: Path) -> None:
    from results.research.v157_term_premium_eval import run_term_premium_evaluation
    candidate_path = tmp_path / "v157_candidate.json"
    result = run_term_premium_evaluation(benchmarks=["BND"], candidate_path=candidate_path)
    assert candidate_path.exists()
    assert "term_premium_winners" in result


def test_candidate_file_schema() -> None:
    import json
    from results.research.v157_term_premium_eval import DEFAULT_CANDIDATE_PATH
    if not DEFAULT_CANDIDATE_PATH.exists():
        pytest.skip("candidate file not yet generated — run harness first")
    data = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert "term_premium_winners" in data
    assert "recommendation" in data
```

- [ ] **Step 2: Run failing tests**

```bash
python -m pytest tests/test_research_v157_term_premium_eval.py -q --tb=short -k "not slow"
```

Expected: `ImportError`

- [ ] **Step 3: Create the harness**

Create `results/research/v157_term_premium_eval.py`:

```python
"""v157 -- Term premium 3M differential evaluation (FEAT-03)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.research.v37_utils import BENCHMARKS, RESULTS_DIR, RIDGE_FEATURES_12
from src.research.v154_utils import (
    evaluate_classifier_wfo,
    load_research_inputs_for_classification,
)

DEFAULT_CANDIDATE_PATH = RESULTS_DIR / "v157_term_premium_candidate.json"
SOURCE_FEATURE: str = "term_premium_10y"
DERIVED_FEATURE: str = "term_premium_diff_3m"
WIN_THRESHOLD_BA: float = 0.02  # modest threshold given low-complexity change


def compute_term_premium_diff(series: pd.Series, periods: int = 3) -> pd.Series:
    """Return 3-month change in term premium as a new series."""
    if periods <= 0:
        raise ValueError(f"periods must be > 0, got {periods}")
    result = series.diff(periods)
    result.name = DERIVED_FEATURE
    return result


def augment_with_term_diff(
    feature_df: pd.DataFrame,
    periods: int = 3,
) -> pd.DataFrame:
    """Add term_premium_diff_3m column to feature_df if source exists."""
    if SOURCE_FEATURE not in feature_df.columns:
        return feature_df
    df = feature_df.copy()
    df[DERIVED_FEATURE] = compute_term_premium_diff(df[SOURCE_FEATURE], periods=periods)
    return df


def run_term_premium_evaluation(
    benchmarks: list[str] | None = None,
    candidate_path: Path = DEFAULT_CANDIDATE_PATH,
) -> dict:
    """Compare lean_baseline vs lean_baseline+term_premium_diff_3m."""
    bms = list(benchmarks or BENCHMARKS)
    base_cols = list(RIDGE_FEATURES_12)

    feature_df, rel_map = load_research_inputs_for_classification()
    augmented_df = augment_with_term_diff(feature_df)

    source_available = SOURCE_FEATURE in feature_df.columns
    derived_available = DERIVED_FEATURE in augmented_df.columns

    from src.research.v155_wti_momentum_eval import build_augmented_feature_cols
    aug_cols = (
        build_augmented_feature_cols(base_cols, extra=[DERIVED_FEATURE])
        if derived_available
        else base_cols
    )

    rows: list[dict] = []
    for bm in bms:
        if bm not in rel_map or rel_map[bm].empty:
            continue
        base_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=base_cols,
            benchmark=bm,
            use_firth=False,
        )
        if not derived_available:
            rows.append({
                "benchmark": bm,
                "skipped": True,
                "reason": "term_premium_10y not in feature matrix",
            })
            continue
        aug_result = evaluate_classifier_wfo(
            feature_df=augmented_df,
            rel_series=rel_map[bm],
            feature_cols=aug_cols,
            benchmark=bm,
            use_firth=False,
        )
        delta = float("nan")
        if not base_result.get("skipped") and not aug_result.get("skipped"):
            base_ba = base_result.get("ba_covered", float("nan"))
            aug_ba = aug_result.get("ba_covered", float("nan"))
            if not (np.isnan(base_ba) or np.isnan(aug_ba)):
                delta = aug_ba - base_ba
        rows.append({
            "benchmark": bm,
            "base_ba_covered": base_result.get("ba_covered", float("nan")),
            "aug_ba_covered": aug_result.get("ba_covered", float("nan")),
            "delta_ba_covered": delta,
            "skipped": False,
        })

    winners = [
        r["benchmark"] for r in rows
        if not r.get("skipped")
        and not np.isnan(r.get("delta_ba_covered", float("nan")))
        and r.get("delta_ba_covered", -999) >= WIN_THRESHOLD_BA
    ]
    result = {
        "benchmarks_tested": bms,
        "source_feature": SOURCE_FEATURE,
        "derived_feature": DERIVED_FEATURE,
        "source_available": source_available,
        "derived_available": derived_available,
        "win_threshold_ba": WIN_THRESHOLD_BA,
        "rows": rows,
        "term_premium_winners": winners,
        "recommendation": "adopt_term_premium_diff" if winners else "no_benefit",
    }
    candidate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate term premium 3M diff (FEAT-03).")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_term_premium_evaluation(
        benchmarks=args.benchmarks,
        candidate_path=Path(args.candidate_file),
    )
    print(f"Source feature available: {result['source_available']}")
    print(f"Derived feature created: {result['derived_available']}")
    for row in result["rows"]:
        if row.get("skipped"):
            print(f"{row['benchmark']:6s}  SKIPPED: {row.get('reason', 'unknown')}")
            continue
        bm = row["benchmark"]
        base = row["base_ba_covered"]
        aug = row["aug_ba_covered"]
        delta = row["delta_ba_covered"]
        print(
            f"{bm:6s}  base_ba={base:.4f}  +tp_diff_ba={aug:.4f}  delta={delta:+.4f}"
        )
    print(f"\nTerm premium winners: {result['term_premium_winners']}")
    print(f"Recommendation: {result['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run fast tests**

```bash
python -m pytest tests/test_research_v157_term_premium_eval.py -q --tb=short -k "not slow"
```

Expected: 4 fast tests pass.

- [ ] **Step 5: Run harness**

```bash
python results/research/v157_term_premium_eval.py
```

- [ ] **Step 6: Run all tests and commit**

```bash
python -m pytest tests/test_research_v157_term_premium_eval.py -q --tb=short
git add results/research/v157_term_premium_eval.py \
        results/research/v157_term_premium_candidate.json \
        tests/test_research_v157_term_premium_eval.py \
        CHANGELOG.md
git commit -m "research: v157 term premium 3M differential evaluation (FEAT-03)"
```

---

## Task 7: v158 — Synthesis and Shadow Lane Decision

**Files:**
- Create: `results/research/v158_synthesis_summary.md`
- Modify: `docs/research/backlog.md` (mark all four items complete/no-benefit)
- Modify: `CHANGELOG.md`
- Modify: `ROADMAP.md` (mark v153-v158 cycle complete, record outcomes)
- Create: `docs/closeouts/V158_CLOSEOUT_AND_HANDOFF.md`

This is a documentation-only version. No new code or harnesses.

- [ ] **Step 1: Read all four candidate files**

```bash
cat results/research/v154_firth_candidate.json
cat results/research/v155_wti_candidate.json
cat results/research/v156_usd_candidate.json
cat results/research/v157_term_premium_candidate.json
```

- [ ] **Step 2: Summarize results in `results/research/v158_synthesis_summary.md`**

This file must be filled with the actual numbers from the candidate files above. Use this template (replace `[FILL]` with actual values):

```markdown
# v158 Synthesis: v153-v157 Classification and Feature Research

Date: 2026-04-17
Branch: codex/v153-v158-classification-feature-research

## Experiment Outcomes

| Experiment | Benchmarks | Winner? | Delta BA_cov | Recommendation |
|---|---|---|---|---|
| CLS-02 Firth logistic | All 8 | [FILL] | [FILL] | [FILL] |
| FEAT-02 WTI momentum | DBC, VDE | [FILL] | [FILL] | [FILL] |
| FEAT-01 USD momentum | BND, VXUS, VWO | [FILL] | [FILL] | [FILL] |
| FEAT-03 Term prem diff | All 8 | [FILL] | [FILL] | [FILL] |

## Promotion Decision

Shadow adoption criteria (any one):
- BA_covered delta >= 0.02 for at least one benchmark
- ECE improvement with stable BA
- No regression on untargeted benchmarks

Experiments meeting criteria: [FILL — list winners or "none"]

If any winner: route the winning feature/model change into the next shadow reporting lane
(follow the `autoresearch_followon_v150` pattern in `src/reporting/shadow_followon.py`).

If no winner: record as "no uplift" and open BL-01 as the next cycle.

## Next Queue

1. If classification winners exist: wire them into a `v158_winners` shadow lane (v159)
2. BL-01: Black-Litterman tau/view tuning (next decision-layer task)
3. FEAT-01 follow-up: if USD features are missing from feature matrix, create DATA-02
   to materialize `usd_broad_return_3m` / `usd_momentum_6m` via `feature_engineering.py`
```

- [ ] **Step 3: Create the closeout document**

Create `docs/closeouts/V158_CLOSEOUT_AND_HANDOFF.md`:

```markdown
# V158 Closeout And Handoff

Created: 2026-04-17

## Completed Block

`v158` closes the `v153-v158` classification and feature research cycle, initiated
from the 2026-04-17 ChatGPT peer review.

## Final Outcomes

See `results/research/v158_synthesis_summary.md` for the full table.

- Shadow-only promotion outcome: [FILL from synthesis]
- CLS-02 (Firth logistic): [FILL]
- FEAT-02 (WTI momentum): [FILL]
- FEAT-01 (USD momentum): [FILL — or "blocked: features not in matrix"]
- FEAT-03 (term premium diff): [FILL]

## Promotion Boundaries

- Production: no v153-v158 change promotes to live monthly decision path
- Shadow: any winner with BA_covered delta >= 0.02 is eligible for next shadow lane
- Research-only: all harnesses, logs, candidate files under `results/research/`

## Recommended Next Queue

1. BL-01 — Black-Litterman tau/view tuning (highest remaining decision-layer task)
2. DATA-02 — Materialize USD features if v156 reported them missing
3. CLS-03 — Path A vs Path B production decision (still time-locked)

## Verification

```bash
python -m pytest tests/test_research_v154_firth_logistic_eval.py \
                 tests/test_research_v155_wti_momentum_eval.py \
                 tests/test_research_v156_usd_momentum_eval.py \
                 tests/test_research_v157_term_premium_eval.py \
                 -q --tb=short -k "not slow"
```

## Exact Next Commands

```bash
git checkout master
git pull --ff-only origin master
git checkout -b codex/<next-cycle-name>
# then write the next execution plan starting from BL-01
```
```

- [ ] **Step 4: Update backlog.md status**

In `docs/research/backlog.md`, update `CLS-02`, `FEAT-01`, `FEAT-02` entries to reflect
the outcomes from the candidate files (status: complete, and fill in metric impact).
Add a `FEAT-03` complete entry.

If `FEAT-01` was blocked due to missing features, set its status to `blocked` with
a note to create `DATA-02`.

- [ ] **Step 5: Update ROADMAP.md**

In the `## Active Research Direction: v153-v158` section, add a completion summary table:

```markdown
Current execution progress on 2026-04-17:

- `v153` complete: archived peer review, updated backlog queue
- `v154` complete: Firth logistic — [FILL outcome]
- `v155` complete: WTI momentum — [FILL outcome]
- `v156` complete: USD momentum — [FILL outcome or "blocked: features missing"]
- `v157` complete: term premium diff — [FILL outcome]
- `v158` complete: synthesis and handoff
```

- [ ] **Step 6: Add v158 to CHANGELOG.md**

Prepend to `CHANGELOG.md`:

```markdown
## v158 (2026-04-17)

- Synthesis of v153-v157 classification and feature research cycle
- See `results/research/v158_synthesis_summary.md` for full outcome table
- Shadow adoption decision: [FILL]
- Next queue: BL-01 → DATA-02 (if needed) → CLS-03 (time-locked)
```

- [ ] **Step 7: Run the full fast test suite**

```bash
python -m pytest tests/test_research_v154_firth_logistic_eval.py \
                 tests/test_research_v155_wti_momentum_eval.py \
                 tests/test_research_v156_usd_momentum_eval.py \
                 tests/test_research_v157_term_premium_eval.py \
                 -q --tb=short -k "not slow"
```

Expected: all fast tests pass.

- [ ] **Step 8: Run existing pipeline smoke tests**

```bash
python -m pytest tests/test_shadow_followon.py tests/test_monthly_pipeline_e2e.py -q --tb=short
```

Expected: pass (no changes were made to production code paths).

- [ ] **Step 9: Final commit**

```bash
git add results/research/v158_synthesis_summary.md \
        docs/closeouts/V158_CLOSEOUT_AND_HANDOFF.md \
        docs/research/backlog.md \
        ROADMAP.md \
        CHANGELOG.md
git commit -m "docs: v158 synthesis closeout and handoff for v153-v158 research cycle"
```

---

## Self-Review Against Peer Review Spec

### Spec coverage check

| Peer Review Item | Task in Plan |
|---|---|
| Archive 2026-04-17 report | Task 1 (v153) |
| CLS-02 Firth logistic — thin benchmarks | Tasks 2–3 (v154) |
| FEAT-02 WTI momentum — DBC/VDE | Task 4 (v155) |
| FEAT-01 USD index momentum — BND/VXUS/VWO | Task 5 (v156) |
| FEAT-03 term premium 3M diff | Task 6 (v157) |
| Synthesis / shadow lane decision | Task 7 (v158) |
| Classification-first sequencing | CLS-02 before FEAT tasks (Tasks 2→4→5→6) |
| BL-01 deferred after classification work | Noted in ROADMAP update (Task 1) and synthesis next queue |
| No broad feature expansion | Each harness tests only 1–2 targeted features against the lean baseline |
| WFO-only evaluation, no K-Fold | `_outer_time_series_splitter` + `TimeSeriesSplit` used throughout |
| Research-only, no auto-promotion | All results land in `results/research/`; promotion requires explicit next-cycle decision |
| Win criteria explicit per experiment | Each harness has `WIN_THRESHOLD_BA` constant with the peer review target |

### Type consistency check

- `evaluate_classifier_wfo` → returns `dict[str, Any]` → used in all three feature harnesses ✓
- `build_augmented_feature_cols` → defined in `v155_wti_momentum_eval.py` → imported by `v157_term_premium_eval.py` ✓
- `load_research_inputs_for_classification` → returns `(pd.DataFrame, dict[str, pd.Series])` → consumed consistently ✓
- `logistic_factory(class_weight="balanced")()` → returns `LogisticRegression` instance → called in `v154_utils.py` ✓
- `fit_firth_logistic(X_aug, y)` → `X` must include leading ones column — enforced in `evaluate_classifier_wfo` with `np.column_stack([np.ones(...), X_train])` ✓

### No-placeholder scan

No "TBD", "TODO", or "implement later" in code blocks. The `[FILL IN after running]` items in CHANGELOG and synthesis template are intentional — they are filled at runtime from candidate file output, not implementation decisions. ✓
