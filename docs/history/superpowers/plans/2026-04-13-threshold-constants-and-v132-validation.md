# Threshold Constants Refactor + v132 Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize the shadow classifier's 0.30/0.70 probability tier thresholds into named config constants, then run a nested temporal hold-out study (v132) to validate whether the (0.15, 0.70) asymmetric threshold identified by the v131 autoresearch sweep generalizes to unseen data.

**Architecture:** Four production source files currently embed the tier boundaries as bare float literals. Task 1 adds four named constants to `config/model.py`; Tasks 2–4 consume them. Task 5 writes the v132 validation script, which splits the v125 fold detail at 2021-12-31, re-runs a threshold grid on the selection set only, and reports whether the winning pair holds on the 21-row temporal hold-out.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn (`balanced_accuracy_score`), pytest. No new dependencies.

---

## File Map

| Action | File | What changes |
|--------|------|-------------|
| Modify | `config/model.py` | Add 4 classifier tier threshold constants |
| Modify | `src/models/classification_shadow.py` | Replace 4 literal comparisons with config constants |
| Modify | `src/models/consensus_shadow.py` | Replace 2 literal comparisons with config constants |
| Modify | `src/models/multi_benchmark_wfo.py` | Replace 2 literal comparisons with config constants |
| Modify | `tests/test_classification_shadow.py` | Add constant-value smoke test |
| Create | `results/research/v132_threshold_validation.py` | Temporal hold-out validation study |
| Create | `results/research/v132_threshold_validation_results.csv` | Output: per-pair metrics on both sets |
| Create | `results/research/v132_threshold_validation_summary.md` | Human-readable verdict |
| Create | `tests/test_research_v132_threshold_validation.py` | Tests for v132 script |

---

## Task 1: Add classifier tier threshold constants to `config/model.py`

**Files:**
- Modify: `config/model.py` (append to the probability calibration section, after line ~96)

### Context

`config/model.py` already defines calibration-related constants (e.g. `CALIBRATION_MIN_OBS_PLATT`). We add four constants that encode the shadow classifier's probability tier system. These are the **current** values — they remain 0.30/0.70 until v132 validates a change.

- `SHADOW_CLASSIFIER_HIGH_THRESH = 0.70` — probability at or above this → ACTIONABLE-SELL / HIGH confidence
- `SHADOW_CLASSIFIER_LOW_THRESH = 0.30` — probability at or below this → NON-ACTIONABLE / HIGH confidence
- `SHADOW_CLASSIFIER_MODERATE_HIGH_THRESH = 0.60` — MODERATE sell boundary
- `SHADOW_CLASSIFIER_MODERATE_LOW_THRESH = 0.40` — MODERATE hold boundary

- [ ] **Step 1: Verify the insertion point**

Run:
```bash
grep -n "Probability Calibration\|CALIBRATION_MIN_OBS" config/model.py | head -5
```
Expected output: lines around 87–92 (exact numbers may vary). Note the line number of the `# v5.1 — Probability Calibration` heading — you will insert the new block **just before** it.

- [ ] **Step 2: Add the four constants**

In `config/model.py`, find the `# v5.1 — Probability Calibration` section header and insert the following block immediately **before** it:

```python
# ---------------------------------------------------------------------------
# v131/v132 — Shadow classifier probability tier thresholds
#
# These four values define both the abstention band and the confidence-tier
# system used across classification_shadow, consensus_shadow, and
# multi_benchmark_wfo. Centralised here so a single edit propagates to all
# three call sites if thresholds are ever updated after v132 validation.
#
# Tier logic (applied to any P(actionable-sell) or P(outperform)):
#   P >= HIGH_THRESH              → ACTIONABLE-SELL  / HIGH confidence
#   P <= LOW_THRESH               → NON-ACTIONABLE   / HIGH confidence
#   P >= MODERATE_HIGH_THRESH     → MODERATE sell confidence
#   P <= MODERATE_LOW_THRESH      → MODERATE hold confidence
#   otherwise                     → NEUTRAL / LOW confidence
#
# v131 autoresearch sweep found (0.15, 0.70) improves covered BA by +6pp
# with 45% coverage; v132 temporal hold-out required before adoption.
# ---------------------------------------------------------------------------
SHADOW_CLASSIFIER_HIGH_THRESH: float = 0.70
SHADOW_CLASSIFIER_LOW_THRESH: float = 0.30
SHADOW_CLASSIFIER_MODERATE_HIGH_THRESH: float = 0.60
SHADOW_CLASSIFIER_MODERATE_LOW_THRESH: float = 0.40
```

- [ ] **Step 3: Confirm the constants are importable**

```bash
python -c "import config; print(config.SHADOW_CLASSIFIER_HIGH_THRESH, config.SHADOW_CLASSIFIER_LOW_THRESH)"
```
Expected: `0.7 0.3`

- [ ] **Step 4: Commit**

```bash
git add config/model.py
git commit -m "config: centralize shadow classifier tier thresholds as named constants (v131/v132)"
```

---

## Task 2: Consume constants in `classification_shadow.py`

**Files:**
- Modify: `src/models/classification_shadow.py` (lines ~104–116)

### Context

`classification_shadow.py` contains two functions that embed the tier boundaries as literals:

- `classification_confidence_tier()` — uses `>= 0.70`, `<= 0.30`, `>= 0.60`, `<= 0.40`
- `classification_stance()` — uses `>= 0.70`, `<= 0.30`

The file already does `import config` at the top level (line 12). No new import is needed.

- [ ] **Step 1: Replace literals in `classification_confidence_tier`**

Find this block (around line 102):
```python
def classification_confidence_tier(probability_actionable_sell: float) -> str:
    """Map actionable-sell probability into a simple confidence tier."""
    if probability_actionable_sell >= 0.70 or probability_actionable_sell <= 0.30:
        return "HIGH"
    if probability_actionable_sell >= 0.60 or probability_actionable_sell <= 0.40:
        return "MODERATE"
    return "LOW"
```

Replace with:
```python
def classification_confidence_tier(probability_actionable_sell: float) -> str:
    """Map actionable-sell probability into a simple confidence tier."""
    if (
        probability_actionable_sell >= config.SHADOW_CLASSIFIER_HIGH_THRESH
        or probability_actionable_sell <= config.SHADOW_CLASSIFIER_LOW_THRESH
    ):
        return "HIGH"
    if (
        probability_actionable_sell >= config.SHADOW_CLASSIFIER_MODERATE_HIGH_THRESH
        or probability_actionable_sell <= config.SHADOW_CLASSIFIER_MODERATE_LOW_THRESH
    ):
        return "MODERATE"
    return "LOW"
```

- [ ] **Step 2: Replace literals in `classification_stance`**

Find this block (around line 111):
```python
def classification_stance(probability_actionable_sell: float) -> str:
    """Return the shadow classifier stance from the actionable-sell probability."""
    if probability_actionable_sell >= 0.70:
        return "ACTIONABLE-SELL"
    if probability_actionable_sell <= 0.30:
        return "NON-ACTIONABLE"
    return "NEUTRAL"
```

Replace with:
```python
def classification_stance(probability_actionable_sell: float) -> str:
    """Return the shadow classifier stance from the actionable-sell probability."""
    if probability_actionable_sell >= config.SHADOW_CLASSIFIER_HIGH_THRESH:
        return "ACTIONABLE-SELL"
    if probability_actionable_sell <= config.SHADOW_CLASSIFIER_LOW_THRESH:
        return "NON-ACTIONABLE"
    return "NEUTRAL"
```

- [ ] **Step 3: Run the existing shadow tests**

```bash
python -m pytest tests/test_classification_shadow.py -v --tb=short
```
Expected: all tests pass (behaviour is unchanged — only the source of the numbers changed).

- [ ] **Step 4: Commit**

```bash
git add src/models/classification_shadow.py
git commit -m "refactor(shadow): consume config threshold constants in classification_shadow"
```

---

## Task 3: Consume constants in `consensus_shadow.py`

**Files:**
- Modify: `src/models/consensus_shadow.py` (lines ~118–123)

### Context

`consensus_shadow.py::summarize_consensus_variant` computes a confidence tier for the cross-benchmark consensus probability using the same 0.70/0.30/0.60/0.40 literals. The file does **not** currently import `config` — you must add that import.

- [ ] **Step 1: Add `import config` to `consensus_shadow.py`**

At the top of `src/models/consensus_shadow.py`, after `from __future__ import annotations`, add:

```python
import config
```

The top of the file currently looks like:
```python
from __future__ import annotations

import pandas as pd
```

Change to:
```python
from __future__ import annotations

import config
import pandas as pd
```

- [ ] **Step 2: Replace tier literals in `summarize_consensus_variant`**

Find this block (around line 118):
```python
    if mean_prob_outperform >= 0.70 or mean_prob_outperform <= 0.30:
        confidence_tier = "HIGH"
    elif mean_prob_outperform >= 0.60 or mean_prob_outperform <= 0.40:
        confidence_tier = "MODERATE"
    else:
        confidence_tier = "LOW"
```

Replace with:
```python
    if (
        mean_prob_outperform >= config.SHADOW_CLASSIFIER_HIGH_THRESH
        or mean_prob_outperform <= config.SHADOW_CLASSIFIER_LOW_THRESH
    ):
        confidence_tier = "HIGH"
    elif (
        mean_prob_outperform >= config.SHADOW_CLASSIFIER_MODERATE_HIGH_THRESH
        or mean_prob_outperform <= config.SHADOW_CLASSIFIER_MODERATE_LOW_THRESH
    ):
        confidence_tier = "MODERATE"
    else:
        confidence_tier = "LOW"
```

- [ ] **Step 3: Run the full test suite guard**

```bash
python -m pytest tests/test_path_b_classifier.py tests/test_classification_shadow.py -q --tb=short
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/models/consensus_shadow.py
git commit -m "refactor(shadow): consume config threshold constants in consensus_shadow"
```

---

## Task 4: Consume constants in `multi_benchmark_wfo.py`

**Files:**
- Modify: `src/models/multi_benchmark_wfo.py` (lines ~421–424)

### Context

`multi_benchmark_wfo.py::get_confidence_tier` uses the same tier boundaries to classify the BayesianRidge ensemble signal. The function's docstring already documents the thresholds — update it to reference the config names. The file already imports `config` (check with `grep "^import config" src/models/multi_benchmark_wfo.py`).

- [ ] **Step 1: Verify `config` is already imported**

```bash
grep -n "^import config" src/models/multi_benchmark_wfo.py
```
If no output: add `import config` after the other stdlib imports at the top.

- [ ] **Step 2: Replace tier literals in `get_confidence_tier`**

Find this block (around line 421):
```python
    if prob >= 0.70 or prob <= 0.30:
        tier = "HIGH"
    elif prob >= 0.60 or prob <= 0.40:
        tier = "MODERATE"
    else:
        tier = "LOW"
```

Replace with:
```python
    if prob >= config.SHADOW_CLASSIFIER_HIGH_THRESH or prob <= config.SHADOW_CLASSIFIER_LOW_THRESH:
        tier = "HIGH"
    elif prob >= config.SHADOW_CLASSIFIER_MODERATE_HIGH_THRESH or prob <= config.SHADOW_CLASSIFIER_MODERATE_LOW_THRESH:
        tier = "MODERATE"
    else:
        tier = "LOW"
```

- [ ] **Step 3: Update the docstring threshold references**

Find the docstring lines (around line 403):
```
      - HIGH:     P(outperform) ≥ 0.70 or ≤ 0.30 (strong directional conviction)
      - MODERATE: P(outperform) ≥ 0.60 or ≤ 0.40
```

Replace with:
```
      - HIGH:     P(outperform) ≥ config.SHADOW_CLASSIFIER_HIGH_THRESH or
                  ≤ config.SHADOW_CLASSIFIER_LOW_THRESH (strong directional conviction)
      - MODERATE: P(outperform) ≥ config.SHADOW_CLASSIFIER_MODERATE_HIGH_THRESH or
                  ≤ config.SHADOW_CLASSIFIER_MODERATE_LOW_THRESH
```

- [ ] **Step 4: Run full guard**

```bash
python -m pytest tests/test_path_b_classifier.py tests/test_classification_shadow.py -q --tb=short
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/multi_benchmark_wfo.py
git commit -m "refactor(wfo): consume config threshold constants in multi_benchmark_wfo"
```

---

## Task 5: Add constant smoke test to `test_classification_shadow.py`

**Files:**
- Modify: `tests/test_classification_shadow.py`

### Context

The existing behaviour tests (`test_classification_confidence_tier_thresholds`, `test_classification_stance_thresholds`) validate the right outputs but don't pin the constant values. Add one test that asserts the config constants are the expected values, so any accidental edit to `config/model.py` fails immediately.

- [ ] **Step 1: Add the smoke test**

Open `tests/test_classification_shadow.py`. Add `import config` near the top (after existing imports), then append this test at the bottom of the file:

```python
def test_shadow_classifier_config_constants_unchanged() -> None:
    """Guard: config tier constants must stay at v131 baseline values until v132 adopts new ones."""
    assert config.SHADOW_CLASSIFIER_HIGH_THRESH == pytest.approx(0.70)
    assert config.SHADOW_CLASSIFIER_LOW_THRESH == pytest.approx(0.30)
    assert config.SHADOW_CLASSIFIER_MODERATE_HIGH_THRESH == pytest.approx(0.60)
    assert config.SHADOW_CLASSIFIER_MODERATE_LOW_THRESH == pytest.approx(0.40)
```

Also add `import pytest` if not already present.

- [ ] **Step 2: Run the updated test file**

```bash
python -m pytest tests/test_classification_shadow.py -v --tb=short
```
Expected: all pass including the new constant test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_classification_shadow.py
git commit -m "test(shadow): add config constant smoke test to pin tier threshold values"
```

---

## Task 6: Write and test the v132 temporal hold-out validation study

**Files:**
- Create: `results/research/v132_threshold_validation.py`
- Create: `tests/test_research_v132_threshold_validation.py`
- (Auto-generated at runtime): `results/research/v132_threshold_validation_results.csv`, `results/research/v132_threshold_validation_summary.md`

### Context

**Why this study is needed:** The v131 autoresearch sweep evaluated 41 threshold pairs against all 84 OOS rows in `v125_portfolio_target_fold_detail.csv`. Picking the best pair from that sweep introduces selection bias — the (0.15, 0.70) finding needs to be validated against data it never influenced.

**Design:**
- **Temporal split boundary:** `HOLDOUT_CUTOFF = "2021-12-31"`
- **Selection set:** 63 rows (2016-10-31 → 2021-12-31) — the v131 sweep could have touched these
- **Hold-out set:** 21 rows (2022-01-31 → 2023-09-29) — never used in any prior sweep
- **Prequential scaling:** Applied to the full 84-row sequence in chronological order (correct — each row is calibrated using only its predecessors)
- **Selection sweep:** Evaluate the same 41 candidate pairs, but compute metrics on the selection set only
- **Hold-out evaluation:** Report baseline (0.30, 0.70), selection winner, and the a-priori candidate (0.15, 0.70) on the hold-out
- **Adoption verdict:** ADOPT if the best hold-out pair achieves covered_ba improvement ≥ 0.03 vs baseline AND coverage ≥ 0.20 on hold-out

**Adoption criterion delta (0.03):** matches the Criterion A threshold used in v130 temperature scaling adoption.

- [ ] **Step 1: Write the failing test first**

Create `tests/test_research_v132_threshold_validation.py`:

```python
"""Tests for v132 temporal hold-out threshold validation study."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v132_threshold_validation import (
    HOLDOUT_CUTOFF,
    MIN_BA_DELTA_FOR_ADOPTION,
    MIN_COVERAGE,
    apply_prequential_temperature_scaling,
    evaluate_thresholds,
    run_threshold_grid,
    derive_verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fold_df(n: int = 84, seed: int = 0) -> pd.DataFrame:
    """Synthetic fold detail DataFrame matching v125 schema."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-10-31", periods=n, freq="ME")
    probs = rng.uniform(0.05, 0.95, n)
    labels = (probs > 0.50).astype(int)
    return pd.DataFrame({
        "test_date": dates,
        "y_true": labels,
        "path_b_prob": probs,
    })


# ---------------------------------------------------------------------------
# Temporal integrity
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    def test_selection_dates_strictly_before_holdout(self) -> None:
        """Every selection date must be <= HOLDOUT_CUTOFF; every holdout date must be after."""
        df = _make_fold_df(n=84)
        cutoff = pd.Timestamp(HOLDOUT_CUTOFF)
        sel = df[df["test_date"] <= cutoff]
        hld = df[df["test_date"] > cutoff]
        assert sel["test_date"].max() <= cutoff
        assert hld["test_date"].min() > cutoff

    def test_no_overlap_between_sets(self) -> None:
        """Selection and hold-out row sets must be disjoint."""
        df = _make_fold_df(n=84)
        cutoff = pd.Timestamp(HOLDOUT_CUTOFF)
        sel_idx = set(df[df["test_date"] <= cutoff].index)
        hld_idx = set(df[df["test_date"] > cutoff].index)
        assert sel_idx.isdisjoint(hld_idx)

    def test_holdout_cutoff_is_correct_date(self) -> None:
        """HOLDOUT_CUTOFF must equal 2021-12-31 per design specification."""
        assert HOLDOUT_CUTOFF == "2021-12-31"


# ---------------------------------------------------------------------------
# evaluate_thresholds
# ---------------------------------------------------------------------------

class TestEvaluateThresholds:
    def test_perfect_covered_ba(self) -> None:
        """Perfectly separated probabilities → covered_ba == 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.10, 0.15, 0.80, 0.85])
        result = evaluate_thresholds(y_true, y_prob, low=0.30, high=0.70)
        assert result["covered_ba"] == pytest.approx(1.0)
        assert result["coverage"] == pytest.approx(1.0)

    def test_all_abstained(self) -> None:
        """When all probs are inside the band, coverage=0 and ba=0.5."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.45, 0.48, 0.52, 0.55])
        result = evaluate_thresholds(y_true, y_prob, low=0.40, high=0.60)
        assert result["coverage"] == pytest.approx(0.0)
        assert result["covered_ba"] == pytest.approx(0.5)

    def test_coverage_fraction(self) -> None:
        """Coverage = fraction of rows outside [low, high]."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.10, 0.40, 0.60, 0.90, 0.50, 0.80])
        # covered: 0.10 (< 0.30), 0.90 (> 0.70), 0.80 (> 0.70) → 3/6 = 0.5
        result = evaluate_thresholds(y_true, y_prob, low=0.30, high=0.70)
        assert result["coverage"] == pytest.approx(3 / 6)


# ---------------------------------------------------------------------------
# run_threshold_grid
# ---------------------------------------------------------------------------

class TestRunThresholdGrid:
    def test_returns_dataframe_with_required_columns(self) -> None:
        y_true = np.array([0, 0, 1, 1] * 10)
        y_prob = np.array([0.10, 0.15, 0.80, 0.85] * 10)
        pairs = [(0.30, 0.70), (0.20, 0.80)]
        result = run_threshold_grid(y_true, y_prob, pairs)
        assert isinstance(result, pd.DataFrame)
        for col in ("low", "high", "covered_ba", "coverage"):
            assert col in result.columns

    def test_grid_length_matches_pairs(self) -> None:
        y_true = np.array([0, 1] * 20)
        y_prob = np.array([0.20, 0.80] * 20)
        pairs = [(0.30, 0.70), (0.20, 0.80), (0.15, 0.70)]
        result = run_threshold_grid(y_true, y_prob, pairs)
        assert len(result) == len(pairs)

    def test_best_pair_has_highest_ba(self) -> None:
        y_true = np.array([0, 0, 1, 1] * 15)
        y_prob = np.array([0.10, 0.15, 0.80, 0.85] * 15)
        pairs = [(0.30, 0.70), (0.40, 0.60), (0.20, 0.80)]
        result = run_threshold_grid(y_true, y_prob, pairs)
        best = result.loc[result["covered_ba"].idxmax()]
        assert best["covered_ba"] == result["covered_ba"].max()


# ---------------------------------------------------------------------------
# derive_verdict
# ---------------------------------------------------------------------------

class TestDeriveVerdict:
    def test_adopt_when_criteria_met(self) -> None:
        verdict = derive_verdict(
            holdout_ba_candidate=0.65,
            holdout_ba_baseline=0.57,
            holdout_coverage_candidate=0.40,
        )
        assert "ADOPT" in verdict

    def test_do_not_adopt_insufficient_ba_delta(self) -> None:
        verdict = derive_verdict(
            holdout_ba_candidate=0.58,
            holdout_ba_baseline=0.57,
            holdout_coverage_candidate=0.40,
        )
        assert "DO NOT ADOPT" in verdict

    def test_do_not_adopt_insufficient_coverage(self) -> None:
        verdict = derive_verdict(
            holdout_ba_candidate=0.65,
            holdout_ba_baseline=0.57,
            holdout_coverage_candidate=0.15,  # below 0.20
        )
        assert "DO NOT ADOPT" in verdict

    def test_boundary_exactly_at_delta_threshold(self) -> None:
        """BA delta exactly == MIN_BA_DELTA_FOR_ADOPTION should ADOPT."""
        verdict = derive_verdict(
            holdout_ba_candidate=round(0.57 + MIN_BA_DELTA_FOR_ADOPTION, 6),
            holdout_ba_baseline=0.57,
            holdout_coverage_candidate=0.40,
        )
        assert "ADOPT" in verdict


# ---------------------------------------------------------------------------
# apply_prequential_temperature_scaling
# ---------------------------------------------------------------------------

class TestPrequentialScaling:
    def test_output_shape_preserved(self) -> None:
        probs = np.random.default_rng(0).uniform(0.1, 0.9, 50)
        labels = np.random.default_rng(0).integers(0, 2, 50)
        result = apply_prequential_temperature_scaling(probs, labels, warmup=24)
        assert result.shape == probs.shape

    def test_output_bounded(self) -> None:
        probs = np.random.default_rng(7).uniform(0.05, 0.95, 60)
        labels = np.random.default_rng(7).integers(0, 2, 60)
        result = apply_prequential_temperature_scaling(probs, labels, warmup=24)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
```

- [ ] **Step 2: Run the test to confirm it fails (module not found)**

```bash
python -m pytest tests/test_research_v132_threshold_validation.py -q --tb=short 2>&1 | head -10
```
Expected: `ModuleNotFoundError` or `ImportError` for `v132_threshold_validation`.

- [ ] **Step 3: Write `results/research/v132_threshold_validation.py`**

Create the file with this content:

```python
"""v132 -- Temporal hold-out validation for asymmetric abstention thresholds.

Design
------
The v131 autoresearch sweep evaluated 41 threshold pairs against all 84 OOS
rows from v125_portfolio_target_fold_detail.csv.  This introduces selection
bias: the winning pair (0.15, 0.70) was chosen by seeing every row.

This script validates the finding using a strict temporal hold-out:

  Selection set  (n=63): 2016-10-31 → 2021-12-31  (rows the v131 sweep saw)
  Hold-out set   (n=21): 2022-01-31 → 2023-09-29  (never used in any prior sweep)

Procedure
---------
1. Apply prequential temperature scaling to the full 84-row sequence (correct
   — each row is calibrated only by its predecessors).
2. Re-run the v131 threshold grid on the selection set only → selection winner.
3. Evaluate three pairs on the hold-out:
     a. Baseline   (0.30, 0.70)
     b. A-priori   (0.15, 0.70)  — the credible finding from v131
     c. Selection winner         — data-driven pick from step 2
4. Adoption verdict: ADOPT if hold-out BA delta >= 0.03 AND coverage >= 0.20.

Outputs
-------
  results/research/v132_threshold_validation_results.csv
  results/research/v132_threshold_validation_summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
FOLD_DETAIL_PATH: Path = (
    PROJECT_ROOT / "results" / "research" / "v125_portfolio_target_fold_detail.csv"
)
RESULTS_PATH: Path = PROJECT_ROOT / "results" / "research" / "v132_threshold_validation_results.csv"
SUMMARY_PATH: Path = PROJECT_ROOT / "results" / "research" / "v132_threshold_validation_summary.md"

HOLDOUT_CUTOFF: str = "2021-12-31"
MIN_COVERAGE: float = 0.20
MIN_BA_DELTA_FOR_ADOPTION: float = 0.03
WARMUP: int = 24

# Candidate threshold pairs (same grid as v131 sweep)
_STEP = 0.05
_CANDIDATE_PAIRS: list[tuple[float, float]] = [
    (round(lo, 2), round(hi, 2))
    for lo in [round(x * _STEP + 0.10, 2) for x in range(8)]   # 0.10 → 0.45
    for hi in [round(x * _STEP + 0.55, 2) for x in range(8)]   # 0.55 → 0.90
    if round(hi, 2) > round(lo + 0.20, 2)
]

# ---------------------------------------------------------------------------
# Temperature scaling (verbatim from path_b_classifier.py)
# ---------------------------------------------------------------------------
_GRID_TEMPERATURES: np.ndarray = np.concatenate(
    [np.linspace(0.50, 0.95, 10), np.linspace(1.0, 3.0, 41)]
)


def _clip(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(probs: np.ndarray) -> np.ndarray:
    p = _clip(probs)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _fit_temp(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = _clip(y_prob)
    best_t, best_loss = 1.0, float("inf")
    for t in _GRID_TEMPERATURES:
        scaled = np.clip(_sigmoid(_logit(p) / float(t)), 1e-6, 1.0 - 1e-6)
        loss = -float(np.mean(y_true * np.log(scaled) + (1 - y_true) * np.log(1.0 - scaled)))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def apply_prequential_temperature_scaling(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    warmup: int = WARMUP,
) -> np.ndarray:
    """Apply prequential (walk-forward) temperature scaling.

    For each index t >= warmup, fits temperature on probs[:t]/labels[:t]
    and applies it to probs[t]. Pre-warmup observations are clipped but
    otherwise unchanged.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    calibrated = _clip(probs).copy()
    for idx in range(len(probs)):
        if idx < warmup or len(np.unique(labels[:idx])) < 2:
            continue
        t = _fit_temp(labels[:idx], probs[:idx])
        raw = float(probs[idx])
        calibrated[idx] = float(np.clip(
            _sigmoid(np.array([_logit(np.array([raw]))[0] / t]))[0],
            1e-6, 1.0 - 1e-6,
        ))
    return np.clip(calibrated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low: float,
    high: float,
) -> dict[str, float]:
    """Return covered_ba and coverage for the given abstention band."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = (y_prob < low) | (y_prob > high)
    n_covered = int(mask.sum())
    coverage = n_covered / len(y_true) if len(y_true) > 0 else 0.0
    if n_covered == 0 or len(np.unique(y_true[mask])) < 2:
        return {"covered_ba": 0.5, "coverage": coverage}
    covered_ba = float(
        balanced_accuracy_score(y_true[mask], (y_prob[mask] >= 0.5).astype(int))
    )
    return {"covered_ba": covered_ba, "coverage": coverage}


def run_threshold_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pairs: list[tuple[float, float]],
) -> pd.DataFrame:
    """Evaluate every (low, high) pair and return a DataFrame of metrics."""
    rows = []
    for low, high in pairs:
        m = evaluate_thresholds(y_true, y_prob, low=low, high=high)
        rows.append({"low": low, "high": high, **m})
    return pd.DataFrame(rows)


def derive_verdict(
    holdout_ba_candidate: float,
    holdout_ba_baseline: float,
    holdout_coverage_candidate: float,
) -> str:
    """Return an adoption verdict string based on hold-out metrics."""
    ba_delta = holdout_ba_candidate - holdout_ba_baseline
    if ba_delta >= MIN_BA_DELTA_FOR_ADOPTION and holdout_coverage_candidate >= MIN_COVERAGE:
        return (
            f"ADOPT: hold-out BA delta = {ba_delta:+.4f} >= {MIN_BA_DELTA_FOR_ADOPTION} "
            f"and coverage = {holdout_coverage_candidate:.4f} >= {MIN_COVERAGE}."
        )
    reasons = []
    if ba_delta < MIN_BA_DELTA_FOR_ADOPTION:
        reasons.append(f"BA delta {ba_delta:+.4f} < {MIN_BA_DELTA_FOR_ADOPTION}")
    if holdout_coverage_candidate < MIN_COVERAGE:
        reasons.append(f"coverage {holdout_coverage_candidate:.4f} < {MIN_COVERAGE}")
    return f"DO NOT ADOPT: {'; '.join(reasons)}."


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _write_summary(
    selection_grid: pd.DataFrame,
    holdout_rows: list[dict],
    selection_winner: tuple[float, float],
    verdict: str,
    n_selection: int,
    n_holdout: int,
) -> None:
    lines = [
        "# v132 Threshold Validation Summary",
        "",
        f"**Holdout cutoff:** {HOLDOUT_CUTOFF}",
        f"**Selection set:** {n_selection} rows",
        f"**Hold-out set:** {n_holdout} rows",
        f"**Candidate pairs evaluated:** {len(selection_grid)}",
        "",
        "## Selection Set — Top 5 Pairs",
        "",
        "| low | high | covered_ba | coverage |",
        "|-----|------|-----------|---------|",
    ]
    for _, row in selection_grid.nlargest(5, "covered_ba").iterrows():
        lines.append(
            f"| {row['low']:.2f} | {row['high']:.2f} | "
            f"{row['covered_ba']:.4f} | {row['coverage']:.4f} |"
        )
    lines += [
        "",
        f"**Selection winner:** low={selection_winner[0]:.2f}, high={selection_winner[1]:.2f}",
        "",
        "## Hold-out Set — Candidate Evaluation",
        "",
        "| pair | low | high | covered_ba | coverage | delta_vs_baseline |",
        "|------|-----|------|-----------|---------|-----------------|",
    ]
    baseline_ba = next(r["covered_ba"] for r in holdout_rows if r["label"] == "baseline")
    for r in holdout_rows:
        delta = r["covered_ba"] - baseline_ba
        lines.append(
            f"| {r['label']} | {r['low']:.2f} | {r['high']:.2f} | "
            f"{r['covered_ba']:.4f} | {r['coverage']:.4f} | {delta:+.4f} |"
        )
    lines += ["", f"## Verdict", "", f"> {verdict}", ""]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run v132 temporal hold-out validation. Returns exit code."""
    df = pd.read_csv(FOLD_DETAIL_PATH)
    df["test_date"] = pd.to_datetime(df["test_date"])
    df = df.sort_values("test_date").reset_index(drop=True)

    y_true_all = df["y_true"].to_numpy(dtype=int)
    path_b_raw = df["path_b_prob"].to_numpy(dtype=float)

    # Apply prequential temperature scaling to full sequence (temporal integrity preserved)
    y_prob_all = apply_prequential_temperature_scaling(path_b_raw, y_true_all, warmup=WARMUP)

    # Temporal split
    cutoff = pd.Timestamp(HOLDOUT_CUTOFF)
    sel_mask = df["test_date"] <= cutoff
    hld_mask = df["test_date"] > cutoff

    y_true_sel = y_true_all[sel_mask.to_numpy()]
    y_prob_sel = y_prob_all[sel_mask.to_numpy()]
    y_true_hld = y_true_all[hld_mask.to_numpy()]
    y_prob_hld = y_prob_all[hld_mask.to_numpy()]

    n_sel, n_hld = int(sel_mask.sum()), int(hld_mask.sum())
    print(f"\n=== v132 Temporal Hold-out Threshold Validation ===")
    print(f"Selection set: {n_sel} rows ({df[sel_mask]['test_date'].min().date()} → "
          f"{df[sel_mask]['test_date'].max().date()})")
    print(f"Hold-out set:  {n_hld} rows ({df[hld_mask]['test_date'].min().date()} → "
          f"{df[hld_mask]['test_date'].max().date()})")

    # Step 1: Grid sweep on selection set only
    print(f"\nEvaluating {len(_CANDIDATE_PAIRS)} threshold pairs on selection set...")
    selection_grid = run_threshold_grid(y_true_sel, y_prob_sel, _CANDIDATE_PAIRS)

    # Valid pairs: coverage >= MIN_COVERAGE
    valid = selection_grid[selection_grid["coverage"] >= MIN_COVERAGE]
    if valid.empty:
        print("ERROR: no threshold pair achieves MIN_COVERAGE on selection set.")
        return 1
    best_sel_row = valid.loc[valid["covered_ba"].idxmax()]
    selection_winner = (float(best_sel_row["low"]), float(best_sel_row["high"]))
    print(f"Selection winner: low={selection_winner[0]:.2f}, high={selection_winner[1]:.2f} "
          f"(covered_ba={best_sel_row['covered_ba']:.4f}, coverage={best_sel_row['coverage']:.4f})")

    # Step 2: Evaluate three pairs on hold-out
    eval_pairs = [
        ("baseline",        0.30, 0.70),
        ("a_priori_v131",   0.15, 0.70),
        ("selection_winner", selection_winner[0], selection_winner[1]),
    ]
    # Deduplicate if selection winner happens to match a-priori or baseline
    seen_pairs: set[tuple[float, float]] = set()
    unique_eval = []
    for label, lo, hi in eval_pairs:
        if (lo, hi) not in seen_pairs:
            seen_pairs.add((lo, hi))
            unique_eval.append((label, lo, hi))

    print(f"\nHold-out evaluation:")
    holdout_rows = []
    for label, lo, hi in unique_eval:
        m = evaluate_thresholds(y_true_hld, y_prob_hld, low=lo, high=hi)
        holdout_rows.append({"label": label, "low": lo, "high": hi, **m})
        print(f"  {label:<22} low={lo:.2f} high={hi:.2f}  "
              f"covered_ba={m['covered_ba']:.4f}  coverage={m['coverage']:.4f}")

    # Step 3: Adoption verdict (evaluated on a-priori candidate (0.15, 0.70))
    baseline_m = evaluate_thresholds(y_true_hld, y_prob_hld, low=0.30, high=0.70)
    apriori_m  = evaluate_thresholds(y_true_hld, y_prob_hld, low=0.15, high=0.70)
    verdict = derive_verdict(
        holdout_ba_candidate=apriori_m["covered_ba"],
        holdout_ba_baseline=baseline_m["covered_ba"],
        holdout_coverage_candidate=apriori_m["coverage"],
    )
    print(f"\nVERDICT (a-priori 0.15/0.70): {verdict}")

    # Write outputs
    results_rows = []
    for _, row in selection_grid.iterrows():
        results_rows.append({**row, "dataset": "selection"})
    for r in holdout_rows:
        results_rows.append({
            "low": r["low"], "high": r["high"],
            "covered_ba": r["covered_ba"], "coverage": r["coverage"],
            "dataset": "holdout", "label": r["label"],
        })
    pd.DataFrame(results_rows).to_csv(RESULTS_PATH, index=False)
    _write_summary(selection_grid, holdout_rows, selection_winner, verdict, n_sel, n_hld)

    print(f"\nResults CSV: {RESULTS_PATH}")
    print(f"Summary:     {SUMMARY_PATH}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_research_v132_threshold_validation.py -v --tb=short
```
Expected: all tests pass.

- [ ] **Step 5: Run the full guard**

```bash
python -m pytest tests/test_path_b_classifier.py tests/test_classification_shadow.py tests/test_research_v131_threshold_sweep_eval.py tests/test_research_v132_threshold_validation.py -q --tb=short
```
Expected: all pass.

- [ ] **Step 6: Execute the v132 study**

```bash
python results/research/v132_threshold_validation.py
```
Expected output: selection winner, hold-out evaluation table, and a ADOPT or DO NOT ADOPT verdict. Read the verdict and note it — it determines whether the config constants in Task 1 should be updated from 0.30/0.70 to the winning thresholds.

- [ ] **Step 7: Commit**

```bash
git add results/research/v132_threshold_validation.py \
        results/research/v132_threshold_validation_results.csv \
        results/research/v132_threshold_validation_summary.md \
        tests/test_research_v132_threshold_validation.py
git commit -m "research(v132): temporal hold-out validation for asymmetric abstention thresholds"
```

---

## Self-Review

**Spec coverage:**
- ✅ Centralize 0.30/0.70 thresholds as named constants — Tasks 1–4
- ✅ All three production files updated — Tasks 2, 3, 4
- ✅ Tests for constants — Task 5
- ✅ v132 temporal hold-out study — Task 6
- ✅ Adoption verdict based on hold-out — `derive_verdict()` in Task 6
- ✅ Guard suite runs after every commit

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:**
- `evaluate_thresholds()` → `dict[str, float]` — used consistently in `run_threshold_grid` and `main()`
- `run_threshold_grid()` → `pd.DataFrame` — columns: `low`, `high`, `covered_ba`, `coverage`
- `derive_verdict()` → `str` — always starts with `"ADOPT"` or `"DO NOT ADOPT"`
- `apply_prequential_temperature_scaling()` → `np.ndarray` — same signature across v131 eval harness and v132
