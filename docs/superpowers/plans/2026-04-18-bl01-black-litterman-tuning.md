# BL-01 Black-Litterman Tau/View Tuning Implementation Plan

Status: historical plan. Execution is complete; see
`docs/closeouts/BL01_CLOSEOUT_AND_HANDOFF.md` and the `BL-01` section in
`CHANGELOG.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a reproducible Monte Carlo parameter sweep over tau × risk_aversion to determine whether updating the BL model's current defaults (tau=0.05, risk_aversion=2.5) improves IC-rank-weighted portfolio quality, then update `config/model.py` only if the winner exceeds the uplift threshold.

**Architecture:** A standalone research harness in `results/research/` generates 50 seeded synthetic scenarios — each with 60-month ETF returns plus ensemble signals whose IC and MAE are randomly sampled from realistic ranges — and runs the existing `build_bl_weights()` under a 5×5 grid of (tau, risk_aversion) combinations. Each combination is scored by the Spearman rank correlation between the resulting BL weights and the per-benchmark IC values (higher = BL correctly upweights better signals). The winner is selected if it beats the incumbent by ≥ 0.05 rank correlation. Results are written to `results/research/bl01_tau_candidate.json`. `config/model.py` is updated only when the candidate recommends it. The BL model remains a shadow diagnostic throughout — no live recommendation path changes.

**Tech Stack:** Python 3.10+, numpy, pandas. PyPortfolioOpt is used indirectly via the existing `build_bl_weights()` function (already installed).

---

## Prerequisite: Branch Setup

```bash
git checkout master
git pull --ff-only origin master
git checkout -b codex/bl01-black-litterman-tuning
```

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `tests/test_bl01_sweep.py` | Create | Unit tests for sweep harness helper functions |
| `results/research/bl01_tau_sweep_eval.py` | Create | Monte Carlo grid sweep harness |
| `results/research/bl01_tau_candidate.json` | Create (by harness) | Winning parameter set |
| `config/model.py` | Modify (conditional) | Update BL_TAU / BL_RISK_AVERSION if winner recommended |
| `docs/closeouts/BL01_CLOSEOUT_AND_HANDOFF.md` | Create | Closeout and next-session handoff |
| `CHANGELOG.md` | Modify | Prepend BL-01 entry |
| `ROADMAP.md` | Modify | Mark BL-01 complete |

---

## Task 1: Create Sweep Harness + Tests (TDD)

**Files:**
- Create: `tests/test_bl01_sweep.py`
- Create: `results/research/bl01_tau_sweep_eval.py`

### Step 1: Write failing tests

Create `tests/test_bl01_sweep.py`:

```python
"""Unit tests for the BL-01 tau/risk_aversion sweep harness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_make_scenario_returns_correct_shapes() -> None:
    from results.research.bl01_tau_sweep_eval import BENCHMARKS, _make_scenario
    returns_df, signals = _make_scenario(seed=0, benchmarks=BENCHMARKS, n_months=60)
    assert returns_df.shape == (60, len(BENCHMARKS))
    assert set(signals.keys()) == set(BENCHMARKS)


def test_make_scenario_different_seeds_differ() -> None:
    from results.research.bl01_tau_sweep_eval import BENCHMARKS, _make_scenario
    returns_a, signals_a = _make_scenario(seed=0, benchmarks=BENCHMARKS)
    returns_b, signals_b = _make_scenario(seed=1, benchmarks=BENCHMARKS)
    assert not returns_a.equals(returns_b)
    ics_a = [signals_a[b].mean_ic for b in BENCHMARKS]
    ics_b = [signals_b[b].mean_ic for b in BENCHMARKS]
    assert ics_a != ics_b


def test_rank_corr_perfect_positive() -> None:
    from results.research.bl01_tau_sweep_eval import _rank_corr
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(_rank_corr(x, x) - 1.0) < 1e-9


def test_rank_corr_perfect_negative() -> None:
    from results.research.bl01_tau_sweep_eval import _rank_corr
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(_rank_corr(x, x[::-1]) + 1.0) < 1e-9


def test_rank_corr_short_input_returns_zero() -> None:
    from results.research.bl01_tau_sweep_eval import _rank_corr
    assert _rank_corr(np.array([1.0, 2.0]), np.array([2.0, 1.0])) == 0.0


def test_compute_ic_rank_correlation_positive_alignment() -> None:
    """Weights concentrating in high-IC benchmarks → high rank correlation."""
    from results.research.bl01_tau_sweep_eval import (
        _compute_ic_rank_correlation,
        _make_signal,
    )
    benchmarks = ["A", "B", "C", "D"]
    ics = [0.20, 0.15, 0.10, 0.05]
    weights = {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10}
    signals = {bm: _make_signal(bm, ic) for bm, ic in zip(benchmarks, ics)}
    corr = _compute_ic_rank_correlation(weights, signals)
    assert corr > 0.9


def test_compute_ic_rank_correlation_negative_alignment() -> None:
    """Weights concentrating in low-IC benchmarks → low (negative) rank correlation."""
    from results.research.bl01_tau_sweep_eval import (
        _compute_ic_rank_correlation,
        _make_signal,
    )
    benchmarks = ["A", "B", "C", "D"]
    ics = [0.20, 0.15, 0.10, 0.05]
    weights = {"A": 0.10, "B": 0.20, "C": 0.30, "D": 0.40}
    signals = {bm: _make_signal(bm, ic) for bm, ic in zip(benchmarks, ics)}
    corr = _compute_ic_rank_correlation(weights, signals)
    assert corr < -0.9


def test_select_winner_keeps_incumbent_when_delta_below_threshold() -> None:
    from results.research.bl01_tau_sweep_eval import select_winner
    rows = [
        {"tau": 0.05, "risk_aversion": 2.5, "mean_rank_corr": 0.50, "fallback_rate": 0.05},
        {"tau": 0.10, "risk_aversion": 2.0, "mean_rank_corr": 0.54, "fallback_rate": 0.05},
    ]
    result = select_winner(rows, incumbent_tau=0.05, incumbent_ra=2.5, win_threshold=0.05)
    assert result["bl_tau_winner"] == 0.05
    assert result["bl_risk_aversion_winner"] == 2.5
    assert result["recommendation"] == "keep_incumbent"


def test_select_winner_returns_winner_when_delta_exceeds_threshold() -> None:
    from results.research.bl01_tau_sweep_eval import select_winner
    rows = [
        {"tau": 0.05, "risk_aversion": 2.5, "mean_rank_corr": 0.50, "fallback_rate": 0.05},
        {"tau": 0.10, "risk_aversion": 2.0, "mean_rank_corr": 0.60, "fallback_rate": 0.05},
    ]
    result = select_winner(rows, incumbent_tau=0.05, incumbent_ra=2.5, win_threshold=0.05)
    assert result["bl_tau_winner"] == 0.10
    assert result["bl_risk_aversion_winner"] == 2.0
    assert result["recommendation"] == "update_bl_params"


def test_select_winner_output_keys() -> None:
    from results.research.bl01_tau_sweep_eval import select_winner
    rows = [
        {"tau": 0.05, "risk_aversion": 2.5, "mean_rank_corr": 0.50, "fallback_rate": 0.05},
    ]
    result = select_winner(rows, incumbent_tau=0.05, incumbent_ra=2.5, win_threshold=0.05)
    for key in [
        "bl_tau_winner",
        "bl_risk_aversion_winner",
        "incumbent_tau",
        "incumbent_risk_aversion",
        "winner_rank_corr",
        "incumbent_rank_corr",
        "delta_rank_corr",
        "win_threshold",
        "recommendation",
    ]:
        assert key in result, f"Missing key: {key}"
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && python -m pytest tests/test_bl01_sweep.py -q --tb=short
```

Expected: `ModuleNotFoundError: No module named 'results.research.bl01_tau_sweep_eval'`

- [ ] **Step 3: Create `results/research/bl01_tau_sweep_eval.py`**

```python
"""BL-01: Monte Carlo parameter sweep for Black-Litterman tau / risk_aversion tuning.

Run:  python results/research/bl01_tau_sweep_eval.py
Writes: results/research/bl01_tau_candidate.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.black_litterman import BLDiagnostics, build_bl_weights
from src.models.multi_benchmark_wfo import EnsembleWFOResult
from src.models.wfo_engine import FoldResult, WFOResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BENCHMARKS: list[str] = ["VOO", "VXUS", "VWO", "VMBS", "BND", "GLD", "DBC", "VDE"]
TAU_GRID: list[float] = [0.01, 0.025, 0.05, 0.10, 0.20]
RISK_AVERSION_GRID: list[float] = [1.5, 2.0, 2.5, 3.0, 4.0]
N_SCENARIOS: int = 50
N_MONTHS: int = 60
INCUMBENT_TAU: float = 0.05
INCUMBENT_RA: float = 2.5
WIN_THRESHOLD_RANK_CORR: float = 0.05

CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "bl01_tau_candidate.json"


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable)
# ---------------------------------------------------------------------------

def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (numpy only, no scipy)."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    return float(1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1)))


def _make_signal(
    benchmark: str,
    mean_ic: float,
    mean_mae: float = 0.05,
    n_preds: int = 6,
    seed: int = 0,
) -> EnsembleWFOResult:
    """Build a minimal EnsembleWFOResult for use in the BL sweep.

    The fold's y_hat is set to mean_ic * 0.12 (annualized IC-proportional return)
    so that the BL view Q vector reflects signal quality ordering.
    """
    rng = np.random.default_rng(seed)
    y_hat = np.full(n_preds, mean_ic * 0.12)
    y_true = y_hat + rng.normal(0, mean_mae, size=n_preds)
    fold = FoldResult(
        fold_idx=0,
        train_start=pd.Timestamp("2019-01-31"),
        train_end=pd.Timestamp("2022-12-31"),
        test_start=pd.Timestamp("2023-01-31"),
        test_end=pd.Timestamp("2023-06-30"),
        y_true=y_true,
        y_hat=y_hat,
        optimal_alpha=0.01,
        feature_importances={},
        n_train=48,
        n_test=n_preds,
    )
    wfo = WFOResult(
        folds=[fold],
        benchmark=benchmark,
        target_horizon=6,
        model_type="elasticnet",
    )
    return EnsembleWFOResult(
        benchmark=benchmark,
        target_horizon=6,
        mean_ic=mean_ic,
        mean_hit_rate=max(0.5 + mean_ic * 2, 0.0),
        mean_mae=mean_mae,
        model_results={"elasticnet": wfo},
    )


def _make_scenario(
    seed: int,
    benchmarks: list[str],
    n_months: int = 60,
) -> tuple[pd.DataFrame, dict[str, EnsembleWFOResult]]:
    """Generate synthetic monthly returns + ensemble signals for one MC scenario.

    Returns (returns_df, signals) where:
    - returns_df: (n_months × n_benchmarks) DataFrame of monthly returns
    - signals: dict benchmark → EnsembleWFOResult with IC ~ U(-0.05, 0.20)
    """
    rng = np.random.default_rng(seed)
    n = len(benchmarks)

    # Realistic ETF covariance: base correlation 0.40, monthly vol 4%
    base_corr = 0.40 * np.ones((n, n)) + 0.60 * np.eye(n)
    monthly_std = 0.04
    cov = (monthly_std ** 2) * base_corr

    idx = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    returns_arr = rng.multivariate_normal(np.full(n, 0.008), cov, size=n_months)
    returns_df = pd.DataFrame(returns_arr, index=idx, columns=benchmarks)

    signals: dict[str, EnsembleWFOResult] = {}
    for i, bm in enumerate(benchmarks):
        ic = float(rng.uniform(-0.05, 0.20))
        mae = float(rng.uniform(0.04, 0.08))
        signals[bm] = _make_signal(bm, ic, mae, seed=seed * 100 + i)

    return returns_df, signals


def _compute_ic_rank_correlation(
    weights: dict[str, float],
    signals: dict[str, EnsembleWFOResult],
) -> float:
    """Spearman rank correlation between BL weights and benchmark IC values.

    Positive values mean higher-IC benchmarks get higher BL weights (good).
    Measured only across benchmarks present in both weights and signals.
    """
    common = sorted(set(weights) & set(signals))
    if len(common) < 3:
        return 0.0
    w = np.array([weights[bm] for bm in common])
    ic = np.array([signals[bm].mean_ic for bm in common])
    return _rank_corr(w, ic)


def run_tau_sweep(
    tau_grid: list[float],
    ra_grid: list[float],
    n_scenarios: int,
    benchmarks: list[str],
    n_months: int = 60,
) -> list[dict[str, Any]]:
    """Run the full (tau × risk_aversion) grid sweep.

    For each combination, evaluate across n_scenarios seeded Monte Carlo draws.
    Returns list of result rows, one per (tau, risk_aversion) combination.

    Row keys: tau, risk_aversion, mean_rank_corr, fallback_rate, n_scenarios.
    """
    rows: list[dict[str, Any]] = []

    for tau in tau_grid:
        for ra in ra_grid:
            rank_corrs: list[float] = []
            n_fallbacks: int = 0

            for seed in range(n_scenarios):
                returns_df, signals = _make_scenario(seed, benchmarks, n_months)
                result = build_bl_weights(
                    signals,
                    returns_df,
                    risk_aversion=ra,
                    return_diagnostics=True,
                )
                # build_bl_weights with tau uses config.BL_TAU unless we pass it
                # We need to override tau; patch config temporarily.
                import config as _cfg
                orig_tau = _cfg.BL_TAU
                _cfg.BL_TAU = tau
                try:
                    result = build_bl_weights(
                        signals,
                        returns_df,
                        risk_aversion=ra,
                        return_diagnostics=True,
                    )
                finally:
                    _cfg.BL_TAU = orig_tau

                weights, diag = result
                if diag.fallback_used:
                    n_fallbacks += 1
                    continue
                rank_corrs.append(_compute_ic_rank_correlation(weights, signals))

            n_valid = len(rank_corrs)
            rows.append({
                "tau": tau,
                "risk_aversion": ra,
                "mean_rank_corr": float(np.mean(rank_corrs)) if rank_corrs else 0.0,
                "fallback_rate": n_fallbacks / n_scenarios,
                "n_valid_scenarios": n_valid,
                "n_scenarios": n_scenarios,
            })

    return rows


def select_winner(
    rows: list[dict[str, Any]],
    incumbent_tau: float,
    incumbent_ra: float,
    win_threshold: float,
) -> dict[str, Any]:
    """Select the winning (tau, risk_aversion) from sweep rows.

    Winner is the row with the highest mean_rank_corr where fallback_rate < 0.50.
    Winner is adopted only if it beats the incumbent's mean_rank_corr by win_threshold.
    If no row beats the threshold, the incumbent parameters are returned.
    """
    # Find the incumbent row
    incumbent_row = next(
        (r for r in rows if r["tau"] == incumbent_tau and r["risk_aversion"] == incumbent_ra),
        {"tau": incumbent_tau, "risk_aversion": incumbent_ra, "mean_rank_corr": 0.0, "fallback_rate": 0.0},
    )
    incumbent_corr = incumbent_row["mean_rank_corr"]

    # Find best row among non-degenerate options (fallback_rate < 0.50)
    eligible = [r for r in rows if r["fallback_rate"] < 0.50]
    if not eligible:
        eligible = rows
    best_row = max(eligible, key=lambda r: r["mean_rank_corr"])
    best_corr = best_row["mean_rank_corr"]
    delta = best_corr - incumbent_corr

    if delta >= win_threshold and (best_row["tau"] != incumbent_tau or best_row["risk_aversion"] != incumbent_ra):
        winner_tau = best_row["tau"]
        winner_ra = best_row["risk_aversion"]
        recommendation = "update_bl_params"
    else:
        winner_tau = incumbent_tau
        winner_ra = incumbent_ra
        recommendation = "keep_incumbent"

    return {
        "bl_tau_winner": winner_tau,
        "bl_risk_aversion_winner": winner_ra,
        "incumbent_tau": incumbent_tau,
        "incumbent_risk_aversion": incumbent_ra,
        "winner_rank_corr": best_corr if recommendation == "update_bl_params" else incumbent_corr,
        "incumbent_rank_corr": incumbent_corr,
        "delta_rank_corr": round(delta, 6),
        "win_threshold": win_threshold,
        "recommendation": recommendation,
        "rows": rows,
    }


def run_bl01_sweep(
    candidate_path: Path = CANDIDATE_PATH,
    tau_grid: list[float] = TAU_GRID,
    ra_grid: list[float] = RISK_AVERSION_GRID,
    n_scenarios: int = N_SCENARIOS,
) -> dict[str, Any]:
    """Run the full BL-01 sweep and write the candidate JSON.

    Returns the candidate dict (same content as written to disk).
    """
    print(f"Running BL-01 sweep: {len(tau_grid)}×{len(ra_grid)} grid × {n_scenarios} scenarios...")
    rows = run_tau_sweep(tau_grid, ra_grid, n_scenarios, BENCHMARKS)
    candidate = select_winner(rows, INCUMBENT_TAU, INCUMBENT_RA, WIN_THRESHOLD_RANK_CORR)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    print(f"Candidate written to {candidate_path}")
    print(f"  Recommendation: {candidate['recommendation']}")
    print(f"  Winner:   tau={candidate['bl_tau_winner']}, risk_aversion={candidate['bl_risk_aversion_winner']}")
    print(f"  Incumbent tau={candidate['incumbent_tau']}, risk_aversion={candidate['incumbent_risk_aversion']}")
    print(f"  Delta rank_corr: {candidate['delta_rank_corr']:+.4f}  (threshold={WIN_THRESHOLD_RANK_CORR})")
    return candidate


if __name__ == "__main__":
    run_bl01_sweep()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && python -m pytest tests/test_bl01_sweep.py -q --tb=short
```

Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && git add results/research/bl01_tau_sweep_eval.py tests/test_bl01_sweep.py && git commit -m "research: BL-01 add tau/risk_aversion sweep harness and tests"
```

---

## Task 2: Run Harness, Write Candidate JSON, Update Config

**Files:**
- Create: `results/research/bl01_tau_candidate.json` (written by harness)
- Modify (conditional): `config/model.py` (only if `recommendation == "update_bl_params"`)

- [ ] **Step 1: Run the harness**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && python results/research/bl01_tau_sweep_eval.py
```

Expected output (approximate — values depend on Monte Carlo draw):
```
Running BL-01 sweep: 5×5 grid × 50 scenarios...
Candidate written to results/research/bl01_tau_candidate.json
  Recommendation: update_bl_params  (or keep_incumbent)
  Winner:   tau=X.XX, risk_aversion=X.X
  Incumbent tau=0.05, risk_aversion=2.5
  Delta rank_corr: +X.XXXX  (threshold=0.05)
```

- [ ] **Step 2: Inspect the candidate JSON**

```bash
cat results/research/bl01_tau_candidate.json
```

Read `recommendation`, `bl_tau_winner`, `bl_risk_aversion_winner`, and `delta_rank_corr`.

**If `recommendation == "update_bl_params"`:** proceed to Step 3.
**If `recommendation == "keep_incumbent"`:** skip Step 3, go directly to Step 4.

- [ ] **Step 3 (conditional): Update `config/model.py` with winning parameters**

Open `config/model.py` and locate lines 69-71:

```python
BL_RISK_AVERSION: float = 2.5           # Moderate risk aversion (1=aggressive, 5=conservative)
BL_TAU: float = 0.05                    # Uncertainty in equilibrium returns (small = trust prior)
BL_VIEW_CONFIDENCE_SCALAR: float = 1.0  # Scales Ω = RMSE² × scalar
```

Replace with the winning values from `bl01_tau_candidate.json`. For example, if winner is tau=0.10, risk_aversion=2.0:

```python
BL_RISK_AVERSION: float = 2.0           # Moderate risk aversion (1=aggressive, 5=conservative)
BL_TAU: float = 0.10                    # Uncertainty in equilibrium returns (BL-01 tuned 2026-04-18)
BL_VIEW_CONFIDENCE_SCALAR: float = 1.0  # Scales Ω = RMSE² × scalar
```

Use the exact numeric values from the candidate JSON — do not guess.

- [ ] **Step 4: Run existing BL tests to verify nothing broke**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && python -m pytest tests/test_black_litterman.py tests/test_bl_fallback_monthly.py tests/test_bl01_sweep.py -q --tb=short
```

Expected: all tests pass (the BL unit tests are config-independent — they pass explicit risk_aversion values, so they tolerate config changes).

If any BL test fails after a config change, inspect the failure. The most likely cause is a test that reads `config.BL_TAU` or `config.BL_RISK_AVERSION` directly; in that case, update the test's expected value to match the new config.

- [ ] **Step 5: Commit**

Stage all changed files (harness output JSON, and conditionally config/model.py):

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && git add results/research/bl01_tau_candidate.json && git commit -m "research: BL-01 tau sweep candidate JSON"
```

If config was updated:
```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && git add config/model.py && git commit -m "research: BL-01 update BL_TAU and BL_RISK_AVERSION to sweep winner"
```

---

## Task 3: Documentation and Closeout

**Files:**
- Create: `docs/closeouts/BL01_CLOSEOUT_AND_HANDOFF.md`
- Modify: `CHANGELOG.md`
- Modify: `ROADMAP.md`
- Modify: `docs/research/backlog.md`

- [ ] **Step 1: Create `docs/closeouts/BL01_CLOSEOUT_AND_HANDOFF.md`**

Read `results/research/bl01_tau_candidate.json` first to fill in the actual outcome values. Then write the closeout with the real numbers substituted below:

```markdown
# BL-01 Closeout And Handoff

Created: 2026-04-18

## Completed Block

`BL-01` closes the Black-Litterman tau/view tuning research cycle. A 5×5 Monte Carlo
sweep (50 scenarios × 25 parameter combinations) evaluated whether updating the
BL prior trust parameter and risk aversion coefficient improves IC-rank-weighted
portfolio quality.

## Final Outcomes

- Harness: `results/research/bl01_tau_sweep_eval.py`
- Candidate: `results/research/bl01_tau_candidate.json`
- Recommendation: <recommendation from JSON>
- Winner: tau=<bl_tau_winner>, risk_aversion=<bl_risk_aversion_winner>
- Incumbent: tau=0.05, risk_aversion=2.5
- Delta rank_corr: <delta_rank_corr> (threshold=0.05)
- config/model.py updated: <yes/no>

## Promotion Boundaries

- Production: no change to live monthly recommendation path
- Shadow: BL diagnostic output in "Portfolio Optimizer Status" section may differ
  if config was updated (shadow-only diagnostic, not decision-driving)
- Config change is research-only shadow tuning — not a production policy change

## Recommended Next Queue

1. `CLS-03` — Path A vs Path B production decision (time-locked on 24 matured months)
2. `CLS-01` — SCHD per-benchmark classifier addition (depends on CLS-03)

## Verification

```bash
python -m pytest tests/test_black_litterman.py tests/test_bl_fallback_monthly.py tests/test_bl01_sweep.py -q --tb=short
```
```

- [ ] **Step 2: Prepend BL-01 to `CHANGELOG.md`**

Read the current top of `CHANGELOG.md` (starts with `## v159 ...`).

Insert this block before `## v159`:

```markdown
## BL-01 (2026-04-18)

- Black-Litterman tau/risk_aversion Monte Carlo sweep: 5×5 grid × 50 scenarios
- Harness: `results/research/bl01_tau_sweep_eval.py`
- Candidate: `results/research/bl01_tau_candidate.json`
- Recommendation: <recommendation> — BL_TAU=<winner_tau>, BL_RISK_AVERSION=<winner_ra>
- config/model.py updated: <yes/no>
- Next: CLS-03 (time-locked), CLS-01

```

Substitute the actual values from `bl01_tau_candidate.json`.

- [ ] **Step 3: Update `ROADMAP.md`**

Locate the section `## Active Research Direction: v159 + BL-01` (added by v159).

Replace it with:

```markdown
## Active Research Direction: v159 + BL-01

- `v159` complete: Firth shadow lane wired; see `docs/closeouts/V159_CLOSEOUT_AND_HANDOFF.md`
- `BL-01` complete: BL tau/risk_aversion sweep; see `docs/closeouts/BL01_CLOSEOUT_AND_HANDOFF.md`
- Next: `CLS-03` (time-locked on 24 matured months), `CLS-01` (depends on CLS-03)
```

- [ ] **Step 4: Mark BL-01 complete in `docs/research/backlog.md`**

Locate the BL-01 entry (around line 36):

```
### BL-01 â€" Black-Litterman Tau/View Tuning
**Status:** open
```

Replace with:

```markdown
### BL-01 — Black-Litterman Tau/View Tuning
**Status:** complete
**Priority:** medium
**Rationale:** The decision layer still uses untuned BL priors even though the regression and classifier research stack has moved materially since the original defaults.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** recommendation accuracy up, policy uplift up modestly
**Last touched:** 2026-04-18
**Outcome:** <recommendation from bl01_tau_candidate.json — e.g., "keep_incumbent: delta +0.024 below threshold 0.05">
```

Also update the ranked queue at the top. Locate:

```
2. `BL-01` - Black-Litterman tau/view tuning — open
```

Replace with:

```
2. `BL-01` - Black-Litterman tau/view tuning — **complete (2026-04-18)**
```

- [ ] **Step 5: Run full test suite one final time**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && python -m pytest tests/test_black_litterman.py tests/test_bl_fallback_monthly.py tests/test_bl01_sweep.py -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 6: Commit documentation**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && git add docs/closeouts/BL01_CLOSEOUT_AND_HANDOFF.md CHANGELOG.md ROADMAP.md docs/research/backlog.md && git commit -m "docs: BL-01 closeout, CHANGELOG, ROADMAP, backlog update"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| Monte Carlo sweep over tau × risk_aversion | Task 1 (run_tau_sweep) |
| 50 seeded scenarios, 60-month synthetic returns | Task 1 (_make_scenario) |
| 5×5 grid (tau ∈ {0.01, 0.025, 0.05, 0.10, 0.20}, RA ∈ {1.5, 2.0, 2.5, 3.0, 4.0}) | Task 1 (constants) |
| Score by IC-rank correlation (Spearman, numpy-only) | Task 1 (_rank_corr, _compute_ic_rank_correlation) |
| Select winner if delta ≥ 0.05, else keep incumbent | Task 1 (select_winner) |
| Write bl01_tau_candidate.json | Task 1 (run_bl01_sweep) + Task 2 Step 1 |
| Update config/model.py only if winner recommended | Task 2 Step 3 (conditional) |
| Existing BL tests still pass | Task 2 Step 4 |
| 10 unit tests for sweep helpers | Task 1 (test file) |
| Closeout, CHANGELOG, ROADMAP, backlog | Task 3 |
| No live recommendation path changes | All tasks (BL shadow-only) |

### Placeholder scan

- Task 3 Steps 1–2 contain `<recommendation from JSON>` references — these are data-driven substitutions from the harness output, not planning placeholders. ✓
- No "TBD", "TODO", "implement later", or "similar to Task N" patterns. ✓

### Type consistency

- `_make_signal(...) -> EnsembleWFOResult` used in tests as `_make_signal(bm, ic)` — matches signature `(benchmark, mean_ic, ...)` ✓
- `_make_scenario(...) -> tuple[pd.DataFrame, dict[str, EnsembleWFOResult]]` ✓
- `_rank_corr(x: np.ndarray, y: np.ndarray) -> float` ✓
- `_compute_ic_rank_correlation(weights: dict[str, float], signals: dict[str, EnsembleWFOResult]) -> float` ✓
- `run_tau_sweep(...) -> list[dict[str, Any]]` — row keys (tau, risk_aversion, mean_rank_corr, fallback_rate) used in `select_winner` match ✓
- `select_winner(...) -> dict[str, Any]` — output keys checked in `test_select_winner_output_keys` ✓

### Known implementation note

`run_tau_sweep` temporarily patches `config.BL_TAU` to override the global config during each `build_bl_weights` call. This is the minimal approach since `build_bl_weights` reads `config.BL_TAU` internally and does not accept `tau` as a parameter. The patch is wrapped in a try/finally to ensure restoration even if an exception occurs.
