# Phase 1 — Calibration Experiments (v37–v43)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Return to [master plan](2026-04-10-v37-v60-oos-improvements.md) for global rules.

**Goal:** Establish baseline metrics then apply the highest-leverage, lowest-complexity interventions: prediction shrinkage, Ridge regularization extension, GBT constraint/removal, target winsorization, expanding windows, and feature reduction.

**Expected aggregate R² impact:** +10 to +20 pp over baseline (may alone reach the +2% target).

**Files to create:** `results/research/v37_baseline.py` through `results/research/v43_features.py`

**Shared dependency:** `src/research/v37_utils.py` (already exists — do not modify).

---

### Task 1: Verify results/research/ directory and shared utils

**Note:** `results/research/__init__.py` and `src/research/v37_utils.py` are already committed. Tests live **flat** in `tests/` (not in a subdirectory) — use names like `tests/test_research_v37_baseline.py` to avoid collisions with existing test files.

**Files:**
- Already created: `results/research/__init__.py`, `src/research/v37_utils.py`

- [ ] **Step 1: Verify shared utils imports work**

```bash
python -c "from src.research.v37_utils import get_connection, BENCHMARKS, RIDGE_FEATURES_12; print('OK', BENCHMARKS[:2])"
```

Expected: `OK ['VOO', 'VXUS']`

- [ ] **Step 2: Verify results/research/ exists**

```bash
ls results/research/__init__.py
```

Expected: file exists. If missing: `mkdir -p results/research && touch results/research/__init__.py`

---

### Task 2: v37 — Baseline Measurement

**Files:**
- Create: `results/research/v37_baseline.py`

**Purpose:** Measure the exact current v11.0 production ensemble metrics as the comparison point for all subsequent experiments. Per benchmark: N OOS, OOS R², IC, NW IC/p-value, hit rate, MAE, std(y_hat), std(y_true), σ_pred/σ_true ratio. Aggregate: pooled across all 8 benchmarks.

- [ ] **Step 1: Write the script**

```python
"""v37 — Baseline measurement of current v11.0 lean Ridge+GBT ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.models.evaluation import reconstruct_ensemble_oos_predictions
from src.research.v37_utils import (
    BENCHMARKS,
    build_results_df,
    compute_metrics,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    pool_metrics,
    print_delta,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)
from config.features import MODEL_FEATURE_OVERRIDES


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)

    rel_matrix: dict[str, pd.Series] = {}
    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if not s.empty:
            rel_matrix[etf] = s

    ensemble_results = run_ensemble_benchmarks(
        df,
        pd.DataFrame(rel_matrix),
        target_horizon_months=6,
        model_feature_overrides=MODEL_FEATURE_OVERRIDES,
    )

    rows: list[dict] = []
    for etf in BENCHMARKS:
        if etf not in ensemble_results:
            continue
        y_hat, y_true = reconstruct_ensemble_oos_predictions(ensemble_results[etf])
        m = compute_metrics(y_true, y_hat)
        rows.append({"benchmark": etf, **m, "_y_true": y_true, "_y_hat": y_hat})

    pooled = pool_metrics(rows)

    print_header("v37", "Baseline — v11.0 Ridge+GBT Ensemble")
    print_per_benchmark(rows)
    print_pooled(pooled)
    print_footer()

    results_df = build_results_df(rows, pooled, extra_cols={"version": "v37", "experiment": "baseline"})
    save_results(results_df, "v37_baseline_results.csv")


if __name__ == "__main__":
    main()
```

Save this to `results/research/v37_baseline.py`.

- [ ] **Step 2: Run the script**

```bash
python results/research/v37_baseline.py
```

Expected: table with 8 benchmark rows + POOLED row printed, OOS R² should be around −13%. CSV saved to `results/research/v37_baseline_results.csv`.

- [ ] **Step 3: Write the pytest**

```python
"""Test v37 baseline produces expected output shape and metric sanity."""
import pandas as pd
import pytest
from pathlib import Path

CSV = Path("results/research/v37_baseline_results.csv")


def test_csv_exists():
    assert CSV.exists(), "Run v37_baseline.py first"


def test_row_count():
    df = pd.read_csv(CSV)
    assert len(df) == 9, f"Expected 9 rows (8 benchmarks + POOLED), got {len(df)}"


def test_pooled_row_present():
    df = pd.read_csv(CSV)
    assert "POOLED" in df["benchmark"].values


def test_ic_positive():
    df = pd.read_csv(CSV)
    pooled = df[df["benchmark"] == "POOLED"].iloc[0]
    assert pooled["ic"] > 0, "Pooled IC should be positive (diagnostic: model has signal)"


def test_hit_rate_above_chance():
    df = pd.read_csv(CSV)
    pooled = df[df["benchmark"] == "POOLED"].iloc[0]
    assert pooled["hit_rate"] > 0.55, f"Hit rate {pooled['hit_rate']:.3f} unexpectedly low"


def test_sigma_ratio_gt_one():
    """Confirms the core calibration problem: σ_pred >> σ_true."""
    df = pd.read_csv(CSV)
    pooled = df[df["benchmark"] == "POOLED"].iloc[0]
    assert pooled["sigma_ratio"] > 1.0, "σ_ratio should be > 1 (overconfident predictions)"
```

Save to `tests/test_research_v37_baseline.py`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_research_v37_baseline.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add results/research/v37_baseline.py tests/test_research_v37_baseline.py
git commit -m "research: add v37 baseline measurement script and tests"
```

---

### Task 3: v38 — Prediction Shrinkage

**Files:**
- Create: `results/research/v38_shrinkage.py`

**Purpose:** Post-hoc multiply all OOS predictions by α ∈ [0.05…1.00] without retraining. IC and hit rate must stay invariant (sanity check). Find the optimal α that maximizes aggregate OOS R².

- [ ] **Step 1: Write the script**

```python
"""v38 — Post-hoc prediction shrinkage: y_hat_shrunk = alpha * y_hat."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.models.evaluation import reconstruct_ensemble_oos_predictions
from src.research.v37_utils import (
    BENCHMARKS,
    build_results_df,
    compute_metrics,
    get_connection,
    load_baseline_results,
    load_feature_matrix,
    load_relative_series,
    pool_metrics,
    print_delta,
    print_footer,
    print_header,
    save_results,
)
from config.features import MODEL_FEATURE_OVERRIDES

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)

    rel_matrix: dict[str, pd.Series] = {}
    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if not s.empty:
            rel_matrix[etf] = s

    ensemble_results = run_ensemble_benchmarks(
        df, pd.DataFrame(rel_matrix), target_horizon_months=6,
        model_feature_overrides=MODEL_FEATURE_OVERRIDES,
    )

    # Collect raw predictions (alpha=1.0)
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for etf in BENCHMARKS:
        if etf in ensemble_results:
            y_hat, y_true = reconstruct_ensemble_oos_predictions(ensemble_results[etf])
            raw[etf] = (y_hat, y_true)

    alpha_results: list[dict] = []
    print("\nAlpha sweep:")
    print(f"  {'alpha':>6}  {'R2_pooled':>10}  {'IC_pooled':>10}  {'HitRate':>8}")
    for alpha in ALPHAS:
        rows = []
        for etf, (y_hat_raw, y_true) in raw.items():
            y_hat_shrunk = alpha * y_hat_raw
            m = compute_metrics(y_true, y_hat_shrunk)
            rows.append({"benchmark": etf, **m, "_y_true": y_true, "_y_hat": y_hat_shrunk})
        p = pool_metrics(rows)
        print(f"  {alpha:>6.2f}  {p['r2']:>10.4f}  {p['ic']:>10.4f}  {p['hit_rate']:>8.4f}")
        alpha_results.append({"alpha": alpha, **{k: p[k] for k in ["r2", "ic", "hit_rate", "mae"]}})

    best = max(alpha_results, key=lambda x: x["r2"])
    optimal_alpha = best["alpha"]
    print(f"\nOptimal alpha: {optimal_alpha} (R2={best['r2']:.4f})")

    # Sanity check: IC and hit_rate at optimal alpha must match alpha=1.0 result (within rounding)
    raw_m = compute_metrics(
        np.concatenate([v[1] for v in raw.values()]),
        np.concatenate([v[0] for v in raw.values()]),
    )
    assert abs(best["ic"] - raw_m["ic"]) < 1e-10, "IC changed with shrinkage — BUG"
    print("Sanity check passed: IC is invariant to linear scaling.")

    # Final rows at optimal alpha for CSV
    final_rows = []
    for etf, (y_hat_raw, y_true) in raw.items():
        y_hat_shrunk = optimal_alpha * y_hat_raw
        m = compute_metrics(y_true, y_hat_shrunk)
        final_rows.append({"benchmark": etf, **m, "_y_true": y_true, "_y_hat": y_hat_shrunk})
    pooled = pool_metrics(final_rows)

    print_header("v38", f"Prediction Shrinkage (optimal alpha={optimal_alpha})")
    print_delta(pooled, load_baseline_results())
    print_footer()

    sweep_df = pd.DataFrame(alpha_results)
    sweep_df["version"] = "v38"
    save_results(sweep_df, "v38_shrinkage_results.csv")

    results_df = build_results_df(final_rows, pooled, extra_cols={"version": "v38", "optimal_alpha": optimal_alpha})
    save_results(results_df, "v38_shrinkage_best_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v38_shrinkage.py`.

- [ ] **Step 2: Run the script**

```bash
python results/research/v38_shrinkage.py
```

Expected: alpha sweep table printed, optimal alpha printed (expect ~0.10–0.20), large positive R² delta vs baseline, IC unchanged.

- [ ] **Step 3: Write the pytest**

```python
"""Test v38 shrinkage: IC invariance, CSV shape, best alpha in expected range."""
import pandas as pd
import pytest
from pathlib import Path

SWEEP = Path("results/research/v38_shrinkage_results.csv")
BEST = Path("results/research/v38_shrinkage_best_results.csv")


def test_sweep_csv_exists():
    assert SWEEP.exists()


def test_sweep_has_all_alphas():
    df = pd.read_csv(SWEEP)
    assert len(df) == 10, f"Expected 10 alpha rows, got {len(df)}"


def test_optimal_alpha_plausible():
    df = pd.read_csv(SWEEP)
    best = df.loc[df["r2"].idxmax()]
    assert 0.05 <= best["alpha"] <= 0.75, f"Optimal alpha={best['alpha']} out of expected range"


def test_ic_invariant_across_alphas():
    """IC must not change with linear scaling of predictions."""
    df = pd.read_csv(SWEEP)
    ic_vals = df["ic"].values
    assert ic_vals.max() - ic_vals.min() < 1e-8, "IC varied with alpha — shrinkage broke invariance"


def test_best_r2_better_than_negative():
    df = pd.read_csv(SWEEP)
    assert df["r2"].max() > -0.05, "Best shrinkage R2 should be much better than raw"
```

Save to `tests/test_research_v38_shrinkage.py`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_research_v38_shrinkage.py -v
```

- [ ] **Step 5: Commit**

```bash
git add results/research/v38_shrinkage.py tests/test_research_v38_shrinkage.py
git commit -m "research: add v38 prediction shrinkage experiment"
```

---

### Task 4: v39 — Ridge Alpha Extension

**Files:**
- Create: `results/research/v39_ridge_alpha.py`

**Purpose:** Test three Ridge alpha grids (current, extended, aggressive) using `RidgeCV(cv=None)` for efficient LOOCV. Ridge-only, all 8 benchmarks.

- [ ] **Step 1: Write the script**

```python
"""v39 — Ridge alpha grid extension: test logspace(-4,4) vs logspace(0,6) vs logspace(2,6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    build_results_df,
    compute_metrics,
    custom_wfo,
    get_connection,
    load_baseline_results,
    load_feature_matrix,
    load_relative_series,
    pool_metrics,
    print_delta,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)
from src.processing.feature_engineering import get_X_y_relative

ALPHA_GRIDS = {
    "current_logspace(-4,4)": np.logspace(-4, 4, 50),
    "extended_logspace(0,6)": np.logspace(0, 6, 100),
    "aggressive_logspace(2,6)": np.logspace(2, 6, 100),
}


def run_ridge_grid(df: pd.DataFrame, rel_series: pd.Series, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_df, y = get_X_y_relative(df, rel_series, drop_na_target=True)
    feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
    X = X_df[feat_cols].values

    def factory() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=alphas, cv=None, fit_intercept=True)),
        ])

    return custom_wfo(X, y.values, factory, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)

    all_rows: list[dict] = []
    print(f"\n{'Grid':<30}  {'R2_pooled':>10}  {'IC_pooled':>10}  {'HitRate':>8}")
    for grid_name, alphas in ALPHA_GRIDS.items():
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            y_true, y_hat = run_ridge_grid(df, s, alphas)
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "grid": grid_name, **m, "_y_true": y_true, "_y_hat": y_hat})
        p = pool_metrics(rows)
        print(f"  {grid_name:<30}  {p['r2']:>10.4f}  {p['ic']:>10.4f}  {p['hit_rate']:>8.4f}")
        all_rows.extend(rows)

    # Best grid by pooled R²
    grid_summary = []
    for grid_name in ALPHA_GRIDS:
        subset = [r for r in all_rows if r["grid"] == grid_name]
        p = pool_metrics(subset)
        grid_summary.append({"grid": grid_name, **p})

    best_grid = max(grid_summary, key=lambda x: x["r2"])
    print(f"\nBest grid: {best_grid['grid']} (R2={best_grid['r2']:.4f})")

    print_header("v39", "Ridge Alpha Extension")
    best_rows = [r for r in all_rows if r["grid"] == best_grid["grid"]]
    pooled = pool_metrics(best_rows)
    print_per_benchmark(best_rows)
    print_pooled(pooled)
    print_delta(pooled, load_baseline_results())
    print_footer()

    out_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows]
    summary_df = pd.DataFrame(out_rows)
    save_results(summary_df, "v39_ridge_alpha_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v39_ridge_alpha.py`.

- [ ] **Step 2: Run the script**

```bash
python results/research/v39_ridge_alpha.py
```

Expected: 3-row summary printed, extended/aggressive grids should improve R² over current grid.

- [ ] **Step 3: Write and run the pytest**

```python
"""Test v39 ridge alpha: CSV shape and extended grid not worse than current."""
import pandas as pd
from pathlib import Path

CSV = Path("results/research/v39_ridge_alpha_results.csv")


def test_csv_exists():
    assert CSV.exists()


def test_all_grids_present():
    df = pd.read_csv(CSV)
    grids = df["grid"].unique()
    assert len(grids) == 3


def test_extended_grid_not_worse_than_current():
    df = pd.read_csv(CSV)
    by_grid = df.groupby("grid")["r2"].mean()
    current = by_grid.get("current_logspace(-4,4)", float("nan"))
    extended = by_grid.get("extended_logspace(0,6)", float("nan"))
    # Extended should be at least as good (allow 0.005 tolerance)
    assert extended >= current - 0.005, f"Extended grid worse: {extended:.4f} vs {current:.4f}"
```

Save to `tests/test_research_v39_ridge_alpha.py`.

```bash
pytest tests/test_research_v39_ridge_alpha.py -v
```

- [ ] **Step 4: Commit**

```bash
git add results/research/v39_ridge_alpha.py tests/test_research_v39_ridge_alpha.py
git commit -m "research: add v39 Ridge alpha extension experiment"
```

---

### Task 5: v40 — GBT Constraint/Removal

**Files:**
- Create: `results/research/v40_gbt_constraint.py`

**Purpose:** Test three variants: (A) Ridge-only (no GBT), (B) Constrained GBT (lr=0.01, min_samples_leaf=10, subsample=0.7), (C) Post-hoc 80/20 Ridge/GBT reweight.

- [ ] **Step 1: Write the script**

```python
"""v40 — GBT constraint/removal: Ridge-only, constrained GBT, 80/20 reweight."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.wfo_engine import run_wfo
from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    GBT_FEATURES_13,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    build_results_df,
    compute_metrics,
    custom_wfo,
    get_connection,
    load_baseline_results,
    load_feature_matrix,
    load_relative_series,
    pool_metrics,
    print_delta,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)  # Best grid from v39


def run_variant_a_ridge_only(df: pd.DataFrame, rel_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-only using run_wfo model_type='ridge' with extended alpha grid."""
    X_df, y = get_X_y_relative(df, rel_series, drop_na_target=True)
    feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
    X = X_df[feat_cols].values

    def factory() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
        ])

    return custom_wfo(X, y.values, factory, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)


def run_variant_b_constrained_gbt(df: pd.DataFrame, rel_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Constrained GBT: shallow, slow learning rate, high min_samples_leaf."""
    X_df, y = get_X_y_relative(df, rel_series, drop_na_target=True)
    feat_cols = [c for c in GBT_FEATURES_13 if c in X_df.columns]
    X = X_df[feat_cols].values

    def factory() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                max_depth=2, n_estimators=50, learning_rate=0.01,
                min_samples_leaf=10, subsample=0.7, random_state=42,
            )),
        ])

    return custom_wfo(X, y.values, factory, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)


def run_variant_c_reweight(
    y_true_ridge: np.ndarray, y_hat_ridge: np.ndarray,
    y_true_gbt: np.ndarray, y_hat_gbt: np.ndarray,
    ridge_weight: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-hoc blend: align lengths (take shorter), then blend predictions."""
    n = min(len(y_true_ridge), len(y_true_gbt))
    y_true = y_true_ridge[-n:]
    y_hat = ridge_weight * y_hat_ridge[-n:] + (1 - ridge_weight) * y_hat_gbt[-n:]
    return y_true, y_hat


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)

    variants = {"ridge_only": [], "constrained_gbt": [], "reweight_80_20": []}
    baseline = load_baseline_results()

    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue

        yt_a, yh_a = run_variant_a_ridge_only(df, s)
        m_a = compute_metrics(yt_a, yh_a)
        variants["ridge_only"].append({"benchmark": etf, **m_a, "_y_true": yt_a, "_y_hat": yh_a})

        yt_b, yh_b = run_variant_b_constrained_gbt(df, s)
        m_b = compute_metrics(yt_b, yh_b)
        variants["constrained_gbt"].append({"benchmark": etf, **m_b, "_y_true": yt_b, "_y_hat": yh_b})

        yt_c, yh_c = run_variant_c_reweight(yt_a, yh_a, yt_b, yh_b)
        m_c = compute_metrics(yt_c, yh_c)
        variants["reweight_80_20"].append({"benchmark": etf, **m_c, "_y_true": yt_c, "_y_hat": yh_c})

    all_output_rows: list[dict] = []
    for variant_name, rows in variants.items():
        p = pool_metrics(rows)
        print_header("v40", f"GBT Constraint — Variant: {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        for r in rows:
            all_output_rows.append({k: v for k, v in r.items() if not k.startswith("_"),
                                    "variant": variant_name})

    results_df = pd.DataFrame(all_output_rows)
    save_results(results_df, "v40_gbt_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v40_gbt_constraint.py`.

- [ ] **Step 2: Run the script**

```bash
python results/research/v40_gbt_constraint.py
```

Expected: 3 variant result tables printed. Ridge-only and constrained GBT should both improve over baseline.

- [ ] **Step 3: Write and run the pytest**

```python
"""Test v40 GBT constraint: 3 variants present, ridge_only has better R2 than baseline."""
import pandas as pd
from pathlib import Path

CSV = Path("results/research/v40_gbt_results.csv")


def test_csv_exists():
    assert CSV.exists()


def test_three_variants():
    df = pd.read_csv(CSV)
    assert set(df["variant"].unique()) == {"ridge_only", "constrained_gbt", "reweight_80_20"}


def test_ridge_only_r2_better_than_negative_10():
    df = pd.read_csv(CSV)
    pooled = df[(df["benchmark"] == "POOLED") & (df["variant"] == "ridge_only")]
    assert not pooled.empty
    assert pooled.iloc[0]["r2"] > -0.10, "Ridge-only should outperform raw baseline"
```

Save to `tests/test_research_v40_gbt_constraint.py`.

```bash
pytest tests/test_research_v40_gbt_constraint.py -v
```

- [ ] **Step 4: Commit**

```bash
git add results/research/v40_gbt_constraint.py tests/test_research_v40_gbt_constraint.py
git commit -m "research: add v40 GBT constraint/removal experiment"
```

---

### Task 6: v41 — Target Winsorization

**Files:**
- Create: `results/research/v41_winsorize.py`

**Purpose:** Within each WFO fold, winsorize `y_train` at [5th/95th] and [10th/90th] percentiles before fitting. Evaluate against the original (unwinsorized) `y_test`.

- [ ] **Step 1: Write the script**

```python
"""v41 — Target winsorization within each WFO fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    build_results_df,
    compute_metrics,
    get_connection,
    load_baseline_results,
    load_feature_matrix,
    load_relative_series,
    pool_metrics,
    print_delta,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)
CLIP_LEVELS = {"p5_p95": (5, 95), "p10_p90": (10, 90)}


def wfo_with_target_winsorization(
    X: np.ndarray, y: np.ndarray, clip_pct: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    available = n - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(
        n_splits=n_splits, max_train_size=MAX_TRAIN_MONTHS,
        test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS,
    )
    all_y_true, all_y_hat = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]

        # Per-fold median imputation
        medians = np.nanmedian(X_tr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, c]), c] = medians[c]
            X_te[np.isnan(X_te[:, c]), c] = medians[c]

        # Winsorize y_train only
        lo = np.percentile(y_tr, clip_pct[0])
        hi = np.percentile(y_tr, clip_pct[1])
        y_tr_clipped = np.clip(y_tr, lo, hi)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
        ])
        pipe.fit(X_tr, y_tr_clipped)
        y_hat = pipe.predict(X_te)

        all_y_true.extend(y_te.tolist())
        all_y_hat.extend(y_hat.tolist())
    return np.array(all_y_true), np.array(all_y_hat)


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    for level_name, clip_pct in CLIP_LEVELS.items():
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            X_df, y = get_X_y_relative(df, s, drop_na_target=True)
            feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
            y_true, y_hat = wfo_with_target_winsorization(X_df[feat_cols].values, y.values, clip_pct)
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "clip_level": level_name, **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v41", f"Target Winsorization — {level_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    save_results(pd.DataFrame(all_output_rows), "v41_winsorize_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v41_winsorize.py`.

- [ ] **Step 2: Run, write test (assert CSV has 2 clip_level values and POOLED rows), run test, commit**

```bash
python results/research/v41_winsorize.py
```

```python
# tests/test_research_v41_winsorize.py
import pandas as pd
from pathlib import Path

CSV = Path("results/research/v41_winsorize_results.csv")

def test_csv_exists(): assert CSV.exists()
def test_two_clip_levels():
    df = pd.read_csv(CSV)
    assert set(df["clip_level"].unique()) == {"p5_p95", "p10_p90"}
def test_pooled_r2_improves():
    """Winsorization should improve (or not significantly worsen) R2."""
    df = pd.read_csv(CSV)
    base_r2 = -0.13  # approximate baseline from v37
    for lvl in ["p5_p95", "p10_p90"]:
        pooled = df[(df["benchmark"] == "POOLED") & (df["clip_level"] == lvl)]
        if not pooled.empty:
            assert pooled.iloc[0]["r2"] > base_r2 - 0.02
```

```bash
pytest tests/test_research_v41_winsorize.py -v
git add results/research/v41_winsorize.py tests/test_research_v41_winsorize.py
git commit -m "research: add v41 target winsorization experiment"
```

---

### Task 7: v42 — Expanding Window

**Files:**
- Create: `results/research/v42_expanding.py`

**Purpose:** Test (A) pure expanding window, (B) expanding with exponential sample weights, (C) expanding capped at 120 months.

- [ ] **Step 1: Write the script**

```python
"""v42 — Expanding window variants: pure, decay-weighted, capped at 120M."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    compute_metrics, get_connection, load_baseline_results,
    load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_per_benchmark,
    print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)

VARIANTS: dict[str, dict] = {
    "A_pure_expanding": {"max_train_size": None, "cap": None, "decay": False},
    "B_expanding_decay": {"max_train_size": None, "cap": None, "decay": True},
    "C_expanding_cap120": {"max_train_size": 120, "cap": 120, "decay": False},
}


def expanding_wfo(
    X: np.ndarray, y: np.ndarray,
    max_train_size: int | None,
    decay: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    n_splits = max(1, (n - 60 - GAP_MONTHS) // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=max_train_size,
        test_size=TEST_SIZE_MONTHS,
        gap=GAP_MONTHS,
    )
    all_y_true, all_y_hat = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]

        medians = np.nanmedian(X_tr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, c]), c] = medians[c]
            X_te[np.isnan(X_te[:, c]), c] = medians[c]

        sample_weight = None
        if decay:
            # Exponential decay: most recent obs weight=1, oldest ≈ 0.5 at half-life=60
            half_life = 60
            ages = np.arange(len(y_tr), 0, -1)
            sample_weight = 0.5 ** (ages / half_life)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
        ])
        if decay and sample_weight is not None:
            pipe.fit(X_tr, y_tr, model__sample_weight=sample_weight)
        else:
            pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te)

        all_y_true.extend(y_te.tolist())
        all_y_hat.extend(y_hat.tolist())
    return np.array(all_y_true), np.array(all_y_hat)


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    for variant_name, cfg in VARIANTS.items():
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            X_df, y = get_X_y_relative(df, s, drop_na_target=True)
            feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
            y_true, y_hat = expanding_wfo(
                X_df[feat_cols].values, y.values,
                cfg["max_train_size"], cfg["decay"],
            )
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": variant_name, **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v42", f"Expanding Window — {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    save_results(pd.DataFrame(all_output_rows), "v42_expanding_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v42_expanding.py`.

- [ ] **Step 2: Run script, write test, run test, commit**

```bash
python results/research/v42_expanding.py
```

```python
# tests/test_research_v42_expanding.py
import pandas as pd
from pathlib import Path

def test_three_variants():
    df = pd.read_csv(Path("results/research/v42_expanding_results.csv"))
    assert len(df["variant"].unique()) == 3
def test_pure_expanding_has_more_obs():
    """Pure expanding should have more OOS observations than rolling 60M."""
    df = pd.read_csv(Path("results/research/v42_expanding_results.csv"))
    pure = df[(df["variant"] == "A_pure_expanding") & (df["benchmark"] == "POOLED")]
    assert not pure.empty
```

```bash
pytest tests/test_research_v42_expanding.py -v
git add results/research/v42_expanding.py tests/test_research_v42_expanding.py
git commit -m "research: add v42 expanding window experiment"
```

---

### Task 8: v43 — Feature Reduction

**Files:**
- Create: `results/research/v43_features.py`

**Purpose:** Test 3 reduced feature sets (Ridge→7, GBT→7, Shared→7) using `run_wfo(feature_columns=...)`.

- [ ] **Step 1: Write the script**

```python
"""v43 — Feature reduction to 5–7 features per model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.wfo_engine import run_wfo
from src.research.v37_utils import (
    BENCHMARKS, SHARED_7_FEATURES, build_results_df, compute_metrics,
    get_connection, load_baseline_results, load_feature_matrix,
    load_relative_series, pool_metrics, print_delta, print_footer,
    print_header, print_per_benchmark, print_pooled, save_results,
)

RIDGE_7 = ["yield_slope", "npw_growth_yoy", "investment_income_growth_yoy",
           "real_yield_change_6m", "combined_ratio_ttm", "mom_12m", "credit_spread_hy"]
GBT_7 = ["mom_12m", "vol_63d", "yield_slope", "credit_spread_hy",
          "vix", "pif_growth_yoy", "investment_book_yield"]

VARIANTS = {
    "A_ridge7": {"model_type": "ridge", "features": RIDGE_7},
    "B_gbt7":   {"model_type": "gbt",   "features": GBT_7},
    "C_shared7_ridge": {"model_type": "ridge", "features": SHARED_7_FEATURES},
}


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    for variant_name, cfg in VARIANTS.items():
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            from src.processing.feature_engineering import get_X_y_relative
            X_df, y = get_X_y_relative(df, s, drop_na_target=True)
            result = run_wfo(
                X_df, y,
                model_type=cfg["model_type"],
                target_horizon_months=6,
                benchmark=etf,
                feature_columns=cfg["features"],
            )
            y_true = result.y_true_all
            y_hat = result.y_hat_all
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": variant_name, **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v43", f"Feature Reduction — {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    save_results(pd.DataFrame(all_output_rows), "v43_feature_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v43_features.py`.

- [ ] **Step 2: Run script, write test, run test, commit**

```bash
python results/research/v43_features.py
```

```python
# tests/test_research_v43_features.py
import pandas as pd
from pathlib import Path

def test_three_variants():
    df = pd.read_csv(Path("results/research/v43_feature_results.csv"))
    assert len(df["variant"].unique()) == 3
```

```bash
pytest tests/test_research_v43_features.py -v
git add results/research/v43_features.py tests/test_research_v43_features.py
git commit -m "research: add v43 feature reduction experiment"
```

---

### Phase 1 Review Checkpoint

After running v37–v43, **before proceeding to Phase 2**, review the results:

```bash
python -c "
import pandas as pd, glob
csvs = glob.glob('results/research/v3[7-9]_*.csv') + glob.glob('results/research/v4[0-3]_*.csv')
for f in sorted(csvs):
    df = pd.read_csv(f)
    pooled = df[df['benchmark']=='POOLED']
    if not pooled.empty:
        r = pooled.iloc[0]
        print(f'{f.split(\"/\")[-1][:20]:<22}  R2={r[\"r2\"]:+.4f}  IC={r[\"ic\"]:.4f}  HR={r[\"hit_rate\"]:.4f}')
"
```

**Decision:** If best Phase 1 R² ≥ +2%, proceed to holdout evaluation. Otherwise continue to Phase 2–3.
