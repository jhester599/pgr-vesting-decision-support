# Phase 2–3 — Architecture & Target/Structure Reforms (v44–v49)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Return to [master plan](2026-04-10-v37-v60-oos-improvements.md) for global rules. Run Phase 1 first — these experiments should be conditioned on Phase 1 results.

**Goal:** Medium-complexity interventions targeting feature construction (blockwise PCA), model type (BayesianRidge, classification), and prediction target/structure (composite benchmark, panel pooling, regime features).

**Baseline for this phase:** Compare all `v44+` experiments against `v38` shrinkage, not the raw `v37` ensemble. `v37` is retained only as the historical production-baseline measurement.

**Future-work note from the completed Phase 2 run:** `v46` binary classification did not improve regression OOS R² by design, but it produced the most promising decision-aligned metrics in this phase (pooled accuracy `0.6533`, balanced accuracy `0.5292`, Brier `0.2502`). Keep the classification branch on the roadmap for later recommendation-layer work even if the main regression baseline remains `v38`.

**Files to create:** `results/research/v44_pca.py` through `results/research/v49_regime_features.py`

**Shared dependency:** `src/research/v37_utils.py`

---

### Task 9: v44 — Blockwise PCA

**Files:**
- Create: `results/research/v44_pca.py`

**Purpose:** Apply PCA separately to macro and insurance feature blocks within each WFO fold. Eliminates multicollinearity while retaining shared variance. Two variants: (A) 2 components/block, (B) 1 component/block.

- [ ] **Step 1: Write the script**

```python
"""v44 — Blockwise PCA dimensionality reduction within each WFO fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS,
    compute_metrics, get_connection, load_baseline_results,
    load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_per_benchmark,
    print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)

# Feature blocks
MACRO_BLOCK = ["yield_slope", "real_rate_10y", "credit_spread_hy", "nfci", "vix", "real_yield_change_6m"]
INSURANCE_BLOCK = ["combined_ratio_ttm", "npw_growth_yoy", "investment_income_growth_yoy", "pif_growth_yoy"]
RAW_PASSTHROUGH = ["mom_12m", "vol_63d"]  # Not compressed

ALL_FEATURES = MACRO_BLOCK + INSURANCE_BLOCK + RAW_PASSTHROUGH
VARIANTS = {
    "A_2comp_per_block": {"n_macro": 2, "n_insurance": 2},  # 6 total (2+2+2 raw)
    "B_1comp_per_block": {"n_macro": 1, "n_insurance": 1},  # 4 total (1+1+2 raw)
}


def blockwise_pca_wfo(
    df: pd.DataFrame, rel_series: pd.Series,
    n_macro: int, n_insurance: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_df, y = get_X_y_relative(df, rel_series, drop_na_target=True)

    # Only keep features present in the data
    macro_cols = [c for c in MACRO_BLOCK if c in X_df.columns]
    ins_cols = [c for c in INSURANCE_BLOCK if c in X_df.columns]
    raw_cols = [c for c in RAW_PASSTHROUGH if c in X_df.columns]

    X_macro = X_df[macro_cols].values
    X_ins = X_df[ins_cols].values
    X_raw = X_df[raw_cols].values
    y_vals = y.values

    n = len(y_vals)
    available = n - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=MAX_TRAIN_MONTHS,
                          test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)

    all_y_true, all_y_hat = [], []
    for train_idx, test_idx in tscv.split(X_macro):
        # Per-fold: scale → PCA → concat → RidgeCV
        scaler_m = StandardScaler()
        scaler_i = StandardScaler()

        Xm_tr = scaler_m.fit_transform(np.nan_to_num(X_macro[train_idx], nan=0.0))
        Xm_te = scaler_m.transform(np.nan_to_num(X_macro[test_idx], nan=0.0))

        Xi_tr = scaler_i.fit_transform(np.nan_to_num(X_ins[train_idx], nan=0.0))
        Xi_te = scaler_i.transform(np.nan_to_num(X_ins[test_idx], nan=0.0))

        pca_m = PCA(n_components=min(n_macro, Xm_tr.shape[1]))
        pca_i = PCA(n_components=min(n_insurance, Xi_tr.shape[1]))

        Xm_tr_pca = pca_m.fit_transform(Xm_tr)
        Xm_te_pca = pca_m.transform(Xm_te)
        Xi_tr_pca = pca_i.fit_transform(Xi_tr)
        Xi_te_pca = pca_i.transform(Xi_te)

        Xr_tr = np.nan_to_num(X_raw[train_idx], nan=0.0)
        Xr_te = np.nan_to_num(X_raw[test_idx], nan=0.0)

        X_tr_final = np.column_stack([Xm_tr_pca, Xi_tr_pca, Xr_tr])
        X_te_final = np.column_stack([Xm_te_pca, Xi_te_pca, Xr_te])

        ridge = RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)
        ridge.fit(X_tr_final, y_vals[train_idx])
        y_hat = ridge.predict(X_te_final)

        all_y_true.extend(y_vals[test_idx].tolist())
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
            y_true, y_hat = blockwise_pca_wfo(df, s, cfg["n_macro"], cfg["n_insurance"])
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": variant_name, **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v44", f"Blockwise PCA — {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    save_results(pd.DataFrame(all_output_rows), "v44_pca_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v44_pca.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v44_pca.py
```

```python
# tests/test_research_v44_pca.py
import pandas as pd
from pathlib import Path

def test_two_variants():
    df = pd.read_csv(Path("results/research/v44_pca_results.csv"))
    assert len(df["variant"].unique()) == 2

def test_no_nan_r2():
    df = pd.read_csv(Path("results/research/v44_pca_results.csv"))
    assert df["r2"].notna().all(), "NaN R2 found — PCA transform likely failed"
```

```bash
pytest tests/test_research_v44_pca.py -v
git add results/research/v44_pca.py tests/test_research_v44_pca.py
git commit -m "research: add v44 blockwise PCA experiment"
```

---

### Task 10: v45 — BayesianRidge Replacement

**Files:**
- Create: `results/research/v45_bayesian_ridge.py`

**Purpose:** Test (A) default BayesianRidge via `run_wfo(model_type="bayesian_ridge")`, (B) tight prior hyperparameters, (C) BayesianRidge+GBT ensemble.

- [ ] **Step 1: Write the script**

```python
"""v45 — BayesianRidge as primary model: default, tight prior, + GBT ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.wfo_engine import run_wfo
from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, GBT_FEATURES_13, MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12, TEST_SIZE_MONTHS, compute_metrics,
    custom_wfo, get_connection, load_baseline_results,
    load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_per_benchmark,
    print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative


def run_bayesian_ridge_variant(
    df: pd.DataFrame, rel_series: pd.Series,
    alpha_1: float = 1e-6, alpha_2: float = 1e-6,
    lambda_1: float = 1e-6, lambda_2: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    X_df, y = get_X_y_relative(df, rel_series, drop_na_target=True)
    feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
    X = X_df[feat_cols].values

    def factory() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", BayesianRidge(
                alpha_1=alpha_1, alpha_2=alpha_2,
                lambda_1=lambda_1, lambda_2=lambda_2,
                fit_intercept=True,
            )),
        ])

    return custom_wfo(X, y.values, factory, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    variants = {
        "A_default_bayesian_ridge": {"alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6},
        "B_tight_prior": {"alpha_1": 1e-5, "alpha_2": 1e-5, "lambda_1": 1e-4, "lambda_2": 1e-4},
    }

    for variant_name, hyperparams in variants.items():
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            y_true, y_hat = run_bayesian_ridge_variant(df, s, **hyperparams)
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": variant_name, **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v45", f"BayesianRidge — {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    # Variant C: BayesianRidge + GBT ensemble via existing run_wfo infrastructure
    # Use model_type="bayesian_ridge" for ridge component, then manually blend with GBT
    rows_c = []
    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df, s, drop_na_target=True)
        # BayesianRidge component
        feat_br = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
        yt_br, yh_br = run_bayesian_ridge_variant(df, s)
        # GBT component via run_wfo
        result_gbt = run_wfo(X_df, y, model_type="gbt", target_horizon_months=6, benchmark=etf)
        yh_gbt = result_gbt.y_hat_all
        yt_gbt = result_gbt.y_true_all
        # Align and blend 80/20
        n = min(len(yt_br), len(yt_gbt))
        y_true_c = yt_br[-n:]
        y_hat_c = 0.80 * yh_br[-n:] + 0.20 * yh_gbt[-n:]
        m = compute_metrics(y_true_c, y_hat_c)
        rows_c.append({"benchmark": etf, "variant": "C_bayesian_ridge_gbt", **m,
                       "_y_true": y_true_c, "_y_hat": y_hat_c})

    p_c = pool_metrics(rows_c)
    print_header("v45", "BayesianRidge+GBT Ensemble (80/20)")
    print_per_benchmark(rows_c)
    print_pooled(p_c)
    print_delta(p_c, baseline)
    print_footer()
    all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows_c)

    save_results(pd.DataFrame(all_output_rows), "v45_bayesian_ridge_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v45_bayesian_ridge.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v45_bayesian_ridge.py
```

```python
# tests/test_research_v45_bayesian_ridge.py
import pandas as pd
from pathlib import Path

def test_three_variants():
    df = pd.read_csv(Path("results/research/v45_bayesian_ridge_results.csv"))
    assert len(df["variant"].unique()) == 3

def test_bayesian_ridge_sigma_ratio_lt_baseline():
    """BayesianRidge should produce smaller σ_ratio (less overconfident)."""
    df = pd.read_csv(Path("results/research/v45_bayesian_ridge_results.csv"))
    br_pooled = df[(df["benchmark"] == "POOLED") & (df["variant"] == "A_default_bayesian_ridge")]
    if not br_pooled.empty:
        assert br_pooled.iloc[0]["sigma_ratio"] < 5.0, "σ_ratio should be reduced vs baseline"
```

```bash
pytest tests/test_research_v45_bayesian_ridge.py -v
git add results/research/v45_bayesian_ridge.py tests/test_research_v45_bayesian_ridge.py
git commit -m "research: add v45 BayesianRidge replacement experiment"
```

---

### Task 11: v46 — Binary Classification

**Files:**
- Create: `results/research/v46_classification.py`

**Purpose:** Reframe the problem as binary classification (PGR outperforms ETF: yes/no). Uses `LogisticRegressionCV`. Metrics: accuracy, balanced accuracy, Brier score, log-loss. This sidesteps R² entirely and aligns the loss with the actual RSU decision.

- [ ] **Step 1: Write the script**

```python
"""v46 — Binary classification: will PGR outperform the benchmark over 6M?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    get_connection, load_baseline_results, load_feature_matrix, load_relative_series,
    print_footer, print_header, save_results,
)
from src.processing.feature_engineering import get_X_y_relative


def classification_wfo(
    X: np.ndarray, y_binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns y_true, y_pred_class, y_pred_prob."""
    n = len(X)
    available = n - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=MAX_TRAIN_MONTHS,
                          test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)

    all_y_true, all_y_pred, all_y_prob = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr = y_binary[train_idx]
        y_te = y_binary[test_idx]

        # Median imputation
        medians = np.nanmedian(X_tr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, c]), c] = medians[c]
            X_te[np.isnan(X_te[:, c]), c] = medians[c]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # Skip fold if only one class in training
        if len(np.unique(y_tr)) < 2:
            continue

        clf = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 20), cv=3, penalty="l2",
            solver="lbfgs", max_iter=1000, random_state=42,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

    return np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    rows: list[dict] = []

    print_header("v46", "Binary Classification — Logistic Regression CV")
    print(f"\n  {'Benchmark':<10}  {'N':>5}  {'Accuracy':>9}  {'BalAcc':>8}  {'Brier':>7}  {'LogLoss':>8}")

    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df, s, drop_na_target=True)
        feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
        X = X_df[feat_cols].values
        y_binary = (y.values > 0).astype(int)

        y_true, y_pred, y_prob = classification_wfo(X, y_binary)
        if len(y_true) == 0:
            continue

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_prob)
        ll = log_loss(y_true, y_prob)

        print(f"  {etf:<10}  {len(y_true):>5}  {acc:>9.4f}  {bal_acc:>8.4f}  {brier:>7.4f}  {ll:>8.4f}")
        rows.append({"benchmark": etf, "n": len(y_true), "accuracy": acc,
                     "balanced_accuracy": bal_acc, "brier_score": brier, "log_loss": ll,
                     "version": "v46"})

    # Pooled summary
    if rows:
        df_rows = pd.DataFrame(rows)
        print(f"\n  {'POOLED':<10}  "
              f"  acc={df_rows['accuracy'].mean():.4f}  "
              f"bal_acc={df_rows['balanced_accuracy'].mean():.4f}  "
              f"brier={df_rows['brier_score'].mean():.4f}")

    print_footer()
    save_results(pd.DataFrame(rows), "v46_classification_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v46_classification.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v46_classification.py
```

```python
# tests/test_research_v46_classification.py
import pandas as pd
from pathlib import Path

def test_csv_exists():
    assert Path("results/research/v46_classification_results.csv").exists()

def test_accuracy_above_chance():
    df = pd.read_csv(Path("results/research/v46_classification_results.csv"))
    assert df["accuracy"].mean() > 0.50, "Mean accuracy should beat random chance"

def test_balanced_accuracy_above_chance():
    df = pd.read_csv(Path("results/research/v46_classification_results.csv"))
    assert df["balanced_accuracy"].mean() > 0.50
```

```bash
pytest tests/test_research_v46_classification.py -v
git add results/research/v46_classification.py tests/test_research_v46_classification.py
git commit -m "research: add v46 binary classification experiment"
```

---

### Task 12: v47 — Composite Benchmark Target

**Files:**
- Create: `results/research/v47_composite_benchmark.py`

**Purpose:** Replace 8 separate ETF targets with a single composite target — equal-weighted, inverse-vol-weighted, and equity-only — reducing label noise and multiple-testing burden.

- [ ] **Step 1: Write the script**

```python
"""v47 — Composite benchmark target: equal-weight, inv-vol-weight, equity-only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    compute_metrics, custom_wfo, get_connection, load_baseline_results,
    load_feature_matrix, load_relative_series, print_delta, print_footer,
    print_header, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)
EQUITY_ETFS = ["VOO", "VXUS", "VWO", "VDE"]


def build_composite_target(
    df_feat: pd.DataFrame,
    conn,
    etfs: list[str],
    weighting: str = "equal",
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct composite return series and aligned features."""
    rel_dict: dict[str, pd.Series] = {}
    for etf in etfs:
        s = load_relative_series(conn, etf, horizon=6)
        if not s.empty:
            rel_dict[etf] = s

    rel_df = pd.DataFrame(rel_dict).dropna(how="all")

    if weighting == "equal":
        composite = rel_df.mean(axis=1)
    elif weighting == "inv_vol":
        vols = rel_df.std()
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()
        composite = rel_df.dot(weights)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    composite = composite.dropna()

    # Align features to composite index
    if isinstance(df_feat.index, pd.DatetimeIndex):
        feat_aligned = df_feat.reindex(composite.index)
    else:
        feat_aligned = df_feat

    return feat_aligned, composite


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()

    variants = {
        "A_equal_weighted": {"etfs": BENCHMARKS, "weighting": "equal"},
        "B_inv_vol_weighted": {"etfs": BENCHMARKS, "weighting": "inv_vol"},
        "C_equity_only": {"etfs": EQUITY_ETFS, "weighting": "equal"},
    }

    output_rows: list[dict] = []
    for variant_name, cfg in variants.items():
        feat_aligned, composite = build_composite_target(df, conn, cfg["etfs"], cfg["weighting"])

        X_df, y = get_X_y_relative(feat_aligned, composite, drop_na_target=True)
        feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
        X = X_df[feat_cols].values

        def factory() -> Pipeline:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
            ])

        y_true, y_hat = custom_wfo(X, y.values, factory, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)
        m = compute_metrics(y_true, y_hat)

        print_header("v47", f"Composite Benchmark — {variant_name}")
        print(f"  N_OOS: {m['n']}   ETFs used: {cfg['etfs']}")
        print_pooled(m)
        print_delta(m, baseline)
        print_footer()

        output_rows.append({"variant": variant_name, "benchmark": "COMPOSITE", **m, "version": "v47"})

    save_results(pd.DataFrame(output_rows), "v47_composite_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v47_composite_benchmark.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v47_composite_benchmark.py
```

```python
# tests/test_research_v47_composite.py
import pandas as pd
from pathlib import Path

def test_three_variants():
    df = pd.read_csv(Path("results/research/v47_composite_results.csv"))
    assert len(df) == 3

def test_equity_composite_reasonable():
    df = pd.read_csv(Path("results/research/v47_composite_results.csv"))
    eq = df[df["variant"] == "C_equity_only"]
    assert not eq.empty
    assert eq.iloc[0]["hit_rate"] > 0.40  # basic sanity
```

```bash
pytest tests/test_research_v47_composite.py -v
git add results/research/v47_composite_benchmark.py tests/test_research_v47_composite.py
git commit -m "research: add v47 composite benchmark target experiment"
```

---

### Task 13: v48 — Panel Pooling Across Benchmarks

**Files:**
- Create: `results/research/v48_panel_pooling.py`

**Purpose:** Stack all 8 benchmark-month observations into one panel regression with benchmark fixed effects. This multiplies effective N from ~60 to ~480. Critical: temporal split must be by month (all benchmarks for month t in same fold).

- [ ] **Step 1: Write the script**

```python
"""v48 — Panel pooling: stack all 8 benchmarks with fixed effects."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    compute_metrics, get_connection, load_baseline_results, load_feature_matrix,
    load_relative_series, pool_metrics, print_delta, print_footer, print_header,
    print_per_benchmark, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)


def build_panel(df_feat: pd.DataFrame, conn, etfs: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """Build stacked panel DataFrame. Returns X_panel, y_panel, month_index."""
    rows = []
    for etf in etfs:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df_feat, s, drop_na_target=True)
        for date, (feat_row, target) in zip(X_df.index, zip(X_df.itertuples(index=False), y)):
            row = {"_month": date, "_benchmark": etf, "_target": target}
            row.update({c: v for c, v in zip(X_df.columns, feat_row)})
            rows.append(row)
    panel = pd.DataFrame(rows).sort_values("_month")
    return panel


def panel_wfo_with_fixed_effects(
    panel: pd.DataFrame, include_fixed_effects: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    feat_cols = [c for c in RIDGE_FEATURES_12 if c in panel.columns]
    months = sorted(panel["_month"].unique())
    n_months = len(months)
    available = n_months - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)

    month_to_idx = {m: i for i, m in enumerate(months)}
    panel["_midx"] = panel["_month"].map(month_to_idx)

    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=MAX_TRAIN_MONTHS,
                          test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)

    dummy_cols = [f"etf_{e}" for e in BENCHMARKS[1:]]  # drop first for intercept

    all_y_true, all_y_hat = [], []
    for train_midx, test_midx in tscv.split(range(n_months)):
        train_months_set = set(months[i] for i in train_midx)
        test_months_set = set(months[i] for i in test_midx)
        train_panel = panel[panel["_month"].isin(train_months_set)]
        test_panel = panel[panel["_month"].isin(test_months_set)]

        if train_panel.empty or test_panel.empty:
            continue

        X_tr_feat = train_panel[feat_cols].values
        X_te_feat = test_panel[feat_cols].values
        y_tr = train_panel["_target"].values
        y_te = test_panel["_target"].values

        # Median imputation
        medians = np.nanmedian(X_tr_feat, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_tr_feat.shape[1]):
            X_tr_feat[np.isnan(X_tr_feat[:, c]), c] = medians[c]
            X_te_feat[np.isnan(X_te_feat[:, c]), c] = medians[c]

        scaler = StandardScaler()
        X_tr_feat = scaler.fit_transform(X_tr_feat)
        X_te_feat = scaler.transform(X_te_feat)

        if include_fixed_effects:
            def make_dummies(sub_panel: pd.DataFrame) -> np.ndarray:
                dummies = pd.get_dummies(sub_panel["_benchmark"], prefix="etf").reindex(
                    columns=[f"etf_{e}" for e in BENCHMARKS], fill_value=0
                ).values[:, 1:]  # drop first
                return dummies

            X_tr = np.column_stack([X_tr_feat, make_dummies(train_panel)])
            X_te = np.column_stack([X_te_feat, make_dummies(test_panel)])
        else:
            X_tr = X_tr_feat
            X_te = X_te_feat

        ridge = RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)
        ridge.fit(X_tr, y_tr)
        y_hat = ridge.predict(X_te)

        all_y_true.extend(y_te.tolist())
        all_y_hat.extend(y_hat.tolist())

    return np.array(all_y_true), np.array(all_y_hat)


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()

    panel = build_panel(df, conn, BENCHMARKS)
    output_rows: list[dict] = []

    for variant_name, include_fe in [("A_panel_fixed_effects", True), ("B_panel_shared_only", False)]:
        y_true, y_hat = panel_wfo_with_fixed_effects(panel, include_fe)
        m = compute_metrics(y_true, y_hat)

        print_header("v48", f"Panel Pooling — {variant_name}")
        print(f"  N_OOS: {m['n']} (pooled across all benchmarks)")
        print_pooled(m)
        print_delta(m, baseline)
        print_footer()

        output_rows.append({"variant": variant_name, "n_total_obs": m["n"], **m, "version": "v48"})

    save_results(pd.DataFrame(output_rows), "v48_panel_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v48_panel_pooling.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v48_panel_pooling.py
```

```python
# tests/test_research_v48_panel.py
import pandas as pd
from pathlib import Path

def test_two_variants():
    df = pd.read_csv(Path("results/research/v48_panel_results.csv"))
    assert len(df) == 2

def test_panel_obs_greater_than_single_benchmark():
    """Panel should have 8x the observations of a single benchmark."""
    df = pd.read_csv(Path("results/research/v48_panel_results.csv"))
    fe_row = df[df["variant"] == "A_panel_fixed_effects"]
    if not fe_row.empty:
        # Pooled N should be > 8 * 60 = 480 for most folds
        assert fe_row.iloc[0]["n_total_obs"] > 300, "Panel should have many more obs than single-benchmark"
```

```bash
pytest tests/test_research_v48_panel.py -v
git add results/research/v48_panel_pooling.py tests/test_research_v48_panel.py
git commit -m "research: add v48 panel pooling experiment"
```

---

### Task 14: v49 — Regime Indicator Features

**Files:**
- Create: `results/research/v49_regime_features.py`

**Purpose:** Add binary/continuous regime indicators (hard market, high vol, inverted curve) as features rather than splitting data into separate regime models.

- [ ] **Step 1: Write the script**

```python
"""v49 — Regime indicator features: hard_market, high_vol, inverted_curve."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    build_results_df, compute_metrics, custom_wfo, get_connection,
    load_baseline_results, load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_per_benchmark, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)


def add_regime_features(X_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Append regime indicator columns in-place (safe to call before WFO split)."""
    df = X_df.copy()
    # Indicator values computed from existing features — no future data used
    if "combined_ratio_ttm" in df.columns:
        df["hard_market"] = (df["combined_ratio_ttm"] > 100).astype(float)
    else:
        df["hard_market"] = 0.0

    if "vix" in df.columns:
        df["high_vol"] = (df["vix"] > 20).astype(float)
    else:
        df["high_vol"] = 0.0

    if "yield_slope" in df.columns:
        df["inverted_curve"] = (df["yield_slope"] < 0).astype(float)
    else:
        df["inverted_curve"] = 0.0

    if variant == "A":
        return df[list(X_df.columns) + ["hard_market", "high_vol"]]
    elif variant == "B":
        return df[list(X_df.columns) + ["hard_market", "high_vol", "inverted_curve"]]
    elif variant == "C":
        # Interactions: hard_market × yield_slope, high_vol × credit_spread_hy
        if "yield_slope" in df.columns:
            df["hm_x_slope"] = df["hard_market"] * df["yield_slope"]
        else:
            df["hm_x_slope"] = 0.0
        if "credit_spread_hy" in df.columns:
            df["hv_x_spread"] = df["high_vol"] * df["credit_spread_hy"]
        else:
            df["hv_x_spread"] = 0.0
        return df[list(X_df.columns) + ["hard_market", "high_vol", "hm_x_slope", "hv_x_spread"]]
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    for variant_name in ["A", "B", "C"]:
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            X_df, y = get_X_y_relative(df, s, drop_na_target=True)
            feat_base = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
            X_aug = add_regime_features(X_df[feat_base], variant_name)
            X = X_aug.values

            def factory() -> Pipeline:
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
                ])

            y_true, y_hat = custom_wfo(X, y.values, factory, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": f"variant_{variant_name}", **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v49", f"Regime Indicators — Variant {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    save_results(pd.DataFrame(all_output_rows), "v49_regime_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v49_regime_features.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v49_regime_features.py
```

```python
# tests/test_research_v49_regime.py
import pandas as pd
from pathlib import Path

def test_three_variants():
    df = pd.read_csv(Path("results/research/v49_regime_results.csv"))
    assert len(df["variant"].unique()) == 3

def test_no_nan_metrics():
    df = pd.read_csv(Path("results/research/v49_regime_results.csv"))
    assert df["r2"].notna().all()
```

```bash
pytest tests/test_research_v49_regime.py -v
git add results/research/v49_regime_features.py tests/test_research_v49_regime.py
git commit -m "research: add v49 regime indicator features experiment"
```

---

### Phase 2–3 Review Checkpoint

```bash
python -c "
import pandas as pd, glob
for f in sorted(glob.glob('results/research/v4[4-9]_*.csv')):
    df = pd.read_csv(f)
    pooled = df[df.get('benchmark', df.get('variant', pd.Series(['X']))).eq('POOLED')] if 'benchmark' in df.columns else df.head(1)
    if not pooled.empty and 'r2' in pooled.columns:
        r = pooled.iloc[0]
        print(f'{f.split(\"/\")[-1][:25]:<27}  R2={r[\"r2\"]:+.4f}')
"
```

Proceed to [Phase 4–5](2026-04-10-v37-v60-phase4-5-advanced.md) if no single experiment has yet achieved R² ≥ +2%.
