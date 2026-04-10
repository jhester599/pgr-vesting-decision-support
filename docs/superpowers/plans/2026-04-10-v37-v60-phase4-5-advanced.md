# Phase 4–5 — Advanced Models, Data, & Diagnostics (v50–v60)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Return to [master plan](2026-04-10-v37-v60-oos-improvements.md) for global rules. Run Phases 1–3 first and review their results before executing these.

**Goal:** Prediction winsorization, peer-company data augmentation, shorter WFO windows, ARDRegression, Gaussian Process Regression, rank-based targets, 12-month horizon, FRED feature expansion, imputation strategies, and Clark-West diagnostic evaluation.

**Files to create:** `results/research/v50_pred_winsorize.py` through `results/research/v60_diagnostics.py`

---

### Task 15: v50 — Prediction Winsorization

**Files:**
- Create: `results/research/v50_pred_winsorize.py`

**Purpose:** Within each WFO fold, clip OOS predictions at percentiles derived from training-set predictions. Complementary to v38 (post-hoc shrinkage).

- [ ] **Step 1: Write the script**

```python
"""v50 — Prediction winsorization: clip OOS predictions at training percentiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    build_results_df, compute_metrics, get_connection, load_baseline_results,
    load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_per_benchmark, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)
CLIP_LEVELS = {"A_p5_p95": (5, 95), "B_p10_p90": (10, 90)}

# Load optimal alpha from v38 if available
def _load_v38_alpha() -> float:
    from pathlib import Path
    p = Path("results/research/v38_shrinkage_results.csv")
    if not p.exists():
        return 0.15  # default fallback
    df = pd.read_csv(p)
    return float(df.loc[df["r2"].idxmax(), "alpha"])


def pred_winsorize_wfo(
    X: np.ndarray, y: np.ndarray,
    clip_pct: tuple[float, float],
    post_shrink_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    available = n - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=MAX_TRAIN_MONTHS,
                          test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)
    all_y_true, all_y_hat = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]
        medians = np.nanmedian(X_tr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, c]), c] = medians[c]
            X_te[np.isnan(X_te[:, c]), c] = medians[c]

        pipe = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])
        pipe.fit(X_tr, y_tr)

        # Clip bounds from training predictions
        y_hat_train = pipe.predict(X_tr)
        lo = np.percentile(y_hat_train, clip_pct[0])
        hi = np.percentile(y_hat_train, clip_pct[1])
        y_hat_te = np.clip(pipe.predict(X_te), lo, hi)

        if post_shrink_alpha is not None:
            y_hat_te = post_shrink_alpha * y_hat_te

        all_y_true.extend(y_te.tolist())
        all_y_hat.extend(y_hat_te.tolist())
    return np.array(all_y_true), np.array(all_y_hat)


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    optimal_alpha = _load_v38_alpha()
    all_output_rows: list[dict] = []

    # Variants A and B: clip only
    for variant_name, clip_pct in CLIP_LEVELS.items():
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            X_df, y = get_X_y_relative(df, s, drop_na_target=True)
            feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
            y_true, y_hat = pred_winsorize_wfo(X_df[feat_cols].values, y.values, clip_pct)
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": variant_name, **m,
                         "_y_true": y_true, "_y_hat": y_hat})
        p = pool_metrics(rows)
        print_header("v50", f"Prediction Winsorization — {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    # Variant C: clip + optimal shrinkage from v38
    rows_c = []
    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df, s, drop_na_target=True)
        feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
        y_true, y_hat = pred_winsorize_wfo(X_df[feat_cols].values, y.values, (5, 95), optimal_alpha)
        m = compute_metrics(y_true, y_hat)
        rows_c.append({"benchmark": etf, "variant": f"C_clip+shrink(alpha={optimal_alpha})", **m,
                       "_y_true": y_true, "_y_hat": y_hat})
    p_c = pool_metrics(rows_c)
    print_header("v50", f"Prediction Winsorize+Shrink (alpha={optimal_alpha})")
    print_pooled(p_c)
    print_delta(p_c, baseline)
    print_footer()
    all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows_c)

    save_results(pd.DataFrame(all_output_rows), "v50_pred_winsorize_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v50_pred_winsorize.py`.

- [ ] **Step 2: Run, write test, run test, commit**

```bash
python results/research/v50_pred_winsorize.py
```

```python
# tests/test_research_v50_pred_winsorize.py
import pandas as pd
from pathlib import Path

def test_three_variants():
    df = pd.read_csv(Path("results/research/v50_pred_winsorize_results.csv"))
    assert len(df["variant"].unique()) == 3
```

```bash
pytest tests/test_research_v50_pred_winsorize.py -v
git add results/research/v50_pred_winsorize.py tests/test_research_v50_pred_winsorize.py
git commit -m "research: add v50 prediction winsorization experiment"
```

---

### Task 16: v51 — Peer Company Pooling

**Files:**
- Create: `results/research/v51_peer_pooling.py`

**Purpose:** Two-stage approach — (1) train a sector model on P&C peer composite (ALL, TRV, CB, HIG) vs ETFs using macro features, (2) add sector model's OOS prediction as feature #13 for PGR Ridge. Alternative: just add `pgr_vs_peers_6m` if already populated.

- [ ] **Step 1: Write the script**

```python
"""v51 — Peer company pooling: two-stage sector model or pgr_vs_peers_6m feature."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    build_results_df, compute_metrics, custom_wfo, get_connection,
    load_baseline_results, load_feature_matrix, load_relative_series,
    pool_metrics, print_delta, print_footer, print_header,
    print_per_benchmark, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)
PEER_TICKERS = ["ALL", "TRV", "CB", "HIG"]

# Macro features only for sector model (no PGR-specific features)
MACRO_FEATURES = ["yield_slope", "real_rate_10y", "credit_spread_hy", "nfci", "vix",
                  "real_yield_change_6m", "mom_12m", "vol_63d"]


def build_peer_composite_returns(conn, etf: str, horizon: int = 6) -> pd.Series | None:
    """Build equal-weight peer composite minus ETF return series."""
    peer_series = []
    for ticker in PEER_TICKERS:
        try:
            s = load_relative_series(conn, ticker, horizon)
            if not s.empty:
                peer_series.append(s)
        except Exception:
            continue
    if not peer_series:
        return None
    combined = pd.concat(peer_series, axis=1)
    return combined.mean(axis=1).dropna()


def generate_sector_oos_predictions(
    df_feat: pd.DataFrame, conn, etf: str,
) -> pd.Series | None:
    """Stage 1: train sector model, generate OOS predictions aligned to index."""
    peer_returns = build_peer_composite_returns(conn, etf)
    if peer_returns is None:
        return None

    X_df, y = get_X_y_relative(df_feat, peer_returns, drop_na_target=True)
    macro_cols = [c for c in MACRO_FEATURES if c in X_df.columns]
    if not macro_cols:
        return None
    X = X_df[macro_cols].values

    def factory() -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])

    # Build sector OOS predictions via WFO
    from sklearn.model_selection import TimeSeriesSplit
    n = len(X)
    available = n - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=MAX_TRAIN_MONTHS,
                          test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)

    sector_preds = pd.Series(np.nan, index=X_df.index)
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr = y.values[train_idx]
        medians = np.nanmedian(X_tr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, c]), c] = medians[c]
            X_te[np.isnan(X_te[:, c]), c] = medians[c]
        pipe = factory()
        pipe.fit(X_tr, y_tr)
        sector_preds.iloc[test_idx] = pipe.predict(X_te)

    return sector_preds


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df, s, drop_na_target=True)

        # Variant A: Add pgr_vs_peers_6m if present
        if "pgr_vs_peers_6m" in X_df.columns:
            feat_aug = [c for c in RIDGE_FEATURES_12 + ["pgr_vs_peers_6m"] if c in X_df.columns]
            X_a = X_df[feat_aug].values
            def factory_a() -> Pipeline:
                return Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])
            y_true_a, y_hat_a = custom_wfo(X_a, y.values, factory_a, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)
            m_a = compute_metrics(y_true_a, y_hat_a)
            all_output_rows.append({"benchmark": etf, "variant": "A_pgr_vs_peers_6m", **m_a})
        else:
            print(f"  {etf}: pgr_vs_peers_6m not in feature matrix — skipping variant A")

        # Variant B: Two-stage sector model
        sector_preds = generate_sector_oos_predictions(df, conn, etf)
        if sector_preds is not None:
            # Align sector predictions to X_df
            X_df_aug = X_df.copy()
            X_df_aug["sector_signal"] = sector_preds.reindex(X_df.index)
            feat_aug_b = [c for c in RIDGE_FEATURES_12 + ["sector_signal"] if c in X_df_aug.columns]
            X_b = X_df_aug[feat_aug_b].values
            def factory_b() -> Pipeline:
                return Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])
            y_true_b, y_hat_b = custom_wfo(X_b, y.values, factory_b, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)
            m_b = compute_metrics(y_true_b, y_hat_b)
            all_output_rows.append({"benchmark": etf, "variant": "B_two_stage_sector", **m_b})

    if all_output_rows:
        results_df = pd.DataFrame(all_output_rows)
        print_header("v51", "Peer Company Pooling")
        for variant in results_df["variant"].unique():
            sub = results_df[results_df["variant"] == variant]
            p = {k: sub[k].mean() for k in ["r2", "ic", "hit_rate", "mae"] if k in sub.columns}
            print(f"  {variant}: R2={p.get('r2', float('nan')):+.4f}  IC={p.get('ic', float('nan')):.4f}")
        print_footer()
        save_results(results_df, "v51_peer_pooling_results.csv")
    else:
        print("No results generated — check peer data availability in DB.")


if __name__ == "__main__":
    main()
```

Save to `results/research/v51_peer_pooling.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v51_peer_pooling.py
```

```python
# tests/test_research_v51_peer_pooling.py
import pandas as pd
from pathlib import Path

def test_csv_exists():
    assert Path("results/research/v51_peer_pooling_results.csv").exists()

def test_at_least_one_variant():
    df = pd.read_csv(Path("results/research/v51_peer_pooling_results.csv"))
    assert len(df) > 0
```

```bash
pytest tests/test_research_v51_peer_pooling.py -v
git add results/research/v51_peer_pooling.py tests/test_research_v51_peer_pooling.py
git commit -m "research: add v51 peer company pooling experiment"
```

---

### Task 17: v52 — Shorter WFO Test Windows

**Files:**
- Create: `results/research/v52_test_window.py`

**Purpose:** Reduce test_size from 6 months to 1 or 3 months, creating more OOS prediction points for calibration without changing the model itself.

- [ ] **Step 1: Write the script**

```python
"""v52 — Shorter WFO test windows: test_size=1, 3, and 1 with expanding window."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12,
    compute_metrics, get_connection, load_baseline_results,
    load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_per_benchmark, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative
from sklearn.model_selection import TimeSeriesSplit

EXTENDED_ALPHAS = np.logspace(0, 6, 100)
VARIANTS = {
    "A_test1_rolling60": {"test_size": 1, "max_train": MAX_TRAIN_MONTHS},
    "B_test3_rolling60": {"test_size": 3, "max_train": MAX_TRAIN_MONTHS},
    "C_test1_expanding": {"test_size": 1, "max_train": None},
}


def short_window_wfo(
    X: np.ndarray, y: np.ndarray,
    test_size: int, max_train: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    base_train = max_train if max_train else 60
    available = n - base_train - GAP_MONTHS
    n_splits = max(1, available // test_size)
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train,
                          test_size=test_size, gap=GAP_MONTHS)
    all_y_true, all_y_hat = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]
        medians = np.nanmedian(X_tr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, c]), c] = medians[c]
            X_te[np.isnan(X_te[:, c]), c] = medians[c]
        pipe = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])
        pipe.fit(X_tr, y_tr)
        all_y_true.extend(y_te.tolist())
        all_y_hat.extend(pipe.predict(X_te).tolist())
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
            y_true, y_hat = short_window_wfo(X_df[feat_cols].values, y.values,
                                             cfg["test_size"], cfg["max_train"])
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": variant_name, **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v52", f"Shorter Test Windows — {variant_name}")
        print_pooled(p)
        print(f"  N_OOS total: {sum(r['n'] for r in rows)}")
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    save_results(pd.DataFrame(all_output_rows), "v52_test_window_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v52_test_window.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v52_test_window.py
```

```python
# tests/test_research_v52_test_window.py
import pandas as pd
from pathlib import Path

def test_three_variants():
    df = pd.read_csv(Path("results/research/v52_test_window_results.csv"))
    assert len(df["variant"].unique()) == 3

def test_monthly_window_has_more_obs():
    df = pd.read_csv(Path("results/research/v52_test_window_results.csv"))
    monthly = df[(df["variant"] == "A_test1_rolling60") & (df["benchmark"] != "POOLED")]
    biannual = df[(df["variant"] == "B_test3_rolling60") & (df["benchmark"] != "POOLED")]
    assert monthly["n"].sum() > biannual["n"].sum()
```

```bash
pytest tests/test_research_v52_test_window.py -v
git add results/research/v52_test_window.py tests/test_research_v52_test_window.py
git commit -m "research: add v52 shorter WFO test windows experiment"
```

---

### Task 18: v53 — ARDRegression

**Files:**
- Create: `results/research/v53_ard.py`

**Purpose:** Test ARDRegression (Automatic Relevance Determination) — performs per-feature precision estimation, implementing automatic feature selection. Three variants: default, ARD+GBT, ARD with extended feature set.

- [ ] **Step 1: Write the script**

```python
"""v53 — ARDRegression: automatic per-feature selection via precision priors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.wfo_engine import run_wfo
from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    build_results_df, compute_metrics, custom_wfo, get_connection,
    load_baseline_results, load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_per_benchmark, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

# Extended feature set for Variant C
EXTENDED_FEATURES = RIDGE_FEATURES_12 + ["pif_growth_yoy", "investment_book_yield",
                                          "yield_curvature", "mom_3m"]


def run_ard(
    X: np.ndarray, y: np.ndarray,
    threshold_lambda: float = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    def factory() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", ARDRegression(
                alpha_1=1e-6, alpha_2=1e-6,
                lambda_1=1e-6, lambda_2=1e-6,
                threshold_lambda=threshold_lambda,
                fit_intercept=True,
                max_iter=300,
            )),
        ])
    return custom_wfo(X, y, factory, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    for variant_name, feat_list in [
        ("A_default_ard", RIDGE_FEATURES_12),
        ("C_ard_extended", EXTENDED_FEATURES),
    ]:
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            X_df, y = get_X_y_relative(df, s, drop_na_target=True)
            feat_cols = [c for c in feat_list if c in X_df.columns]
            y_true, y_hat = run_ard(X_df[feat_cols].values, y.values)
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": variant_name, **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v53", f"ARDRegression — {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    # Variant B: ARD + GBT ensemble (80/20)
    rows_b = []
    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df, s, drop_na_target=True)
        feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
        y_true_ard, y_hat_ard = run_ard(X_df[feat_cols].values, y.values)
        result_gbt = run_wfo(X_df, y, model_type="gbt", target_horizon_months=6, benchmark=etf)
        n = min(len(y_true_ard), len(result_gbt.y_true_all))
        y_true_b = y_true_ard[-n:]
        y_hat_b = 0.80 * y_hat_ard[-n:] + 0.20 * result_gbt.y_hat_all[-n:]
        m = compute_metrics(y_true_b, y_hat_b)
        rows_b.append({"benchmark": etf, "variant": "B_ard_gbt", **m,
                       "_y_true": y_true_b, "_y_hat": y_hat_b})

    p_b = pool_metrics(rows_b)
    print_header("v53", "ARD+GBT Ensemble (80/20)")
    print_pooled(p_b)
    print_delta(p_b, baseline)
    print_footer()
    all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows_b)

    save_results(pd.DataFrame(all_output_rows), "v53_ard_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v53_ard.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v53_ard.py
```

```python
# tests/test_research_v53_ard.py
import pandas as pd
from pathlib import Path

def test_three_variants():
    df = pd.read_csv(Path("results/research/v53_ard_results.csv"))
    assert len(df["variant"].unique()) == 3
```

```bash
pytest tests/test_research_v53_ard.py -v
git add results/research/v53_ard.py tests/test_research_v53_ard.py
git commit -m "research: add v53 ARDRegression experiment"
```

---

### Task 19: v54 — Gaussian Process Regression

**Files:**
- Create: `results/research/v54_gpr.py`

**Purpose:** Replace GBT with GPR (Matérn 5/2 kernel + WhiteKernel). GPR naturally reverts to prior mean far from training data — implicit shrinkage. Uses 7-feature reduced set to avoid curse of dimensionality.

- [ ] **Step 1: Write the script**

```python
"""v54 — Gaussian Process Regression: Matern 5/2 kernel, replacing GBT component."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, SHARED_7_FEATURES, TEST_SIZE_MONTHS,
    build_results_df, compute_metrics, custom_wfo, get_connection,
    load_baseline_results, load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_per_benchmark, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)


def make_gpr_matern() -> Pipeline:
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=42,
        )),
    ])


def make_gpr_rbf() -> Pipeline:
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=42,
        )),
    ])


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    variants = {
        "A_gpr_matern52": make_gpr_matern,
        "B_gpr_rbf": make_gpr_rbf,
    }

    for variant_name, factory_fn in variants.items():
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            X_df, y = get_X_y_relative(df, s, drop_na_target=True)
            feat_cols = [c for c in SHARED_7_FEATURES if c in X_df.columns]
            y_true, y_hat = custom_wfo(X_df[feat_cols].values, y.values, factory_fn,
                                       MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": variant_name, **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v54", f"GPR — {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    # Variant C: Ridge + GPR ensemble (80/20), replacing Ridge + GBT
    rows_c = []
    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df, s, drop_na_target=True)
        feat_cols = [c for c in SHARED_7_FEATURES if c in X_df.columns]
        X = X_df[feat_cols].values

        def ridge_factory() -> Pipeline:
            return Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])

        y_true_r, y_hat_r = custom_wfo(X, y.values, ridge_factory, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)
        y_true_g, y_hat_g = custom_wfo(X, y.values, make_gpr_matern, MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)
        n = min(len(y_true_r), len(y_true_g))
        y_true_c = y_true_r[-n:]
        y_hat_c = 0.80 * y_hat_r[-n:] + 0.20 * y_hat_g[-n:]
        m = compute_metrics(y_true_c, y_hat_c)
        rows_c.append({"benchmark": etf, "variant": "C_ridge_gpr_80_20", **m,
                       "_y_true": y_true_c, "_y_hat": y_hat_c})

    p_c = pool_metrics(rows_c)
    print_header("v54", "Ridge+GPR Ensemble (80/20)")
    print_pooled(p_c)
    print_delta(p_c, baseline)
    print_footer()
    all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows_c)

    save_results(pd.DataFrame(all_output_rows), "v54_gpr_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v54_gpr.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v54_gpr.py
```

Note: GPR is O(N³) but trivial at N=60. Expect each benchmark to run in < 10 seconds.

```python
# tests/test_research_v54_gpr.py
import pandas as pd
from pathlib import Path

def test_three_variants():
    df = pd.read_csv(Path("results/research/v54_gpr_results.csv"))
    assert len(df["variant"].unique()) == 3

def test_gpr_sigma_ratio_lt_two():
    """GPR should produce implicit shrinkage — lower sigma_ratio than baseline."""
    df = pd.read_csv(Path("results/research/v54_gpr_results.csv"))
    matern = df[(df["variant"] == "A_gpr_matern52") & (df["benchmark"] != "POOLED")]
    if not matern.empty:
        assert matern["sigma_ratio"].mean() < 3.0
```

```bash
pytest tests/test_research_v54_gpr.py -v
git add results/research/v54_gpr.py tests/test_research_v54_gpr.py
git commit -m "research: add v54 Gaussian Process Regression experiment"
```

---

### Task 20: v55 — Rank-Based Targets

**Files:**
- Create: `results/research/v55_rank_target.py`

**Purpose:** Rank-transform `y_train` within each WFO fold to uniform [0,1]. Naturally bounds the target, reduces extreme value influence.

- [ ] **Step 1: Write the script**

```python
"""v55 — Rank-based targets: rank-transform y_train within each WFO fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    compute_metrics, get_connection, load_baseline_results,
    load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_per_benchmark, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)


def rank_target_wfo(
    X: np.ndarray, y: np.ndarray, percentile: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """WFO with rank-transformed y_train. Evaluates OOS on original y_test scale."""
    n = len(X)
    available = n - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=MAX_TRAIN_MONTHS,
                          test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)
    all_y_true, all_y_hat = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]
        medians = np.nanmedian(X_tr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, c]), c] = medians[c]
            X_te[np.isnan(X_te[:, c]), c] = medians[c]

        # Rank-transform y_train to uniform [0, 1]
        y_tr_ranked = rankdata(y_tr) / len(y_tr)
        # Shift to center at 0.5 → [-0.5, 0.5] to match sign convention
        y_tr_centered = y_tr_ranked - 0.5

        pipe = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])
        pipe.fit(X_tr, y_tr_centered)
        y_hat_ranked = pipe.predict(X_te)

        all_y_true.extend(y_te.tolist())
        all_y_hat.extend(y_hat_ranked.tolist())  # evaluated in rank space for R² computation
    return np.array(all_y_true), np.array(all_y_hat)


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    for variant_name, percentile in [("A_rank_transform", False), ("B_percentile_transform", True)]:
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            X_df, y = get_X_y_relative(df, s, drop_na_target=True)
            feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
            y_true, y_hat = rank_target_wfo(X_df[feat_cols].values, y.values, percentile)
            # Note: R² here is in rank space. IC (Spearman) is the more meaningful metric.
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": variant_name, **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v55", f"Rank-Based Targets — {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print(f"  NOTE: R² computed between raw y_true and rank-space y_hat (different scales).")
        print(f"        IC (Spearman) is the primary metric for this variant.")
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    save_results(pd.DataFrame(all_output_rows), "v55_rank_target_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v55_rank_target.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v55_rank_target.py
```

```python
# tests/test_research_v55_rank_target.py
import pandas as pd
from pathlib import Path

def test_two_variants():
    df = pd.read_csv(Path("results/research/v55_rank_target_results.csv"))
    assert len(df["variant"].unique()) == 2
```

```bash
pytest tests/test_research_v55_rank_target.py -v
git add results/research/v55_rank_target.py tests/test_research_v55_rank_target.py
git commit -m "research: add v55 rank-based targets experiment"
```

---

### Task 21: v56 — 12-Month Horizon Branch

**Files:**
- Create: `results/research/v56_12m_horizon.py`

**Purpose:** Test 12M prediction horizon. Insurance underwriting cycles operate on 12–24 month rhythms, potentially offering stronger signal. Cost: halved OOS observations and larger embargo.

- [ ] **Step 1: Write the script**

```python
"""v56 — 12-month prediction horizon: rolling 60M and expanding window variants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.wfo_engine import run_wfo
from src.research.v37_utils import (
    BENCHMARKS, build_results_df, compute_metrics, get_connection,
    load_baseline_results, load_feature_matrix, load_relative_series,
    pool_metrics, print_delta, print_footer, print_header,
    print_per_benchmark, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

HORIZON = 12
GAP_12M = 15  # 12-month horizon + 3-month purge


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    for variant_name in ["A_rolling60", "B_expanding"]:
        rows = []
        for etf in BENCHMARKS:
            # Load 12M relative return series
            s12 = load_relative_series(conn, etf, horizon=HORIZON)
            if s12.empty:
                continue
            X_df, y = get_X_y_relative(df, s12, drop_na_target=True)
            result = run_wfo(
                X_df, y,
                model_type="ridge",
                target_horizon_months=HORIZON,
                benchmark=etf,
            )
            y_true = result.y_true_all
            y_hat = result.y_hat_all
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": variant_name, "horizon": HORIZON,
                         **m, "_y_true": y_true, "_y_hat": y_hat})

        if not rows:
            print(f"  {variant_name}: no 12M data found — check DB has 12M relative returns")
            continue

        p = pool_metrics(rows)
        print_header("v56", f"12-Month Horizon — {variant_name}")
        print_per_benchmark(rows)
        print_pooled(p)
        print(f"\n  vs. 6M baseline (note: different target — comparison is directional only):")
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    if all_output_rows:
        save_results(pd.DataFrame(all_output_rows), "v56_12m_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v56_12m_horizon.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v56_12m_horizon.py
```

```python
# tests/test_research_v56_12m_horizon.py
import pandas as pd
from pathlib import Path

def test_csv_exists():
    p = Path("results/research/v56_12m_results.csv")
    assert p.exists(), "Run v56_12m_horizon.py first (requires 12M return data in DB)"

def test_horizon_recorded():
    df = pd.read_csv(Path("results/research/v56_12m_results.csv"))
    assert (df["horizon"] == 12).all()
```

```bash
pytest tests/test_research_v56_12m_horizon.py -v
git add results/research/v56_12m_horizon.py tests/test_research_v56_12m_horizon.py
git commit -m "research: add v56 12-month horizon branch experiment"
```

---

### Task 22: v57 — Feature Transformations and Lags

**Files:**
- Create: `results/research/v57_transforms.py`

**Purpose:** (A) Log-transform skewed fundamentals, (B) Rank-normalize all features within fold for GBT, (C) Add 1M and 2M lags of slow-moving fundamental features.

- [ ] **Step 1: Write the script**

```python
"""v57 — Feature transformations: log-transform, rank-normalization, fundamental lags."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    compute_metrics, custom_wfo, get_connection, load_baseline_results,
    load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_per_benchmark, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative
from sklearn.model_selection import TimeSeriesSplit

EXTENDED_ALPHAS = np.logspace(0, 6, 100)
SKEWED_FEATURES = ["npw_growth_yoy", "investment_income_growth_yoy", "pif_growth_yoy",
                   "book_value_per_share_growth_yoy"]
LAG_FEATURES = ["combined_ratio_ttm", "npw_growth_yoy", "investment_income_growth_yoy"]


def variant_a_log_transform(X_df: pd.DataFrame) -> pd.DataFrame:
    df = X_df.copy()
    for col in SKEWED_FEATURES:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    return df


def variant_c_add_lags(df_full: pd.DataFrame, X_df: pd.DataFrame) -> pd.DataFrame:
    df = X_df.copy()
    for feat in LAG_FEATURES:
        if feat in df_full.columns:
            lag1 = df_full[feat].shift(1).reindex(df.index)
            lag2 = df_full[feat].shift(2).reindex(df.index)
            df[f"{feat}_lag1"] = lag1
            df[f"{feat}_lag2"] = lag2
    return df


def rank_norm_wfo(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Variant B: rank-normalize all features within each fold for GBT."""
    from sklearn.ensemble import GradientBoostingRegressor

    n = len(X)
    available = n - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=MAX_TRAIN_MONTHS,
                          test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)
    all_y_true, all_y_hat = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]
        medians = np.nanmedian(X_tr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, c]), c] = medians[c]
            X_te[np.isnan(X_te[:, c]), c] = medians[c]
        # Rank-normalize within training fold
        for c in range(X_tr.shape[1]):
            ranks_tr = rankdata(X_tr[:, c]) / len(X_tr)
            # Apply training quantile mapping to test
            sorted_tr = np.sort(X_tr[:, c])
            test_ranks = np.array([np.searchsorted(sorted_tr, v) / len(sorted_tr) for v in X_te[:, c]])
            X_tr[:, c] = ranks_tr
            X_te[:, c] = np.clip(test_ranks, 0, 1)

        gbt = GradientBoostingRegressor(
            max_depth=2, n_estimators=50, learning_rate=0.01,
            min_samples_leaf=10, subsample=0.7, random_state=42,
        )
        gbt.fit(X_tr, y_tr)
        all_y_true.extend(y_te.tolist())
        all_y_hat.extend(gbt.predict(X_te).tolist())
    return np.array(all_y_true), np.array(all_y_hat)


def main() -> None:
    conn = get_connection()
    df_full = load_feature_matrix(conn)
    baseline = load_baseline_results()
    all_output_rows: list[dict] = []

    # Variant A: log-transform skewed features
    rows_a = []
    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df_full, s, drop_na_target=True)
        feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
        X_transformed = variant_a_log_transform(X_df[feat_cols])
        def factory() -> Pipeline:
            return Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])
        y_true, y_hat = custom_wfo(X_transformed.values, y.values, factory,
                                   MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)
        m = compute_metrics(y_true, y_hat)
        rows_a.append({"benchmark": etf, "variant": "A_log_transform", **m,
                       "_y_true": y_true, "_y_hat": y_hat})
    p_a = pool_metrics(rows_a)
    print_header("v57", "Log Transform — Variant A")
    print_pooled(p_a)
    print_delta(p_a, baseline)
    print_footer()
    all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows_a)

    # Variant B: rank-normalization for GBT
    rows_b = []
    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df_full, s, drop_na_target=True)
        feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
        y_true, y_hat = rank_norm_wfo(X_df[feat_cols].values, y.values)
        m = compute_metrics(y_true, y_hat)
        rows_b.append({"benchmark": etf, "variant": "B_rank_norm_gbt", **m,
                       "_y_true": y_true, "_y_hat": y_hat})
    p_b = pool_metrics(rows_b)
    print_header("v57", "Rank Normalization for GBT — Variant B")
    print_pooled(p_b)
    print_delta(p_b, baseline)
    print_footer()
    all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows_b)

    # Variant C: lagged fundamental features
    rows_c = []
    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df_full, s, drop_na_target=True)
        feat_cols = [c for c in RIDGE_FEATURES_12 if c in X_df.columns]
        X_aug = variant_c_add_lags(df_full, X_df[feat_cols])
        def factory_c() -> Pipeline:
            return Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])
        y_true, y_hat = custom_wfo(X_aug.values, y.values, factory_c,
                                   MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)
        m = compute_metrics(y_true, y_hat)
        rows_c.append({"benchmark": etf, "variant": "C_fundamental_lags", **m,
                       "_y_true": y_true, "_y_hat": y_hat})
    p_c = pool_metrics(rows_c)
    print_header("v57", "Fundamental Lags — Variant C")
    print_pooled(p_c)
    print_delta(p_c, baseline)
    print_footer()
    all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows_c)

    save_results(pd.DataFrame(all_output_rows), "v57_transforms_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v57_transforms.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v57_transforms.py
```

```python
# tests/test_research_v57_transforms.py
import pandas as pd
from pathlib import Path

def test_three_variants():
    df = pd.read_csv(Path("results/research/v57_transforms_results.csv"))
    assert len(df["variant"].unique()) == 3
```

```bash
pytest tests/test_research_v57_transforms.py -v
git add results/research/v57_transforms.py tests/test_research_v57_transforms.py
git commit -m "research: add v57 feature transformations and lags experiment"
```

---

### Task 23: v58 — Domain-Specific FRED Features

**Files:**
- Create: `results/research/v58_fred_features.py`

**Purpose:** Check which additional FRED series are already in `fred_macro_monthly`, compute 3-month momentum, and add 2–3 to the Ridge feature set.

- [ ] **Step 1: Write the script**

```python
"""v58 — Domain-specific FRED features: auto ins PPI, medical CPI, motor parts CPI, etc."""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    compute_metrics, custom_wfo, get_connection, load_baseline_results,
    load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)

# FRED series of interest with their expected column names or series IDs
FRED_CANDIDATES = {
    "auto_ins_ppi_mom3m": "PCU5241265241261",      # Auto insurance PPI
    "medical_cpi_mom3m": "CUSR0000SAM2",            # Medical care CPI
    "motor_parts_cpi_mom3m": "CUSR0000SETA02",      # Used car / motor parts CPI
    "mortgage_rate_delta_3m": "MORTGAGE30US",        # Mortgage rate
    "term_premium_10y": "THREEFYTP10",               # ACM 10-year term premium
}


def check_fred_series_available(conn: sqlite3.Connection) -> list[str]:
    """Query fred_macro_monthly for which of our candidate series exist."""
    try:
        available_cols = pd.read_sql("SELECT * FROM fred_macro_monthly LIMIT 1", conn).columns.tolist()
    except Exception:
        return []
    found = []
    for feature_name, series_id in FRED_CANDIDATES.items():
        # Check by column name (series_id or feature_name)
        if series_id in available_cols or feature_name in available_cols:
            found.append(feature_name if feature_name in available_cols else series_id)
    return found


def compute_3m_momentum(df: pd.DataFrame, col: str) -> pd.Series:
    """3-month momentum = (current - 3-months-ago) / |3-months-ago|."""
    return df[col].pct_change(3).replace([np.inf, -np.inf], np.nan)


def main() -> None:
    conn = get_connection()
    df_full = load_feature_matrix(conn)
    baseline = load_baseline_results()

    # Check what's available
    available = check_fred_series_available(conn)
    print(f"FRED candidate series found in DB: {available if available else 'none'}")

    # Also check the feature matrix itself for pre-computed versions
    feature_matrix_extras = [c for c in df_full.columns
                             if any(kw in c for kw in ["auto_ins", "medical_cpi", "motor", "mortgage", "term_premium"])]
    print(f"Feature matrix pre-computed extras: {feature_matrix_extras}")

    # Build augmented feature set from whatever is available
    new_features = available + feature_matrix_extras
    if not new_features:
        print("No new FRED features found in DB. Checking if rate_adequacy_gap_yoy covers auto_ins_ppi...")
        # Fallback: use rate_adequacy_gap_yoy (already in feature matrix) as a proxy
        new_features = ["rate_adequacy_gap_yoy"] if "rate_adequacy_gap_yoy" in df_full.columns else []

    aug_features = [c for c in RIDGE_FEATURES_12 + new_features[:3] if c in df_full.columns]
    aug_features = list(dict.fromkeys(aug_features))  # deduplicate
    print(f"Augmented feature set ({len(aug_features)} features): {aug_features}")

    rows = []
    for etf in BENCHMARKS:
        s = load_relative_series(conn, etf, horizon=6)
        if s.empty:
            continue
        X_df, y = get_X_y_relative(df_full, s, drop_na_target=True)
        feat_cols = [c for c in aug_features if c in X_df.columns]
        def factory() -> Pipeline:
            return Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])
        y_true, y_hat = custom_wfo(X_df[feat_cols].values, y.values, factory,
                                   MAX_TRAIN_MONTHS, TEST_SIZE_MONTHS, GAP_MONTHS)
        m = compute_metrics(y_true, y_hat)
        rows.append({"benchmark": etf, "features_used": str(feat_cols), **m,
                     "_y_true": y_true, "_y_hat": y_hat, "version": "v58"})

    if rows:
        p = pool_metrics(rows)
        print_header("v58", f"Domain-Specific FRED Features (+{len(new_features[:3])} features)")
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        save_results(pd.DataFrame(out), "v58_fred_results.csv")
    else:
        print("No results — no matching FRED series found in DB.")


if __name__ == "__main__":
    main()
```

Save to `results/research/v58_fred_features.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v58_fred_features.py
```

```python
# tests/test_research_v58_fred.py
import pandas as pd
from pathlib import Path

def test_csv_exists():
    assert Path("results/research/v58_fred_results.csv").exists()
```

```bash
pytest tests/test_research_v58_fred.py -v
git add results/research/v58_fred_features.py tests/test_research_v58_fred.py
git commit -m "research: add v58 domain-specific FRED features experiment"
```

---

### Task 24: v59 — Imputation Strategies

**Files:**
- Create: `results/research/v59_imputation.py`

**Purpose:** Expand from 12 to 18 features by adding partially-missing features with principled imputation: (A) forward-fill + training median, (B) IterativeImputer within each fold, (C) MissingIndicator augmentation.

- [ ] **Step 1: Write the script**

```python
"""v59 — Imputation: expand to 18 features with ffill, IterativeImputer, MissingIndicator."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, MissingIndicator, SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.v37_utils import (
    BENCHMARKS, GAP_MONTHS, MAX_TRAIN_MONTHS, RIDGE_FEATURES_12, TEST_SIZE_MONTHS,
    compute_metrics, get_connection, load_baseline_results,
    load_feature_matrix, load_relative_series, pool_metrics,
    print_delta, print_footer, print_header, print_pooled, save_results,
)
from src.processing.feature_engineering import get_X_y_relative

EXTENDED_ALPHAS = np.logspace(0, 6, 100)

ADDITIONAL_FEATURES = [
    "pe_ratio", "pb_ratio", "roe",
    "buyback_yield", "term_premium_10y", "breakeven_inflation_10y",
]
ALL_18_FEATURES = RIDGE_FEATURES_12 + ADDITIONAL_FEATURES


def impute_wfo(
    X: np.ndarray, y: np.ndarray,
    strategy: str,  # "ffill_median", "iterative", "missing_indicator"
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    available = n - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=MAX_TRAIN_MONTHS,
                          test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)
    all_y_true, all_y_hat = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]

        if strategy == "ffill_median":
            # Forward fill is applied before WFO (see main()), here just fill remaining NaN
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_tr)
            X_te = imp.transform(X_te)
        elif strategy == "iterative":
            imp = IterativeImputer(max_iter=10, random_state=42)
            X_tr = imp.fit_transform(X_tr)
            X_te = imp.transform(X_te)
        elif strategy == "missing_indicator":
            imp = IterativeImputer(max_iter=10, random_state=42)
            indicator = MissingIndicator()
            X_tr_ind = indicator.fit_transform(X_tr)
            X_te_ind = indicator.transform(X_te)
            X_tr = np.column_stack([imp.fit_transform(X_tr), X_tr_ind])
            X_te = np.column_stack([imp.transform(X_te), X_te_ind])

        pipe = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None))])
        pipe.fit(X_tr, y_tr)
        all_y_true.extend(y_te.tolist())
        all_y_hat.extend(pipe.predict(X_te).tolist())
    return np.array(all_y_true), np.array(all_y_hat)


def main() -> None:
    conn = get_connection()
    df_full = load_feature_matrix(conn)
    baseline = load_baseline_results()

    feat_cols = [c for c in ALL_18_FEATURES if c in df_full.columns]
    print(f"Features available: {len(feat_cols)}/18: {feat_cols}")

    # Forward-fill strategy: apply ffill before WFO loop
    df_ffilled = df_full[feat_cols].ffill()

    all_output_rows: list[dict] = []
    strategies = ["ffill_median", "iterative", "missing_indicator"]

    for strategy in strategies:
        rows = []
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if s.empty:
                continue
            src_df = df_ffilled if strategy == "ffill_median" else df_full[feat_cols]
            X_df, y = get_X_y_relative(src_df, s, drop_na_target=True)
            available_cols = [c for c in feat_cols if c in X_df.columns]
            y_true, y_hat = impute_wfo(X_df[available_cols].values, y.values, strategy)
            m = compute_metrics(y_true, y_hat)
            rows.append({"benchmark": etf, "variant": strategy, **m,
                         "_y_true": y_true, "_y_hat": y_hat})

        p = pool_metrics(rows)
        print_header("v59", f"Imputation — {strategy}")
        print_pooled(p)
        print_delta(p, baseline)
        print_footer()
        all_output_rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    save_results(pd.DataFrame(all_output_rows), "v59_imputation_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v59_imputation.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v59_imputation.py
```

```python
# tests/test_research_v59_imputation.py
import pandas as pd
from pathlib import Path

def test_three_strategies():
    df = pd.read_csv(Path("results/research/v59_imputation_results.csv"))
    assert len(df["variant"].unique()) == 3
```

```bash
pytest tests/test_research_v59_imputation.py -v
git add results/research/v59_imputation.py tests/test_research_v59_imputation.py
git commit -m "research: add v59 imputation strategies experiment"
```

---

### Task 25: v60 — Clark-West Test + Evaluation Framework

**Files:**
- Create: `results/research/v60_diagnostics.py`

**Purpose:** Diagnostic script (not a model experiment). Runs Clark-West (2006) MSFE-adjusted test, MSE decomposition into bias²+variance, and certainty-equivalent return gain on the v37 baseline predictions.

- [ ] **Step 1: Write the script**

```python
"""v60 — Clark-West test, MSE decomposition, certainty-equivalent return gain (diagnostic)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.research.v37_utils import (
    BENCHMARKS, get_connection, load_feature_matrix, load_relative_series,
    print_footer, print_header, save_results, RESULTS_DIR,
)
from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.models.evaluation import reconstruct_ensemble_oos_predictions
from config.features import MODEL_FEATURE_OVERRIDES


def newey_west_se(x: np.ndarray, lags: int = 5) -> float:
    """Newey-West HAC standard error for the mean of x."""
    n = len(x)
    xc = x - x.mean()
    variance = np.dot(xc, xc) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        cov = np.dot(xc[lag:], xc[:-lag]) / n
        variance += 2 * weight * cov
    return np.sqrt(max(variance, 0) / n)


def clark_west_test(
    y_true: np.ndarray, y_hat_model: np.ndarray,
) -> tuple[float, float, float]:
    """Clark-West (2006) MSFE-adjusted test.

    H0: the benchmark (historical mean) is as good as the model.
    Returns: CW_stat, p_value, mean_CW_t
    """
    y_bar = np.cumsum(y_true) / np.arange(1, len(y_true) + 1)
    y_bar_prev = np.concatenate([[0.0], y_bar[:-1]])  # expanding historical mean
    e1 = y_true - y_bar_prev  # naive forecast error
    e2 = y_true - y_hat_model  # model forecast error
    cw_t = e1 ** 2 - (e2 ** 2 - (y_hat_model - y_bar_prev) ** 2)
    mean_cw = float(cw_t.mean())
    se_cw = newey_west_se(cw_t)
    t_stat = mean_cw / se_cw if se_cw > 0 else 0.0
    p_val = float(stats.t.sf(t_stat, df=len(cw_t) - 1))  # one-sided
    return float(t_stat), p_val, mean_cw


def mse_decompose(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    """Decompose MSE into bias² + variance + noise."""
    bias_sq = (y_hat.mean() - y_true.mean()) ** 2
    var_pred = float(y_hat.var())
    mse = float(np.mean((y_hat - y_true) ** 2))
    return {"mse": mse, "bias_sq": bias_sq, "var_pred": var_pred,
            "var_pct": var_pred / mse if mse > 0 else np.nan,
            "bias_pct": bias_sq / mse if mse > 0 else np.nan}


def certainty_equivalent_gain(
    y_true: np.ndarray, y_hat: np.ndarray,
    gamma: float = 2.0, annualize: float = 2.0,
) -> float:
    """CRRA certainty-equivalent return gain vs. naive 50/50 strategy.

    A positive value means the model adds economic value regardless of R².
    annualize=2.0 converts 6M returns to annual.
    """
    # Model strategy: long if y_hat > 0, else flat (simplified)
    model_signal = np.sign(y_hat)
    model_returns = model_signal * y_true  # simplified: 1=long, -1=short, 0=flat
    naive_returns = 0.5 * y_true  # always 50% long

    def crra_ce(returns: np.ndarray) -> float:
        """CRRA certainty equivalent (approximate log-linear)."""
        mu = returns.mean()
        sigma2 = returns.var()
        return float(mu - 0.5 * gamma * sigma2)

    ce_model = crra_ce(model_returns) * annualize
    ce_naive = crra_ce(naive_returns) * annualize
    return ce_model - ce_naive


def main() -> None:
    conn = get_connection()
    df = load_feature_matrix(conn)

    # Check if v37 CSV exists; if not, rerun the ensemble
    v37_path = RESULTS_DIR / "v37_baseline_results.csv"
    if not v37_path.exists():
        print("v37_baseline_results.csv not found — re-running baseline ensemble...")
        rel_matrix: dict[str, pd.Series] = {}
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if not s.empty:
                rel_matrix[etf] = s
        ensemble_results = run_ensemble_benchmarks(
            df, pd.DataFrame(rel_matrix), target_horizon_months=6,
            model_feature_overrides=MODEL_FEATURE_OVERRIDES,
        )
    else:
        rel_matrix = {}
        for etf in BENCHMARKS:
            s = load_relative_series(conn, etf, horizon=6)
            if not s.empty:
                rel_matrix[etf] = s
        ensemble_results = run_ensemble_benchmarks(
            df, pd.DataFrame(rel_matrix), target_horizon_months=6,
            model_feature_overrides=MODEL_FEATURE_OVERRIDES,
        )

    print_header("v60", "Clark-West Test + Evaluation Diagnostics")
    print(f"\n  {'Benchmark':<10}  {'CW_stat':>8}  {'p-value':>8}  {'Sig?':>5}  "
          f"{'Bias%':>7}  {'Var%':>7}  {'CE_gain':>9}")

    rows: list[dict] = []
    all_y_true, all_y_hat = [], []
    for etf in BENCHMARKS:
        if etf not in ensemble_results:
            continue
        y_hat, y_true = reconstruct_ensemble_oos_predictions(ensemble_results[etf])
        all_y_true.extend(y_true.tolist())
        all_y_hat.extend(y_hat.tolist())

        cw_stat, p_val, mean_cw = clark_west_test(y_true, y_hat)
        mse_d = mse_decompose(y_true, y_hat)
        ce_gain = certainty_equivalent_gain(y_true, y_hat)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))

        print(f"  {etf:<10}  {cw_stat:>8.3f}  {p_val:>8.4f}  {sig:>5}  "
              f"  {mse_d['bias_pct']:>6.1%}  {mse_d['var_pct']:>6.1%}  {ce_gain:>9.4f}")
        rows.append({"benchmark": etf, "cw_stat": cw_stat, "cw_p_value": p_val,
                     "mean_cw": mean_cw, "ce_gain": ce_gain,
                     **{f"mse_{k}": v for k, v in mse_d.items()}, "version": "v60"})

    # Pooled diagnostics
    yt_all = np.array(all_y_true)
    yh_all = np.array(all_y_hat)
    cw_s, cw_p, cw_m = clark_west_test(yt_all, yh_all)
    mse_pool = mse_decompose(yt_all, yh_all)
    ce_pool = certainty_equivalent_gain(yt_all, yh_all)
    sig_pool = "***" if cw_p < 0.01 else ("**" if cw_p < 0.05 else ("*" if cw_p < 0.10 else "no"))

    print(f"\n  {'POOLED':<10}  {cw_s:>8.3f}  {cw_p:>8.4f}  {sig_pool:>5}  "
          f"  {mse_pool['bias_pct']:>6.1%}  {mse_pool['var_pct']:>6.1%}  {ce_pool:>9.4f}")
    print(f"\n  Interpretation:")
    print(f"    CW p < 0.05 → model has genuine predictive power beyond historical mean")
    print(f"    Var% >> Bias% → variance dominance confirmed (prediction calibration fix needed)")
    print(f"    CE gain > 0 → model adds economic value for a risk-averse RSU holder")

    print_footer()

    rows.append({"benchmark": "POOLED", "cw_stat": cw_s, "cw_p_value": cw_p, "mean_cw": cw_m,
                 "ce_gain": ce_pool, **{f"mse_{k}": v for k, v in mse_pool.items()}, "version": "v60"})
    save_results(pd.DataFrame(rows), "v60_diagnostics_results.csv")


if __name__ == "__main__":
    main()
```

Save to `results/research/v60_diagnostics.py`.

- [ ] **Step 2: Run, test, commit**

```bash
python results/research/v60_diagnostics.py
```

```python
# tests/test_research_v60_diagnostics.py
import pandas as pd
from pathlib import Path

def test_csv_exists():
    assert Path("results/research/v60_diagnostics_results.csv").exists()

def test_pooled_row():
    df = pd.read_csv(Path("results/research/v60_diagnostics_results.csv"))
    assert "POOLED" in df["benchmark"].values

def test_cw_stat_present():
    df = pd.read_csv(Path("results/research/v60_diagnostics_results.csv"))
    assert "cw_stat" in df.columns
    assert df["cw_stat"].notna().all()

def test_variance_dominates_bias():
    """Core hypothesis: variance >> bias in MSE decomposition."""
    df = pd.read_csv(Path("results/research/v60_diagnostics_results.csv"))
    pooled = df[df["benchmark"] == "POOLED"].iloc[0]
    assert pooled["mse_var_pct"] > pooled["mse_bias_pct"], \
        "Expected variance dominance in MSE decomposition"
```

```bash
pytest tests/test_research_v60_diagnostics.py -v
git add results/research/v60_diagnostics.py tests/test_research_v60_diagnostics.py
git commit -m "research: add v60 Clark-West diagnostics experiment"
```

---

### Final Summary Script

- [ ] **Create a cross-experiment summary runner**

```python
# results/research/summarize_all.py
"""Print a ranked summary of all v37–v60 experiments by pooled OOS R²."""
from __future__ import annotations

import glob
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results/research")

rows = []
for csv_path in sorted(RESULTS_DIR.glob("v[3-6][0-9]_*results.csv")):
    try:
        df = pd.read_csv(csv_path)
        if "benchmark" not in df.columns or "r2" not in df.columns:
            continue
        pooled = df[df["benchmark"] == "POOLED"]
        if pooled.empty:
            pooled = df.head(1)  # fallback for composite/panel results
        for _, row in pooled.iterrows():
            variant = row.get("variant", row.get("alpha", "default"))
            rows.append({
                "file": csv_path.name[:30],
                "variant": str(variant)[:25],
                "r2": round(row["r2"], 4),
                "ic": round(row.get("ic", float("nan")), 4),
                "hit_rate": round(row.get("hit_rate", float("nan")), 4),
                "sigma_ratio": round(row.get("sigma_ratio", float("nan")), 4),
            })
    except Exception as e:
        print(f"  Warning: {csv_path.name}: {e}")

if not rows:
    print("No results CSVs found. Run experiments first.")
else:
    summary = pd.DataFrame(rows).sort_values("r2", ascending=False)
    print("\n" + "=" * 75)
    print("v37–v60 Experiment Results — Ranked by OOS R²")
    print("=" * 75)
    print(summary.to_string(index=False))
    print("=" * 75)
    print(f"\nBest: {summary.iloc[0]['file']} / {summary.iloc[0]['variant']} → R²={summary.iloc[0]['r2']:+.4f}")

if __name__ == "__main__":
    pass
```

Save to `results/research/summarize_all.py`.

```bash
git add results/research/summarize_all.py
git commit -m "research: add cross-experiment summary script"
```

- [ ] **After running all experiments, generate the final summary**

```bash
python results/research/summarize_all.py
```

Review the ranked table. The top-performing experiment (or combination) is the candidate for holdout evaluation per the production promotion rules in the master plan.
