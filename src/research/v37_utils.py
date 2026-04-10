"""Shared utilities for v37–v60 OOS R² improvement experiments.

All experiment scripts import from this module.  Nothing here touches any
production code path — it is read-only access to DB, models, and reporting.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "pgr_financials.db"
RESULTS_DIR = REPO_ROOT / "results" / "research"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Holdout boundary
# Experiments train/validate ONLY on data BEFORE this date.
# The holdout period is reserved for the single final promoted-model evaluation.
# DO NOT read or use data on or after this date in any experiment.
# ---------------------------------------------------------------------------
HOLDOUT_START: str = "2024-04-01"  # last 24 months as of 2026-04-10

# ---------------------------------------------------------------------------
# Benchmark universe  (8 benchmarks used by production v11.0)
# ---------------------------------------------------------------------------
BENCHMARKS: list[str] = ["VOO", "VXUS", "VWO", "VMBS", "BND", "GLD", "DBC", "VDE"]

# ---------------------------------------------------------------------------
# WFO defaults  (must match production config)
# ---------------------------------------------------------------------------
MAX_TRAIN_MONTHS: int = 60
TEST_SIZE_MONTHS: int = 6
GAP_MONTHS: int = 8   # 6-month horizon + 2-month purge buffer

# ---------------------------------------------------------------------------
# Feature sets  (mirror config/features.py MODEL_FEATURE_OVERRIDES for v11.0)
# ---------------------------------------------------------------------------
RIDGE_FEATURES_12: list[str] = [
    "mom_12m",
    "vol_63d",
    "yield_slope",
    "real_yield_change_6m",
    "real_rate_10y",
    "credit_spread_hy",
    "nfci",
    "vix",
    "combined_ratio_ttm",
    "investment_income_growth_yoy",
    "book_value_per_share_growth_yoy",
    "npw_growth_yoy",
]

GBT_FEATURES_13: list[str] = [
    "mom_3m",
    "mom_6m",
    "mom_12m",
    "vol_63d",
    "yield_slope",
    "yield_curvature",
    "vwo_vxus_spread_6m",
    "credit_spread_hy",
    "nfci",
    "vix",
    "rate_adequacy_gap_yoy",
    "pif_growth_yoy",
    "investment_book_yield",
]

# Reduced 7-feature set from v43 research (shared features)
SHARED_7_FEATURES: list[str] = [
    "mom_12m",
    "vol_63d",
    "yield_slope",
    "credit_spread_hy",
    "vix",
    "combined_ratio_ttm",
    "npw_growth_yoy",
]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the main financials DB."""
    return sqlite3.connect(str(DB_PATH))


def load_feature_matrix(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the full feature matrix, trimmed to pre-holdout dates."""
    from src.processing.feature_engineering import build_feature_matrix_from_db

    df = build_feature_matrix_from_db(conn, force_refresh=False)
    if "date" in df.columns:
        df = df[df["date"] < HOLDOUT_START].copy()
    elif df.index.dtype == "datetime64[ns]" or isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index < HOLDOUT_START].copy()
    return df


def load_relative_series(conn: sqlite3.Connection, etf: str, horizon: int = 6) -> pd.Series:
    """Load PGR-minus-ETF relative return series, trimmed to pre-holdout dates."""
    from src.processing.multi_total_return import load_relative_return_matrix

    s = load_relative_return_matrix(conn, etf, horizon)
    if not s.empty:
        s = s[s.index < HOLDOUT_START]
    return s


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
) -> dict[str, float]:
    """Compute the standard metric bundle for one benchmark-experiment pair.

    Returns a dict with keys:
        n, r2, ic, ic_p, hit_rate, mae, std_yhat, std_ytrue, sigma_ratio
    """
    from src.reporting.backtest_report import compute_oos_r_squared, compute_newey_west_ic

    y_t_s = pd.Series(y_true)
    y_h_s = pd.Series(y_hat)

    r2: float = compute_oos_r_squared(y_h_s, y_t_s)
    nw_ic, nw_p = compute_newey_west_ic(y_h_s, y_t_s, lags=5)
    hit_rate = float((np.sign(y_hat) == np.sign(y_true)).mean())
    mae = float(np.mean(np.abs(y_hat - y_true)))
    std_yhat = float(np.std(y_hat))
    std_ytrue = float(np.std(y_true))
    sigma_ratio = std_yhat / std_ytrue if std_ytrue > 0 else np.nan

    return {
        "n": len(y_true),
        "r2": r2,
        "ic": float(nw_ic),
        "ic_p": float(nw_p),
        "hit_rate": hit_rate,
        "mae": mae,
        "std_yhat": std_yhat,
        "std_ytrue": std_ytrue,
        "sigma_ratio": sigma_ratio,
    }


def pool_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Pool per-benchmark arrays into one aggregate metric dict."""
    all_y_true = np.concatenate([r["_y_true"] for r in records])
    all_y_hat = np.concatenate([r["_y_hat"] for r in records])
    m = compute_metrics(all_y_true, all_y_hat)
    return m


# ---------------------------------------------------------------------------
# Custom WFO loop
# ---------------------------------------------------------------------------

def custom_wfo(
    X: np.ndarray,
    y: np.ndarray,
    pipeline_factory: Callable[[], Any],
    max_train: int = MAX_TRAIN_MONTHS,
    test_size: int = TEST_SIZE_MONTHS,
    gap: int = GAP_MONTHS,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic WFO loop for non-standard experiments.

    Imputes NaN with per-fold training medians before each fit.
    Returns (y_true, y_hat) as flat numpy arrays.
    """
    from sklearn.model_selection import TimeSeriesSplit

    n = len(X)
    available = n - max_train - gap
    n_splits = max(1, available // test_size)
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=max_train,
        test_size=test_size,
        gap=gap,
    )

    all_y_true: list[float] = []
    all_y_hat: list[float] = []

    for train_idx, test_idx in tscv.split(X):
        X_train = X[train_idx].copy()
        X_test = X[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Per-fold median imputation — no look-ahead
        medians = np.nanmedian(X_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, c]), c] = medians[c]
            X_test[np.isnan(X_test[:, c]), c] = medians[c]

        pipe = pipeline_factory()
        pipe.fit(X_train, y_train)
        y_hat_fold = pipe.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_hat.extend(y_hat_fold.tolist())

    return np.array(all_y_true), np.array(all_y_hat)


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def load_baseline_results() -> pd.DataFrame | None:
    """Load v37 baseline CSV for delta reporting.  Returns None if not yet run."""
    path = RESULTS_DIR / "v37_baseline_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def save_results(df: pd.DataFrame, filename: str) -> Path:
    """Save results DataFrame to results/research/ and return the path."""
    out = RESULTS_DIR / filename
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SEP = "=" * 62


def print_header(version: str, experiment: str) -> None:
    print(_SEP)
    print(f"{version}: {experiment} — Results Summary")
    print(_SEP)


def print_per_benchmark(rows: list[dict[str, Any]]) -> None:
    cols = ["benchmark", "n", "r2", "ic", "hit_rate", "mae", "sigma_ratio"]
    df = pd.DataFrame(rows)[cols]
    df = df.rename(columns={"r2": "OOS_R2", "ic": "IC", "hit_rate": "Hit%", "sigma_ratio": "σ_ratio"})
    print("\nPer-Benchmark Results:")
    print(df.to_string(index=False, float_format="{:.4f}".format))


def print_pooled(pooled: dict[str, float]) -> None:
    print(
        f"\nAggregate (pooled):\n"
        f"  OOS_R2: {pooled['r2']:.4f}   IC: {pooled['ic']:.4f}   "
        f"Hit_Rate: {pooled['hit_rate']:.4f}   MAE: {pooled['mae']:.4f}   "
        f"σ_ratio: {pooled['sigma_ratio']:.4f}"
    )


def print_delta(pooled: dict[str, float], baseline_df: pd.DataFrame | None) -> None:
    if baseline_df is None:
        print("\n(No v37 baseline CSV found — skipping delta.)")
        return
    base = baseline_df[baseline_df["benchmark"] == "POOLED"]
    if base.empty:
        return
    b = base.iloc[0]
    dr2 = pooled["r2"] - b["r2"]
    dic = pooled["ic"] - b["ic"]
    dhr = pooled["hit_rate"] - b["hit_rate"]
    print(
        f"\nvs. Baseline (v37):\n"
        f"  OOS_R2 delta:    {dr2:+.4f}  ({dr2*100:+.2f} pp)\n"
        f"  IC delta:        {dic:+.4f}\n"
        f"  Hit rate delta:  {dhr:+.4f}  ({dhr*100:+.2f} pp)"
    )


def print_footer() -> None:
    print(_SEP)


def build_results_df(
    rows: list[dict[str, Any]],
    pooled: dict[str, float],
    extra_cols: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a results DataFrame including a POOLED summary row."""
    keep = ["benchmark", "n", "r2", "ic", "ic_p", "hit_rate", "mae",
            "std_yhat", "std_ytrue", "sigma_ratio"]
    per_bench = [{k: v for k, v in r.items() if k in keep} for r in rows]
    pooled_row = {"benchmark": "POOLED", **{k: pooled[k] for k in keep if k in pooled and k != "benchmark"}}
    if extra_cols:
        for r in per_bench:
            r.update(extra_cols)
        pooled_row.update(extra_cols)
    all_rows = per_bench + [pooled_row]
    return pd.DataFrame(all_rows)
