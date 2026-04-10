# v37–v60 OOS R² Improvement Experiments — Master Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve aggregate OOS R² from the historical baseline toward ≥ +2% by fixing prediction calibration and architecture. As of the completed Phase 1 run, `v37` is the archived historical baseline measurement and `v38` shrinkage is the active research baseline for all later comparisons.

**Architecture:** 24 standalone research scripts in `results/research/`, one per version (v37–v60). Each script imports from the existing codebase read-only, writes a CSV to `results/research/`, and prints a standard results table. No production code is modified. A shared utility module `src/research/v37_utils.py` provides common helpers.

**Execution update (2026-04-10):** Phases 1-5 are now complete through `v60`. The ranked summary still leaves `v38_shrinkage_best_results.csv` as the top regression result, so no later-phase model has earned promotion over the low-complexity shrinkage control. The most useful late-phase output was diagnostic rather than architectural: `v60` found pooled Clark-West significance (`p = 0.0004`) and positive CE gain (`+0.0330`), which supports the view that the baseline ensemble contains signal but remains mostly constrained by variance/calibration rather than mean-direction failure.

**Tech Stack:** Python 3.10+, scikit-learn, numpy, pandas, statsmodels, SQLite (`data/pgr_financials.db`), existing `src/` modules.

---

## Sub-Plans (read in order)

| File | Experiments | Focus |
|------|-------------|-------|
| [phase1-calibration.md](2026-04-10-v37-v60-phase1-calibration.md) | v37–v43 | Baseline + prediction shrinkage, Ridge α, GBT constraint, winsorization, expanding window, feature reduction |
| [phase2-3-architecture.md](2026-04-10-v37-v60-phase2-3-architecture.md) | v44–v49 | Blockwise PCA, BayesianRidge, classification, composite benchmark, panel pooling, regime features |
| [phase4-5-advanced.md](2026-04-10-v37-v60-phase4-5-advanced.md) | v50–v60 | Prediction winsorization, peer pooling, shorter WFO windows, ARD, GPR, rank targets, 12M horizon, FRED features, imputation, Clark-West diagnostics |

---

## Environment Setup

All scripts run from the **repo root**. Activate the virtual environment first:

```bash
cd C:\Users\Jeff\Documents\pgr-vesting-decision-support
source .venv/Scripts/activate   # Windows Git Bash
# OR: .venv\Scripts\activate.bat  (CMD)
```

Run any script with:
```bash
python results/research/v37_baseline.py
```

---

## Shared Utility Module

**`src/research/v37_utils.py`** — already created. Provides:
- `get_connection()` → SQLite connection
- `load_feature_matrix(conn)` → pre-holdout feature DataFrame
- `load_relative_series(conn, etf, horizon)` → pre-holdout return series
- `compute_metrics(y_true, y_hat)` → standard metric dict (r2, ic, hit_rate, mae, sigma_ratio, …)
- `pool_metrics(records)` → aggregate pooled metrics
- `custom_wfo(X, y, pipeline_factory, ...)` → generic WFO loop with fold-level median imputation
- `print_header/print_per_benchmark/print_pooled/print_delta/print_footer` → standard output format
- `build_results_df(rows, pooled)` → DataFrame with POOLED summary row
- `save_results(df, filename)` → write to `results/research/`
- Constants: `BENCHMARKS`, `RIDGE_FEATURES_12`, `GBT_FEATURES_13`, `SHARED_7_FEATURES`, `HOLDOUT_START`, `MAX_TRAIN_MONTHS`, `TEST_SIZE_MONTHS`, `GAP_MONTHS`

---

## Critical Rules (apply to every experiment)

1. **Holdout boundary:** `HOLDOUT_START = "2024-04-01"`. No data on or after this date may be used in any experiment. The last 24 months are reserved for the single holdout evaluation of the promoted model.
2. **No production code modification.** All experiments are in `results/research/`. Production `src/` files are imported read-only.
3. **No K-Fold CV.** All validation uses `TimeSeriesSplit` with `gap=8` (or `gap=15` for 12M horizon).
4. **No `StandardScaler` on full dataset.** All scaling happens inside each WFO fold pipeline.
5. **Per-fold median imputation** for missing values — fit on training fold only.
6. **Commit after each experiment** with message `research: add vXX experiment script`.

---

## Run Order & Decision Gates

```
Week 1:  v37 → v38 → v39 → v40   (baseline + highest-leverage calibration)
Week 2:  v41 → v42 → v43 → v50   (remaining calibration)
Week 3:  v47 → v48 → v49          (target/structure reforms — run after Phase 1 shows improvement)
Week 4:  v44 → v45 → v46          (feature/model architecture)
Week 5:  v51 → v52 → v53 → v54 → v55  (additional models)
Week 6:  v56 → v57 → v58 → v59 → v60  (lower priority)
```

**Gate:** After each phase, evaluate results. If Phase 1 alone achieves OOS R² ≥ +2%, skip later phases and proceed directly to holdout evaluation per production promotion rules.

**Baseline convention going forward:** Use `results/research/v38_shrinkage_best_results.csv` as the reference baseline for Phase 2+ delta reporting. Keep `results/research/v37_baseline_results.csv` as the historical measurement of the unmodified production ensemble.

---

## Success Criteria

| Metric | Active Research Baseline | Minimum Pass | Target |
|--------|---------------|--------------|--------|
| Aggregate OOS R² | v38 = -13.10% | ≥ 0% | ≥ +2% |
| Mean IC | v38 = 0.1579 | No degradation > 0.02 | Improvement |
| Mean Hit Rate | v38 = 70.02% | No degradation > 2 pp | Improvement |
| Mean MAE | v38 = 0.1423 | No increase > 10% | Decrease |
| Clark-West p-value | TBD | < 0.05 | < 0.01 |

---

## Production Promotion Rules

1. Pre-commit to one specification before any holdout evaluation.
2. The promoted specification's test version number becomes the new production version.
3. Gate requirements: OOS R² ≥ 0% (minimum) or ≥ +2% (full pass), IC ≥ 0.07, hit rate ≥ 55%, reproducible across 2 independent runs.
4. Run holdout evaluation exactly once. Document result in `docs/results/V37_RESULTS_SUMMARY.md` (or whichever version is promoted).

---

## Source Documents (archived in repo)

- `docs/archive/history/peer-reviews/2026-04-08/claude_opus_peerreview_20260408.md`
- `docs/archive/history/peer-reviews/2026-04-08/chatgpt_peerreview_20260408.md`
- `docs/archive/history/peer-reviews/2026-04-08/v37_v60_research_plan_source.md`
