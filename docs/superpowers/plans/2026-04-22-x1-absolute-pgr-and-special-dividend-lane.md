# x1 Absolute PGR And Special-Dividend Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up a separate `x###` research lane for absolute PGR price/direction targets, BVPS x P/B decomposition targets, and November-snapshot Q1 special-dividend targets.

**Architecture:** Keep the lane research-only and isolated from the `v###` production/research line. Add pure target-construction utilities under `src/research/`, a deterministic inventory script under `scripts/research/`, x-series artifacts under `results/research/`, and focused tests under `tests/`.

**Tech Stack:** Python 3.10+, pandas, numpy, existing SQLite database helpers, existing monthly feature engineering, pytest. No K-Fold CV, no production wiring, no full-sample scaling.

---

## Design Summary

The x-series lane is a fully separate research namespace for absolute PGR
forecasting and annual special-dividend forecasting. It reuses the repo's
checked-in database, monthly feature engineering foundations, DRIP total-return
logic, EDGAR fundamentals, and WFO methodology, but it does not change the live
monthly decision workflow or any shadow artifacts.

x1 sets up the lane and target utilities only. It intentionally does not fit
classification, regression, decomposition, or special-dividend models.

## x-Series Roadmap

| Version | Scope | Output |
|---|---|---|
| `x1` | Lane setup, target utilities, feature inventory, data sufficiency audit | Plan, utilities, tests, inventory JSON, memo |
| `x2` | Multi-horizon absolute direction classification | Logistic and shallow classifier WFO baseline |
| `x3` | Forward-return and log-return regression benchmark | Implied-price regression comparison |
| `x4` | BVPS/BVPS-growth forecasting leg | BVPS leg diagnostics |
| `x5` | P/B/log(P/B) forecasting and recombination | Decomposition price benchmark |
| `x6` | Two-stage Q1 special-dividend annual model | Occurrence, size, expected-value diagnostics |
| `x7` | Targeted TA and insurer-specific feature expansion | Bounded feature follow-up |
| `x8` | Synthesis and shadow-readiness memo | Ranked path recommendation |

## Target Definitions

For horizons `h in {1, 3, 6, 12}`:

- `target_{h}m_return`: forward simple return from current month-end price or
  DRIP value to future month-end.
- `target_{h}m_log_return`: `log(1 + target_{h}m_return)`.
- `target_{h}m_up`: `1` when the forward return is greater than zero, else `0`.
- Future implied price is derived after modeling, not modeled as the primary
  target: `price_t * (1 + predicted_return)` or `price_t * exp(predicted_log_return)`.

For decomposition:

- `target_{h}m_bvps`: future book value per share.
- `target_{h}m_bvps_growth`: `future_bvps / current_bvps - 1`.
- `target_{h}m_log_bvps_growth`: `log(future_bvps / current_bvps)`.
- `target_{h}m_pb`: `future_price / future_bvps`.
- `target_{h}m_log_pb`: `log(target_{h}m_pb)`.

For special dividends:

- Annual observation date is the November month-end snapshot for year `Y`.
- Target event is a Q1 dividend in year `Y + 1`.
- Baseline regular quarterly dividend is inferred from historical dividend data
  before the target ex-date; it is not hardcoded.
- Stage 1 target: `special_dividend_occurred`.
- Stage 2 target: `special_dividend_excess`, equal to Q1 dividend dollars/share
  above the inferred regular baseline.
- Alternative normalized targets: excess divided by November BVPS, November
  share price, and trailing net income per share where available.

## Feature Plan

Reuse immediately:

- price, momentum, volatility, high-52-week features
- current technical indicators already in the matrix
- valuation features such as P/E, P/B, ROE
- underwriting, profitability, gainshare, and PIF features
- capital, balance-sheet, book-value, buyback, and capital-return features
- macro, rates, spreads, VIX/NFCI, and peer/market-relative features

Add in x1/x2:

- target utilities and target sample-depth summaries
- November-only annual snapshot assembly
- short-horizon momentum inventory flags, with actual model use deferred to x2

Defer:

- premium-to-surplus level/change
- underwriting income relative to equity/capital
- realized and unrealized gains relative to equity
- buyback intensity relative to BVPS growth or capital generation
- special-dividend excess-capital proxies
- targeted TA replacement/additive experiments

Exclude initially:

- broad TA feature dumps
- raw future price-level regression as the first modeling path
- any post-November information for annual dividend labels
- any preprocessing fit across the full temporal sample

## Evaluation Roadmap

x2 classification metrics:

- balanced accuracy, Brier score, log loss
- precision/recall by class, especially downside precision and recall
- calibration tables and horizon-specific WFO fold diagnostics

x3 regression metrics:

- MAE, RMSE, directional hit rate, Spearman IC
- no-change, random-walk, and drift baseline comparisons

x4/x5 decomposition metrics:

- BVPS leg error, P/B leg error, recombined price error
- valuation mean-reversion baseline comparison

x6 special-dividend metrics:

- stage-1 balanced accuracy, Brier score, log loss
- stage-2 MAE/RMSE for excess dollars/share and normalized alternatives
- expected-value error after combining occurrence probability and conditional size

Practical/economic overlays remain research-only and should measure usefulness
against simple hold/no-change baselines without altering monthly outputs.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/archive/history/x1-pgr-model-reports/` | Create | Archive the two 2026-04-21 source reports |
| `docs/superpowers/plans/2026-04-22-x1-absolute-pgr-and-special-dividend-lane.md` | Create | Formal x-series plan and x1 implementation plan |
| `src/research/x1_targets.py` | Create | Pure target-construction utilities |
| `scripts/research/x1_feature_inventory.py` | Create | Deterministic data sufficiency and feature inventory artifact writer |
| `tests/test_research_x1_targets.py` | Create | Target alignment, leakage, decomposition, and annual snapshot tests |
| `results/research/x1_feature_inventory.json` | Create | Machine-readable x1 inventory |
| `results/research/x1_research_memo.md` | Create | Short x1 memo and x2/x3/x4 ordering |

## Task 1: Archive Reports And Write Plan

- [ ] Create the x1 report archive directory.
- [ ] Copy the two source reports into the archive.
- [ ] Add archive README.
- [ ] Add this x-series implementation plan.

## Task 2: Target Utility Tests

- [ ] Write tests for multi-horizon return, log-return, and direction target alignment.
- [ ] Write tests proving trailing rows are masked when future observations are unavailable.
- [ ] Write tests for BVPS x P/B decomposition targets.
- [ ] Write tests for November snapshot annual Q1 dividend labels and inferred regular baseline.

## Task 3: Target Utility Implementation

- [ ] Implement `build_forward_return_targets`.
- [ ] Implement `build_decomposition_targets`.
- [ ] Implement `build_special_dividend_targets`.
- [ ] Keep all functions pure and independent of production entrypoints.

## Task 4: Feature Inventory And Memo

- [ ] Add `scripts/research/x1_feature_inventory.py`.
- [ ] Read prices, dividends, and EDGAR rows from the checked-in database
      through existing helpers.
- [ ] Read the existing processed feature matrix cache without refreshing it,
      so x1 does not mutate shared processed artifacts outside
      `results/research/`.
- [ ] Categorize feature columns into x-series feature groups.
- [ ] Summarize history depth and missingness for targets and feature groups.
- [ ] Write `results/research/x1_feature_inventory.json`.
- [ ] Write `results/research/x1_research_memo.md`.

## Task 5: Verification And Review

- [ ] Run `python -m pytest tests/test_research_x1_targets.py -q --tb=short`.
- [ ] Run `python scripts/research/x1_feature_inventory.py`.
- [ ] Run a focused existing regression check if target utilities touch shared behavior.
- [ ] Request code review before closing x1.

## Promotion Boundary

x1 is research-only. It must not edit `scripts/monthly_decision.py`, production
config, monthly decision artifacts, or any `v###` roadmap/governance state except
to leave the existing documents intact.
