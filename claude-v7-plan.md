# PGR Vesting Decision Support — Enhancement Plan v7.x

**Author:** Claude (Opus 4.6) code review session  
**Date:** 2026-04-02  
**Baseline:** v6.5 (984 passed, 1 skipped)  
**Repository:** `jhester599/pgr-vesting-decision-support` (master branch)

---

## Context

This plan was produced after a full codebase review on 2026-04-02.  The review
examined every module in `src/`, `scripts/`, `tests/`, `config.py`, `ROADMAP.md`,
`DEVELOPMENT_PLAN.md`, and the 2026-03 monthly decision outputs.  Each enhancement
below includes the exact files to create/modify, function signatures, test
specifications, and acceptance criteria so that Claude Code can execute each
version with high accuracy.

### Key Findings from Review

1. **Feature bloat without validation** — v6.3 and v6.4 added ~15 features to
   the ML pipeline (channel_mix, npw_growth, underwriting_income variants, ROE,
   investment metrics, buyback signals) but none have been tested for marginal
   predictive power.  The decision log shows mean IC near zero (−0.006 to +0.042),
   raising the question of whether new features are helping or adding noise.
   With ~280 observations and 25+ features the obs/feature ratio per WFO fold
   is approaching dangerous territory.

2. **Tax treatment gap** — The single sell-percentage recommendation ignores the
   17pp STCG-to-LTCG tax differential (37% vs 20%).  Both RSU lots vest with
   zero holding period (STCG at vest).  The model's typical 1–7% predicted
   6M return is dwarfed by the tax savings of holding 366 days.  No scenario
   analysis exists.

3. **EDGAR 8-K parser fragility** — The parser accepts any filing where at least
   one of `combined_ratio` or `pif_total` is parseable, with no cross-validation
   of parsed values.  Month-deduplication uses last-filing-wins rather than
   most-complete-filing-wins.  No proactive alerting when a monthly run finds
   zero new data.

4. **Decision log pollution** — `decision_log.md` contains duplicate dry-run
   entries and malformed table rows.

---

## Version Plan

| Version | Theme | Priority | Est. Effort |
|---------|-------|----------|-------------|
| **v7.0** | Feature Ablation Backtest | P1 — Critical | 2–3 hours |
| **v7.1** | Three-Scenario Tax Framework | P1 — Critical | 3–4 hours |
| **v7.2** | EDGAR 8-K Parser Hardening | P2 — Important | 1.5–2 hours |
| **v7.3** | Monthly Report Tax Section + Decision Log Fix | P2 — Important | 1.5–2 hours |
| **v7.4** | CPCV Path Stability Guard + Obs/Feature Ratio | P3 — Preventive | 1–1.5 hours |

---

## v7.0 — Feature Ablation Backtest

**Theme:** Systematically measure the marginal predictive contribution of every
feature group added since v5.0 and prune features that are net-negative.

**Rationale:** This must be done BEFORE any further feature additions.  If the
new features are noise, every downstream component (Kelly sizing, BL weights,
conformal intervals) is degraded.

### Files to Create

#### `scripts/feature_ablation.py`

New standalone script.  No changes to existing modules.

```
Usage:
    python scripts/feature_ablation.py [--benchmarks VTI,VOO,BND] [--horizons 6]
```

**Implementation specification:**

1. Connect to DB via `db_client.get_connection(config.DB_PATH)` and
   `db_client.initialize_schema(conn)`.

2. Build the full feature matrix via `build_feature_matrix_from_db(conn)`.

3. Call `get_feature_columns(df)` to get the full column list.

4. Define five feature groups as Python dicts.  Each group maps a human-readable
   label to the list of column names to **include** in that group.  Groups are
   cumulative (each successive group adds to the previous):

   ```python
   FEATURE_GROUPS: dict[str, list[str]] = {
       "A_price_only": [
           "mom_3m", "mom_6m", "mom_12m", "vol_63d",
       ],
       "B_plus_macro": [
           # Group A features plus:
           "yield_slope", "yield_curvature", "real_rate_10y",
           "credit_spread_hy", "nfci", "vix", "vmt_yoy",
       ],
       "C_plus_edgar_core": [
           # Group B features plus:
           "combined_ratio_ttm", "pif_growth_yoy", "gainshare_est",
           "pe_ratio", "pb_ratio",
       ],
       "D_plus_v60": [
           # Group C features plus:
           "high_52w", "pgr_vs_peers_6m", "pgr_vs_vfh_6m",
           "pgr_vs_kie_6m", "used_car_cpi_yoy", "medical_cpi_yoy",
           "cr_acceleration",
       ],
       "E_plus_v63_v64": [
           # Group D features plus:
           "channel_mix_agency_pct", "npw_growth_yoy",
           "underwriting_income", "underwriting_income_3m",
           "underwriting_income_growth_yoy",
           "unearned_premium_growth_yoy", "unearned_premium_to_npw_ratio",
           "roe_net_income_ttm", "roe_trend",
           "investment_income_growth_yoy", "investment_book_yield",
           "buyback_yield", "buyback_acceleration",
       ],
   }
   ```

   **Important:** Only include features that actually exist in the built
   DataFrame.  Before running each group, filter the column list:
   ```python
   available = [c for c in group_cols if c in df.columns]
   ```

5. For each feature group, for a configurable list of benchmark ETFs
   (default: `["VTI", "VOO", "VFH", "BND", "VHT", "GLD", "VNQ", "VXUS"]` —
   a representative 8 covering equities, bonds, financials, gold, REITs,
   international):

   a. Build the relative return matrix for that ETF via
      `load_relative_return_matrix(conn, etf, horizon)`.

   b. Call `get_X_y_relative(X_group, rel_series, drop_na_target=True)` where
      `X_group = X_full[available_cols]`.

   c. Call `run_wfo(X_aligned, y_aligned, model_type="elasticnet",
      target_horizon_months=horizon, benchmark=etf)`.

   d. Record `wfo_result.information_coefficient`, `wfo_result.hit_rate`,
      `wfo_result.mean_absolute_error`, and `len(y_aligned)` (observation count).

6. Also run the GBT model for each group to see if non-linear models benefit
   differently from the additional features.  Call `run_wfo` with
   `model_type="gbt"` for each group.

7. Output a CSV to `results/backtests/feature_ablation_YYYYMMDD.csv` with columns:
   ```
   feature_group, benchmark, model_type, n_obs, n_features, ic, hit_rate, mae,
   oos_r2
   ```
   Compute `oos_r2` via `compute_oos_r_squared()` from `src/reporting/backtest_report.py`
   using the `wfo_result.y_true_all` and `wfo_result.y_hat_all` arrays.

8. Print a summary table to stdout showing, for each feature group, the
   **mean IC** and **mean hit rate** averaged across all benchmarks and model
   types.  Highlight any group whose addition *decreases* mean IC (i.e.,
   group E < group D).

9. If any group is net-negative, print a recommendation to add those features
   to `config.FEATURES_TO_DROP`.

**CLI arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--benchmarks` | comma-separated str | `VTI,VOO,VFH,BND,VHT,GLD,VNQ,VXUS` | ETFs to test against |
| `--horizons` | comma-separated int | `6` | Target horizons (months) |
| `--output-dir` | str | `results/backtests` | Output directory for CSV |

### Files to Create: Tests

#### `tests/test_feature_ablation.py`

At least 10 tests:

1. `test_feature_groups_are_cumulative` — Each successive group is a strict
   superset of the prior group's columns.
2. `test_all_group_columns_exist_in_matrix` — Build the full feature matrix
   with synthetic data; every column name referenced in FEATURE_GROUPS must
   appear (after filtering for availability).
3. `test_ablation_runs_single_benchmark` — Run ablation for one benchmark
   (VTI) with a small synthetic dataset; verify the output CSV has one row
   per (group, benchmark, model_type) combination.
4. `test_output_csv_columns` — Verify the output CSV has exactly the expected
   columns.
5. `test_oos_r2_in_valid_range` — OOS R² should be a float (can be negative).
6. `test_empty_group_raises` — An empty feature group should raise ValueError.
7. `test_unavailable_features_filtered` — If a feature in the group doesn't
   exist in the DataFrame, it is silently excluded (not an error).
8. `test_group_a_has_minimum_features` — Group A must have at least 3 features.
9. `test_n_features_monotonically_increases` — In the output CSV, `n_features`
   is non-decreasing across groups A→E for any given benchmark.
10. `test_cli_help_exits_zero` — `feature_ablation.py --help` exits 0.

### Acceptance Criteria

- [ ] `python scripts/feature_ablation.py --benchmarks VTI --horizons 6` runs
  to completion and produces a CSV in `results/backtests/`.
- [ ] All 10+ tests pass.
- [ ] If v6.3/v6.4 features are net-negative, they are added to
  `config.FEATURES_TO_DROP` with a code comment referencing the ablation date.

---

## v7.1 — Three-Scenario Tax Framework

**Theme:** Replace the single sell-percentage recommendation with three
scenario-specific recommendations accounting for the STCG/LTCG/Loss tax
treatment of specific share lots.

### Files to Modify

#### `src/tax/capital_gains.py`

Add a new dataclass and two new functions after the existing code:

```python
@dataclass
class TaxScenario:
    """One of three tax scenarios for a vesting event."""
    label: str                      # "SELL_NOW_STCG", "HOLD_TO_LTCG", "HOLD_FOR_LOSS"
    sell_date: date                 # When the sale would occur
    tax_rate: float                 # Applicable tax rate (STCG or LTCG)
    holding_period_days: int        # Days from vest_date to sell_date
    predicted_return: float         # Model's predicted return over the holding period
    predicted_price: float          # current_price * (1 + predicted_return)
    gross_proceeds: float           # shares * predicted_price
    tax_liability: float            # (predicted_price - cost_basis) * shares * tax_rate
    net_proceeds: float             # gross_proceeds - tax_liability
    breakeven_return: float         # Minimum return needed to prefer this scenario
    probability: float              # Model-assigned probability (0.0–1.0)
    rationale: str                  # Human-readable explanation


@dataclass
class ThreeScenarioResult:
    """Complete three-scenario analysis for a vesting event."""
    vest_date: date
    rsu_type: str
    current_price: float
    cost_basis_per_share: float
    shares: float
    scenarios: list[TaxScenario]    # Always exactly 3 scenarios
    recommended_scenario: str       # Label of the highest-utility scenario
    stcg_ltcg_breakeven: float      # Return needed for LTCG to beat STCG
    days_to_ltcg: int               # Days from vest_date until LTCG-eligible
```

**New function 1: `compute_stcg_ltcg_breakeven`**

```python
def compute_stcg_ltcg_breakeven(
    stcg_rate: float | None = None,
    ltcg_rate: float | None = None,
) -> float:
    """
    Compute the minimum annualized return needed for holding to LTCG to beat
    selling immediately at STCG.

    The breakeven is the return R such that:
        (1+R) * (1 - ltcg_rate) = (1 - stcg_rate) + R
    Simplified: R_breakeven ≈ (stcg_rate - ltcg_rate) / (1 - ltcg_rate)

    With default rates (37% STCG, 20% LTCG): ~21.25%.

    Args:
        stcg_rate: Short-term capital gains tax rate. Default: config.STCG_RATE.
        ltcg_rate: Long-term capital gains tax rate. Default: config.LTCG_RATE.

    Returns:
        Breakeven return as a decimal (e.g., 0.2125 for 21.25%).
    """
```

**Implementation:**
```python
    if stcg_rate is None:
        stcg_rate = config.STCG_RATE
    if ltcg_rate is None:
        ltcg_rate = config.LTCG_RATE
    return (stcg_rate - ltcg_rate) / (1.0 - ltcg_rate)
```

**New function 2: `compute_three_scenarios`**

```python
def compute_three_scenarios(
    vest_date: date,
    rsu_type: str,
    shares: float,
    cost_basis_per_share: float,
    current_price: float,
    predicted_6m_return: float,
    predicted_12m_return: float,
    prob_outperform_6m: float,
    prob_outperform_12m: float,
    stcg_rate: float | None = None,
    ltcg_rate: float | None = None,
) -> ThreeScenarioResult:
    """
    Compute three tax-aware scenarios for a vesting event.

    Scenario A — Sell at Vest (STCG):
        Sell immediately on vest_date.  Tax rate = STCG.
        Proceeds are certain (no holding period risk).

    Scenario B — Hold to LTCG Eligibility:
        Hold for 366 days post-vest, then sell at LTCG rate.
        Uses predicted_12m_return (or interpolated to 366d if only 6M available).
        Tax rate = LTCG.  Proceeds are uncertain.

    Scenario C — Hold for Capital Loss:
        Relevant when predicted return is negative.  Hold until position is
        underwater, harvest the loss.  Loss offsets other capital gains at
        the higher of the two applicable rates, plus up to $3,000/yr of
        ordinary income.  Tax benefit = |loss| × applicable rate.

    Args:
        vest_date:             Date shares vest (holding period starts).
        rsu_type:              "time" or "performance".
        shares:                Number of shares in this lot.
        cost_basis_per_share:  FMV on vest date (= cost basis for RSUs).
        current_price:         Current market price per share.
        predicted_6m_return:   Model's predicted 6-month PGR relative return.
        predicted_12m_return:  Model's predicted 12-month PGR return.
                               If unavailable, pass predicted_6m_return * 2
                               as a rough annualization.
        prob_outperform_6m:    Calibrated P(PGR outperforms) at 6M.
        prob_outperform_12m:   Calibrated P(PGR outperforms) at 12M.
                               If unavailable, pass prob_outperform_6m.
        stcg_rate:             Override STCG rate.  Default: config.STCG_RATE.
        ltcg_rate:             Override LTCG rate.  Default: config.LTCG_RATE.

    Returns:
        ThreeScenarioResult with exactly 3 scenarios and a recommendation.
    """
```

**Implementation logic for `compute_three_scenarios`:**

```
stcg_rate = stcg_rate or config.STCG_RATE
ltcg_rate = ltcg_rate or config.LTCG_RATE
breakeven = compute_stcg_ltcg_breakeven(stcg_rate, ltcg_rate)

# Days until LTCG
ltcg_date = vest_date + timedelta(days=366)
days_to_ltcg = (ltcg_date - vest_date).days  # always 366

# Scenario A: Sell Now at STCG
gain_a = (current_price - cost_basis_per_share) * shares
tax_a = max(0.0, gain_a) * stcg_rate
# If gain_a < 0, tax benefit = gain_a * stcg_rate (negative tax = benefit)
if gain_a < 0:
    tax_a = gain_a * stcg_rate  # negative number = tax benefit
gross_a = shares * current_price
net_a = gross_a - tax_a
# Probability: 1.0 minus risk = certainty (proceeds are known today)
prob_a = 1.0  # certain outcome

# Scenario B: Hold to LTCG
predicted_price_b = current_price * (1.0 + predicted_12m_return)
gain_b = (predicted_price_b - cost_basis_per_share) * shares
tax_b = max(0.0, gain_b) * ltcg_rate
if gain_b < 0:
    tax_b = gain_b * ltcg_rate
gross_b = shares * predicted_price_b
net_b = gross_b - tax_b
prob_b = prob_outperform_12m  # model confidence

# Scenario C: Hold for Capital Loss (only meaningful if predicted return < 0)
if predicted_6m_return < 0:
    predicted_loss_price = current_price * (1.0 + predicted_6m_return)
    loss = (predicted_loss_price - cost_basis_per_share) * shares  # negative
    # Tax benefit: can offset gains at the higher rate, plus $3k ordinary
    tax_benefit = abs(loss) * stcg_rate  # value of the loss at highest rate
    net_c = shares * predicted_loss_price + tax_benefit
    prob_c = 1.0 - prob_outperform_6m  # probability of underperformance
else:
    # No capital loss expected; scenario is degenerate
    predicted_loss_price = current_price
    net_c = shares * current_price
    prob_c = 0.0

# Recommended scenario: highest expected net proceeds weighted by probability
# Use probability-weighted utility: E[U] = prob × net_proceeds
utility_a = prob_a * net_a
utility_b = prob_b * net_b
utility_c = prob_c * net_c if predicted_6m_return < 0 else -float("inf")
best = max(
    [("SELL_NOW_STCG", utility_a), ("HOLD_TO_LTCG", utility_b),
     ("HOLD_FOR_LOSS", utility_c)],
    key=lambda x: x[1],
)
recommended = best[0]

# Build scenario objects and return ThreeScenarioResult
```

#### `src/portfolio/rebalancer.py`

Add to the `VestingRecommendation` dataclass (after `calibrated_prob_outperform`):

```python
    # v7.1 three-scenario tax analysis
    three_scenario: "ThreeScenarioResult | None" = None
```

Add import at top of file:
```python
from src.tax.capital_gains import ThreeScenarioResult
```

No other changes to rebalancer.py; the monthly_decision.py script wires the
three_scenario result into the report (see v7.3).

### Files to Create: Tests

#### `tests/test_three_scenario_tax.py`

At least 18 tests:

**Breakeven tests (4):**
1. `test_breakeven_default_rates` — With STCG=0.37, LTCG=0.20, breakeven ≈ 0.2125.
2. `test_breakeven_zero_rate_difference` — STCG=LTCG → breakeven = 0.0.
3. `test_breakeven_custom_rates` — STCG=0.40, LTCG=0.15 → verify formula.
4. `test_breakeven_is_positive` — Always > 0 when STCG > LTCG.

**Scenario A tests (4):**
5. `test_scenario_a_immediate_gain` — current_price > cost_basis → positive tax.
6. `test_scenario_a_immediate_loss` — current_price < cost_basis → tax benefit
   (negative tax_liability).
7. `test_scenario_a_probability_is_one` — Scenario A always has probability=1.0.
8. `test_scenario_a_net_equals_gross_minus_tax` — Arithmetic identity.

**Scenario B tests (4):**
9. `test_scenario_b_uses_ltcg_rate` — tax_rate field == ltcg_rate.
10. `test_scenario_b_holding_period_366_days` — holding_period_days == 366.
11. `test_scenario_b_positive_predicted_return` — When model predicts +10%,
    gross proceeds > scenario A gross (pre-tax).
12. `test_scenario_b_uses_12m_prediction` — predicted_price reflects
    predicted_12m_return.

**Scenario C tests (3):**
13. `test_scenario_c_loss_harvest_benefit` — When predicted_6m_return = -0.15,
    net_proceeds includes tax benefit of the loss.
14. `test_scenario_c_degenerate_when_positive` — When predicted_6m_return > 0,
    probability = 0.0.
15. `test_scenario_c_tax_benefit_uses_stcg_rate` — Loss offsets at the
    higher STCG rate.

**Recommendation tests (3):**
16. `test_recommended_scenario_is_highest_utility` — recommended_scenario matches
    the scenario with highest (prob × net_proceeds).
17. `test_always_three_scenarios` — `len(result.scenarios) == 3` always.
18. `test_stcg_ltcg_breakeven_stored` — `result.stcg_ltcg_breakeven` matches
    `compute_stcg_ltcg_breakeven()`.

**Integration test (1):**
19. `test_three_scenario_with_real_lot_data` — Load the example lot data from
    the README (`vest_date=2026-07-17, shares=500, cost_basis=116.08`) and
    verify the three scenarios are internally consistent (no negative shares,
    net_a + net_b + net_c makes sense).

### Acceptance Criteria

- [ ] `from src.tax.capital_gains import compute_three_scenarios, ThreeScenarioResult`
  works.
- [ ] All 19+ tests pass.
- [ ] `compute_stcg_ltcg_breakeven()` returns ≈ 0.2125 with default config rates.
- [ ] Three scenarios returned for both positive and negative predicted returns.
- [ ] `VestingRecommendation.three_scenario` field added but optional (backward
  compatible with existing code).

---

## v7.2 — EDGAR 8-K Parser Hardening

**Theme:** Add three defensive layers to the EDGAR 8-K fetcher to prevent
silent mis-parses and missed filings.

### Files to Modify

#### `scripts/edgar_8k_fetcher.py`

**Enhancement 1: Cross-validation of parsed values**

Add a new function `_validate_parsed_record()` called immediately after
`_parse_html_exhibit()` returns a non-None result (line ~834, inside the
`for filing in filings:` loop in `fetch_and_upsert()`):

```python
def _validate_parsed_record(
    record: dict[str, Any],
    filing_date: str,
    accession: str,
) -> dict[str, Any]:
    """Cross-validate parsed 8-K fields for internal consistency.

    Checks:
      1. combined_ratio ≈ loss_lae_ratio + expense_ratio (within 5pp).
         If both sub-ratios are present and th
