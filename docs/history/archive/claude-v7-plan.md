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

v7.2 — EDGAR 8-K Parser Hardening
Theme: Add three defensive layers to the EDGAR 8-K fetcher to prevent
silent mis-parses and missed filings.
Files to Modify
`scripts/edgar_8k_fetcher.py`
Enhancement 1: Cross-validation of parsed values
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
         If both sub-ratios are present and the sum deviates > 5pp from CR,
         log a WARNING and set combined_ratio = None (prefer missing over
         wrong).
      2. net_premiums_written >= sum of segment NPW (agency + direct +
         commercial + property).  If total < sum of parts, log WARNING.
      3. pif_total should be > 1,000,000 for PGR (sanity floor).
         If parsed pif_total < 100,000, likely a mis-parse; set to None.
      4. eps_basic should be in range [-5.0, 15.0] for monthly figures.
         Out-of-range values are set to None.

    Args:
        record:       Parsed record dict from _parse_html_exhibit().
        filing_date:  ISO date string for logging context.
        accession:    Accession number for logging context.

    Returns:
        The record dict, possibly with some fields set to None.
    """
```
Implementation logic:
```python
    cr = record.get("combined_ratio")
    lr = record.get("loss_lae_ratio")
    er = record.get("expense_ratio")

    if cr is not None and lr is not None and er is not None:
        expected_cr = lr + er
        if abs(cr - expected_cr) > 5.0:
            log.warning(
                "VALIDATION: CR=%.1f but loss_ratio+expense_ratio=%.1f+%.1f=%.1f "
                "(delta=%.1f) in %s (filed %s). Setting CR=None.",
                cr, lr, er, expected_cr, abs(cr - expected_cr),
                accession, filing_date,
            )
            record["combined_ratio"] = None

    # NPW segment check
    npw_total = record.get("net_premiums_written")
    npw_parts = sum(
        record.get(k) or 0.0
        for k in ("npw_agency", "npw_direct", "npw_commercial", "npw_property")
    )
    if npw_total is not None and npw_parts > 0 and npw_total < npw_parts * 0.9:
        log.warning(
            "VALIDATION: NPW_total=%.1f < sum_of_segments=%.1f in %s (filed %s).",
            npw_total, npw_parts, accession, filing_date,
        )

    # PIF floor
    pif = record.get("pif_total")
    if pif is not None and pif < 100_000:
        log.warning(
            "VALIDATION: pif_total=%.0f < 100,000 floor in %s. Setting None.",
            pif, accession,
        )
        record["pif_total"] = None

    # EPS range
    eps = record.get("eps_basic")
    if eps is not None and (eps < -5.0 or eps > 15.0):
        log.warning(
            "VALIDATION: eps_basic=%.2f out of [-5, 15] range in %s. Setting None.",
            eps, accession,
        )
        record["eps_basic"] = None

    return record
```
Wire it in: In `fetch_and_upsert()`, after line ~834 where
`parsed = _parse_html_exhibit(html, filing_date)`, add:
```python
            if parsed is not None:
                parsed = _validate_parsed_record(parsed, filing_date, accession)
                # Re-check: if validation nullified both core fields, skip
                if parsed["combined_ratio"] is None and parsed["pif_total"] is None:
                    log.debug("Validation nullified both CR and PIF for %s — skipping.", accession)
                    continue
```
Enhancement 2: Prefer most-complete filing when deduplicating by month_end
Replace the current deduplication block (lines ~869–872):
```python
    # CURRENT (replace this):
    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        seen[rec["month_end"]] = rec
    deduped = sorted(seen.values(), key=lambda r: r["month_end"])
```
With:
```python
    # v7.2: prefer the filing with the most non-null fields when deduplicating
    def _completeness(rec: dict[str, Any]) -> int:
        """Count non-None fields (excluding month_end and derived nulls)."""
        return sum(
            1 for k, v in rec.items()
            if v is not None and k != "month_end"
        )

    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        me = rec["month_end"]
        if me not in seen or _completeness(rec) > _completeness(seen[me]):
            seen[me] = rec
    deduped = sorted(seen.values(), key=lambda r: r["month_end"])
```
Enhancement 3: Zero-new-data alerting in `fetch_and_upsert`
After the final `log.info("Upserted %d rows …")` line (~904), add:
```python
    # Check if we actually added any NEW months vs what was already in the DB
    existing_months_row = conn.execute(
        "SELECT COUNT(DISTINCT month_end) FROM pgr_edgar_monthly"
    ).fetchone()
    existing_count = existing_months_row[0] if existing_months_row else 0

    if n == 0 or (existing_count > 0 and n <= existing_count):
        log.warning(
            "NOTE: No new months added this run (upserted %d rows into "
            "a table with %d existing months). If this persists, check "
            "whether PGR has changed its 8-K filing format.",
            n, existing_count,
        )
```
Files to Create: Tests
`tests/test_edgar_validation.py`
At least 12 tests:
`test_validate_cr_consistent_with_components` — CR=95, LR=65, ER=30 → no warning, CR preserved.
`test_validate_cr_inconsistent_nullifies` — CR=95, LR=65, ER=40 → delta=10 > 5 → CR set to None.
`test_validate_cr_missing_components_no_check` — CR=95, LR=None → no validation, CR preserved.
`test_validate_pif_below_floor` — pif_total=50,000 → set to None.
`test_validate_pif_above_floor` — pif_total=20,000,000 → preserved.
`test_validate_eps_out_of_range_high` — eps_basic=25.0 → set to None.
`test_validate_eps_out_of_range_low` — eps_basic=-10.0 → set to None.
`test_validate_eps_in_range` — eps_basic=3.50 → preserved.
`test_validate_npw_segment_warning` — total < sum of parts → log warning (no nullification).
`test_dedup_prefers_more_complete` — Two records for same month_end: one with 5 non-null fields, one with 10.  After dedup the 10-field record wins.
`test_dedup_same_completeness_uses_last` — Two records for same month_end with equal completeness.  Last one wins (existing behavior for ties).
`test_validation_re_rejects_after_nullify` — If validation nullifies both CR and PIF, the record is skipped (not appended to records list).
Acceptance Criteria
[ ] All 12+ tests pass.
[ ] Running `python scripts/edgar_8k_fetcher.py --dry-run --backfill-years 2`
shows validation log messages for any inconsistent filings in the 2-year window.
[ ] No existing test regressions.
---
v7.3 — Monthly Report Tax Section + Decision Log Fix
Theme: Add tax impact analysis to the monthly recommendation report and
fix the duplicate/malformed entries in `decision_log.md`.
Files to Modify
`scripts/monthly_decision.py`
Enhancement 1: Tax Impact Section in recommendation.md
Add a new function `_write_tax_section()` that is called from the main
`_write_recommendation()` function.
```python
def _write_tax_section(
    f,
    vest_date: date,
    as_of: date,
    current_price: float,
    predicted_6m_return: float,
    predicted_12m_return: float | None,
    prob_outperform_6m: float,
    cost_basis_per_share: float,
    shares: float,
) -> None:
    """Write the Tax Impact section to the recommendation.md file handle.

    Shows:
      1. Days until LTCG eligibility for each lot
      2. STCG vs LTCG tax liability comparison
      3. Three-scenario summary (if v7.1 is available)
      4. Breakeven return for hold-to-LTCG decision

    Args:
        f:                    Open file handle (write mode).
        vest_date:            Next vesting event date.
        as_of:                As-of date for this report.
        current_price:        Latest PGR closing price.
        predicted_6m_return:  Model's weighted-average 6M relative return.
        predicted_12m_return: Model's 12M return prediction (or None).
        prob_outperform_6m:   Calibrated P(outperform) at 6M.
        cost_basis_per_share: Lot cost basis.
        shares:               Shares in this lot.
    """
```
Implementation:
```python
    from src.tax.capital_gains import (
        compute_stcg_ltcg_breakeven,
        compute_three_scenarios,
    )
    import config

    breakeven = compute_stcg_ltcg_breakeven()
    days_to_ltcg = max(0, 366 - (as_of - vest_date).days)

    # Use 12M prediction if available, else rough annualization of 6M
    pred_12m = predicted_12m_return if predicted_12m_return is not None else predicted_6m_return * 2.0

    three = compute_three_scenarios(
        vest_date=vest_date,
        rsu_type="performance",  # default; caller overrides
        shares=shares,
        cost_basis_per_share=cost_basis_per_share,
        current_price=current_price,
        predicted_6m_return=predicted_6m_return,
        predicted_12m_return=pred_12m,
        prob_outperform_6m=prob_outperform_6m,
        prob_outperform_12m=prob_outperform_6m,  # proxy until 12M model exists
    )

    f.write("\n---\n\n## Tax Impact Analysis\n\n")
    f.write(f"| Metric | Value |\n")
    f.write(f"|--------|-------|\n")
    f.write(f"| Days to LTCG eligibility | {days_to_ltcg} |\n")
    f.write(f"| STCG rate | {config.STCG_RATE:.0%} |\n")
    f.write(f"| LTCG rate | {config.LTCG_RATE:.0%} |\n")
    f.write(f"| Breakeven return (hold-to-LTCG) | {breakeven:.1%} |\n")
    f.write(f"| Model predicted 6M return | {predicted_6m_return:+.2%} |\n\n")

    f.write("### Three-Scenario Comparison\n\n")
    f.write("| Scenario | Sell Date | Tax Rate | Predicted Price | Gross | Tax | Net | Probability |\n")
    f.write("|----------|-----------|----------|-----------------|-------|-----|-----|-------------|\n")
    for s in three.scenarios:
        f.write(
            f"| {s.label} | {s.sell_date} | {s.tax_rate:.0%} | "
            f"${s.predicted_price:,.2f} | ${s.gross_proceeds:,.0f} | "
            f"${s.tax_liability:,.0f} | ${s.net_proceeds:,.0f} | "
            f"{s.probability:.0%} |\n"
        )
    f.write(f"\n**Recommended:** {three.recommended_scenario}\n\n")
```
Wire it in: In the existing `_write_recommendation()` function, after the
Per-Benchmark Signals table and before the closing `---`, add a call to
`_write_tax_section()`.  The function needs `vest_date` and `cost_basis_per_share`
which should be loaded from `config.VESTING_EVENTS` or
`data/processed/position_lots.csv` if it exists, otherwise use the hardcoded
values from the README:
```python
# Default lot data (from README — override via position_lots.csv if present)
_DEFAULT_LOTS = [
    {"vest_date": date(2026, 7, 17), "rsu_type": "performance",
     "shares": 500, "cost_basis_per_share": 116.08},
    {"vest_date": date(2027, 1, 19), "rsu_type": "time",
     "shares": 500, "cost_basis_per_share": 133.65},
]
```
For each lot where the vest_date is in the future (relative to as_of), call
`_write_tax_section()`.
Enhancement 2: Decision Log Deduplication
In the function that appends to `decision_log.md` (search for
`decision_log.md` in `monthly_decision.py`), add a guard:
```python
def _append_to_decision_log(log_path: Path, row: str, as_of: date, run_date: date, dry_run: bool) -> None:
    """Append a row to the decision log, skipping if already present.

    Deduplication key: (as_of_date, run_date, dry_run_flag).
    """
    if log_path.exists():
        existing = log_path.read_text()
        as_of_str = as_of.isoformat()
        run_date_str = run_date.isoformat()
        marker = f"| {as_of_str} | {run_date_str} |"
        # For dry runs, check for the [DRY RUN] tag too
        if dry_run:
            if marker in existing and "[DRY RUN]" in existing.split(marker)[-1].split("\n")[0]:
                return  # Already logged
        else:
            # Non-dry-run: skip if the exact (as_of, run_date) exists without [DRY RUN]
            for line in existing.split("\n"):
                if marker in line and "[DRY RUN]" not in line:
                    return  # Already logged

    with open(log_path, "a") as f:
        f.write(row + "\n")
```
Replace the existing raw `open(…, "a").write(…)` call with a call to
`_append_to_decision_log()`.
Also add a one-time cleanup: if `decision_log.md` exists and contains duplicate
rows, deduplicate them when the script starts:
```python
def _clean_decision_log(log_path: Path) -> None:
    """Remove duplicate rows from the decision log (one-time cleanup)."""
    if not log_path.exists():
        return
    lines = log_path.read_text().split("\n")
    seen: set[str] = set()
    cleaned: list[str] = []
    for line in lines:
        # Only deduplicate data rows (start with "| 20")
        if line.startswith("| 20"):
            if line in seen:
                continue
            seen.add(line)
        cleaned.append(line)
    log_path.write_text("\n".join(cleaned))
```
Call `_clean_decision_log()` once at the start of `main()`.
Files to Create: Tests
`tests/test_tax_report_section.py`
At least 10 tests:
`test_tax_section_includes_breakeven` — Output contains "Breakeven return".
`test_tax_section_includes_three_scenarios` — Output contains "SELL_NOW_STCG",
"HOLD_TO_LTCG", "HOLD_FOR_LOSS".
`test_tax_section_days_to_ltcg_future_vest` — For a vest_date 3 months from
now, days_to_ltcg > 0.
`test_tax_section_days_to_ltcg_past_vest` — For a vest_date 400 days ago,
days_to_ltcg = 0 (already eligible).
`test_decision_log_dedup_skips_duplicate` — Appending the same row twice results
in only one entry.
`test_decision_log_dedup_allows_different_dates` — Different as_of dates both appear.
`test_decision_log_dedup_dry_run_separate` — A dry-run row and a non-dry-run row
for the same date are both kept.
`test_clean_decision_log_removes_exact_dupes` — Duplicate table rows removed.
`test_clean_decision_log_preserves_headers` — Header lines not affected.
`test_clean_decision_log_no_file_noop` — Non-existent file → no error.
Acceptance Criteria
[ ] `python scripts/monthly_decision.py --dry-run` produces a recommendation.md
with a "Tax Impact Analysis" section.
[ ] The decision_log.md has no duplicate rows after running.
[ ] All 10+ tests pass.
[ ] No existing test regressions.
---
v7.4 — CPCV Path Stability Guard + Obs/Feature Ratio Warning
Theme: Detect overfitting from new features via CPCV path variance,
and warn when the obs/feature ratio drops below safe thresholds.
Files to Modify
`scripts/feature_ablation.py` (extend from v7.0)
Add a `--cpcv` flag that, for each feature group, also runs CPCV and records
path IC variance:
```python
    if args.cpcv:
        cpcv_result = run_cpcv(
            X_aligned, y_aligned,
            n_folds=config.CPCV_N_FOLDS,
            n_test_folds=config.CPCV_N_TEST_FOLDS,
        )
        row["cpcv_mean_ic"] = cpcv_result.mean_ic
        row["cpcv_ic_std"] = cpcv_result.ic_std
        row["cpcv_n_paths"] = cpcv_result.n_paths
        row["cpcv_positive_paths"] = sum(1 for ic in cpcv_result.path_ics if ic > 0)
```
Add columns `cpcv_mean_ic`, `cpcv_ic_std`, `cpcv_n_paths`, `cpcv_positive_paths`
to the output CSV.  Print a diagnostic: if group E has higher `cpcv_ic_std` than
group C, flag it as "Feature bloat increasing overfitting risk."
`src/models/wfo_engine.py`
Add a warning inside `run_wfo()` after the train/test split loop.  After line
~260 (where the loop over `TimeSeriesSplit` folds iterates), add:
```python
    # v7.4: Obs/feature ratio guard
    n_features = X.shape[1]
    obs_per_fold_min = min(
        fold.y_true.shape[0] for fold in fold_results
    ) if fold_results else 0
    train_obs_min = config.WFO_TRAIN_WINDOW_MONTHS  # 60
    obs_feature_ratio = train_obs_min / max(n_features, 1)

    if obs_feature_ratio < 8.0:
        import warnings
        warnings.warn(
            f"Low obs/feature ratio ({obs_feature_ratio:.1f}:1) with "
            f"{n_features} features and {train_obs_min}-month training window. "
            f"Consider pruning features or increasing training window. "
            f"Benchmark: {benchmark}",
            UserWarning,
            stacklevel=2,
        )
```
`config.py`
Add:
```python
# v7.4 — Feature health guards
OBS_FEATURE_RATIO_WARN: float = 8.0    # Warn when train_obs / n_features < this
OBS_FEATURE_RATIO_MIN: float = 5.0     # Hard floor — raise ValueError below this
CPCV_IC_STD_WARN: float = 0.15         # Flag benchmark if CPCV path IC std > this
```
Files to Create: Tests
`tests/test_feature_health.py`
At least 8 tests:
`test_obs_feature_ratio_warning_fires` — 60 obs, 10 features → ratio 6.0 < 8.0 → UserWarning.
`test_obs_feature_ratio_no_warning` — 60 obs, 5 features → ratio 12.0 ≥ 8.0 → no warning.
`test_cpcv_ic_std_flag` — Synthetic CPCV result with ic_std=0.20 > 0.15 → flagged.
`test_cpcv_ic_std_ok` — Synthetic CPCV result with ic_std=0.05 → not flagged.
`test_ablation_cpcv_flag_output` — When `--cpcv` is used, CSV contains cpcv columns.
`test_ablation_no_cpcv_flag` — When `--cpcv` is not used, CSV omits cpcv columns.
`test_feature_groups_increasing_features` — n_features in output CSV is
non-decreasing across groups A→E.
`test_config_constants_defined` — `OBS_FEATURE_RATIO_WARN`, `OBS_FEATURE_RATIO_MIN`,
`CPCV_IC_STD_WARN` exist in config.
Acceptance Criteria
[ ] `python scripts/feature_ablation.py --benchmarks VTI --cpcv` produces a CSV
with CPCV columns populated.
[ ] `run_wfo()` issues a `UserWarning` when obs/feature ratio < 8.0.
[ ] All 8+ tests pass.
[ ] No existing test regressions.
---
Implementation Order
Execute versions in this order.  Each version should be a single commit (or
small PR) with all tests passing before moving to the next.
```
v7.0  Feature Ablation Backtest
  ↓
  [DECISION POINT: Based on v7.0 results, optionally add features to
   config.FEATURES_TO_DROP before proceeding]
  ↓
v7.1  Three-Scenario Tax Framework
  ↓
v7.2  EDGAR 8-K Parser Hardening
  ↓
v7.3  Monthly Report Tax Section + Decision Log Fix
  ↓
v7.4  CPCV Path Stability Guard + Obs/Feature Ratio Warning
```
Critical: v7.0 must complete first because its results may require pruning
features before v7.1–v7.4 are implemented.  If the ablation shows that v6.3/v6.4
features are net-negative, add them to `config.FEATURES_TO_DROP` as part of the
v7.0 commit.
---
Test Count Targets
Version	New Tests	Cumulative (estimated)
v7.0	10+	994+
v7.1	19+	1013+
v7.2	12+	1025+
v7.3	10+	1035+
v7.4	8+	1043+
---
Files Summary
New files to create:
`scripts/feature_ablation.py`
`tests/test_feature_ablation.py`
`tests/test_three_scenario_tax.py`
`tests/test_edgar_validation.py`
`tests/test_tax_report_section.py`
`tests/test_feature_health.py`
Existing files to modify:
`src/tax/capital_gains.py` (v7.1: add TaxScenario, ThreeScenarioResult, two functions)
`src/portfolio/rebalancer.py` (v7.1: add three_scenario field to VestingRecommendation)
`scripts/edgar_8k_fetcher.py` (v7.2: validation, dedup, alerting)
`scripts/monthly_decision.py` (v7.3: tax section, decision log dedup)
`src/models/wfo_engine.py` (v7.4: obs/feature ratio warning)
`config.py` (v7.4: new guard constants)
`ROADMAP.md` (all versions: append version entries)
---
Notes for Claude Code Execution
Always run `pytest` after each version to verify no regressions.
The baseline is 984 passed, 1 skipped.
Do not modify existing test files unless a test needs updating due to
a new field on an existing dataclass (e.g., `VestingRecommendation`).
Import paths — All `src/` imports use the pattern
`from src.module.submodule import function`.  Scripts in `scripts/` prepend
the project root to `sys.path` via:
```python
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```
Test fixtures — Use the existing pattern from `test_v64_p2x_features.py`:
synthetic price data via `_make_prices()`, synthetic dividends, and optional
`pgr_monthly` DataFrames.  Do NOT use the live database in unit tests.
The SQLite database (`data/pgr_financials.db`) is committed to the repo.
Integration tests may read it but should not modify it.  Use `:memory:`
connections for unit tests.
Type hints — The codebase uses `from __future__ import annotations` in
every file.  All new functions must include full type annotations.
Logging — Use `logging.getLogger(__name__)` pattern consistent with
existing modules.  WARNING level for validation failures, INFO for progress.
