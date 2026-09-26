"""Mutation study of review 2026-09-25, F28 (step 9 re-run).

The review's area-9 method: make one small change to a production formula
in a scratch clone, run the tests related to it, revert, and count the
mutations that no test catches ("survivors"). F28 found 16 of 18 survivors;
step 9 (``docs/reviews/2026-09-25_step9_test_hardening.md``) re-ran them
after steps 1-7 (10 of 18) and added tests until none survive.

M01-M18 are the F28 mutations on today's code (sites moved since the
review; M01/M02 now remove the filing-date placement that replaced the
fixed EDGAR lag, M18 counts 365 days instead of the calendar anniversary).
X19-X24 are extra mutations showing that the step 9 fixes to vacuous tests
bite. A mutation whose site is no longer in the code is reported as
``SITE NOT FOUND``: update it when the code moves.

Usage (never on your working tree; the script refuses to):

    git clone --no-hardlinks . /tmp/mutation-clone
    python scripts/checks/mutation_study.py /tmp/mutation-clone [ID ...]

Set ``ONLY_TESTS=a.py,b.py`` to run only those test files for every
mutation (used to compare the old and new property tests).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FE = "src/processing/feature_engineering.py"
FE_TESTS = ["test_feature_engineering.py", "test_edgar_filing_timing_wp10.py",
            "test_price_features_wp2.py", "test_fred_pipeline_wp3.py",
            "test_v45_features.py", "test_integration.py", "test_v63_channel_mix_features.py",
            "test_fred_pgr_refresh_and_nan_live_features.py", "test_property_feature_engineering.py",
            "test_mutation_kills_wp12.py"]
WFO_TESTS = ["test_wfo_engine.py", "test_embargo_fix.py", "test_multi_benchmark_wfo.py",
             "test_property_wfo_temporal.py", "test_v74_guards.py", "test_elasticnet.py",
             "test_validation_gating_wp7.py", "test_integration.py", "test_mutation_kills_wp12.py"]
TR_TESTS = ["test_total_return.py", "test_drip_closed_form.py", "test_multi_total_return.py",
            "test_target_windows.py", "test_db_price_integrity.py", "test_integration.py",
            "test_property_return_calculations.py", "test_split_registry.py",
            "test_mutation_kills_wp12.py"]
CONS_TESTS = ["test_consensus_shadow.py", "test_monthly_summary.py", "test_monthly_pipeline_e2e.py",
              "test_research_v72_quality_weighted_consensus.py", "test_mutation_kills_wp12.py"]
DEC_TESTS = ["test_live_mapping_wp8.py", "test_validation_gating_wp7.py", "test_v813_recommendation_mode.py",
             "test_monthly_pipeline_e2e.py", "test_monte_carlo_tax.py", "test_data_freshness.py",
             "test_mutation_kills_wp12.py"]
CONF_TESTS = ["test_conformal.py", "test_validation_gating_wp7.py", "test_shadow_followon.py",
              "test_mutation_kills_wp12.py"]
TAX_TESTS = ["test_capital_gains.py", "test_stcg_boundary.py", "test_tax_wp8.py", "test_three_scenario_tax.py",
             "test_tlh.py", "test_property_tax_boundaries.py", "test_monte_carlo_tax.py",
             "test_mutation_kills_wp12.py"]

MUTATIONS = [
    ("M01_edgar_placement_removed", FE,
     "        edgar_raw = place_edgar_rows_by_filing_date(\n            edgar_raw, edgar_raw.get(\"filing_date\")\n        )",
     "        edgar_raw = edgar_raw.set_axis(_snap_to_business_month_end_index(edgar_raw.index))", FE_TESTS),
    ("M02_roe_placement_removed", FE,
     "            roe_placed = place_edgar_rows_by_filing_date(\n                roe_q,",
     "            roe_placed = roe_q if True else place_edgar_rows_by_filing_date(\n                roe_q,", FE_TESTS),
    ("M03_combined_ratio_ttm_window_3", FE,
     "df[\"combined_ratio_ttm\"] = cr_monthly.rolling(12, min_periods=6).mean()",
     "df[\"combined_ratio_ttm\"] = cr_monthly.rolling(3, min_periods=2).mean()", FE_TESTS),
    ("M04_bvps_yoy_pct_change_1", FE,
     "df[\"book_value_per_share_growth_yoy\"] = bvps_monthly.pct_change(\n                periods=12,",
     "df[\"book_value_per_share_growth_yoy\"] = bvps_monthly.pct_change(\n                periods=1,", FE_TESTS),
    ("M05_drip_value_over_price", "src/processing/total_return.py",
     "new_shares = (current_shares * evt_value) / div_price",
     "new_shares = evt_value / div_price", TR_TESTS),
    ("M06_targets_drop_splits", "src/processing/multi_total_return.py",
     "    splits = db_client.get_splits(conn, ticker)\n    if splits.empty:",
     "    splits = db_client.get_splits(conn, ticker)\n    if True:", TR_TESTS),
    ("M07_wfo_max_train_size_none", "src/models/wfo_engine.py",
     "max_train_size=config.WFO_TRAIN_WINDOW_MONTHS,",
     "max_train_size=None,", WFO_TESTS),
    ("M08_live_refit_full_history", "src/models/wfo_engine.py",
     "recent = aligned.iloc[-train_window_months:]",
     "recent = aligned", WFO_TESTS),
    ("M09_cpcv_purged_size_0", "src/models/wfo_engine.py",
     "purged_size=target_horizon_months,",
     "purged_size=0,", WFO_TESTS + ["test_cpcv.py"]),
    ("M10_consensus_lambda_swap", "src/models/consensus_shadow.py",
     "return ((1.0 - lambda_mix) * equal_weight + lambda_mix * normalized).rename(\"weight\")",
     "return (lambda_mix * equal_weight + (1.0 - lambda_mix) * normalized).rename(\"weight\")", CONS_TESTS),
    ("M11_consensus_ic_clip_removed", "src/models/consensus_shadow.py",
     "        .fillna(0.0)\n        .clip(lower=0.0)\n    )",
     "        .fillna(0.0)\n    )", CONS_TESTS),
    ("M12_consensus_lambda_clip_removed", "src/models/consensus_shadow.py",
     "lambda_mix = min(max(float(lambda_mix), 0.0), 1.0)",
     "lambda_mix = float(lambda_mix)", CONS_TESTS),
    ("M13_underperform_sells_75", "src/reporting/decision_rendering.py",
     "    if consensus == \"UNDERPERFORM\":\n        return 1.00",
     "    if consensus == \"UNDERPERFORM\":\n        return 0.75", DEC_TESTS),
    ("M14_cpcv_fail_gate_dropped", "src/reporting/decision_rendering.py",
     "status=GATE_PASS if cpcv_ok else GATE_FAIL,",
     "status=GATE_PASS,", DEC_TESTS),
    ("M15_conformal_no_finite_sample", "src/models/conformal.py",
     "return min(float(np.ceil((1.0 - alpha) * (n + 1)) / n), 1.0)",
     "return min(float(1.0 - alpha), 1.0)", CONF_TESTS),
    ("M16_aci_sign_flip", "src/models/conformal.py",
     "alpha_t = float(np.clip(alpha_t + gamma * (alpha_nominal - err_t), 0.01, 0.99))",
     "alpha_t = float(np.clip(alpha_t - gamma * (alpha_nominal - err_t), 0.01, 0.99))", CONF_TESTS),
    ("M17_relative_return_sign_flip", "src/processing/multi_total_return.py",
     "relative = aligned[\"pgr\"] - aligned[\"etf\"]",
     "relative = aligned[\"etf\"] - aligned[\"pgr\"]", TR_TESTS),
    ("M18_ltcg_365_days", "src/tax/capital_gains.py",
     "return vest_date + relativedelta(years=1) + timedelta(days=1)",
     "return vest_date + timedelta(days=365)", TAX_TESTS),
    # Extra mutations (not in F28): they show the step 9 test fixes bite.
    ("X19_fred_ffill_to_bfill_lookahead", FE,
     'fred_aligned = fred_macro.reindex(monthly_dates, method="ffill")',
     'fred_aligned = fred_macro.reindex(monthly_dates, method="bfill")',
     ["test_fred_features.py", "test_pgr_fred_features.py"]),
    ("X20_insurance_cpi_mom_1m", FE,
     'df["insurance_cpi_mom3m"] = ins_cpi.pct_change(\n                periods=3,',
     'df["insurance_cpi_mom3m"] = ins_cpi.pct_change(\n                periods=1,',
     ["test_fred_features.py", "test_pgr_fred_features.py"]),
    ("X21_vmt_yoy_1m", FE,
     'df["vmt_yoy"] = vmt.pct_change(periods=12, fill_method=None)',
     'df["vmt_yoy"] = vmt.pct_change(periods=1, fill_method=None)',
     ["test_fred_features.py", "test_pgr_fred_features.py"]),
    ("X22_cr_acceleration_diff_1", FE,
     'df["cr_acceleration"] = df["combined_ratio_ttm"].diff(3)',
     'df["cr_acceleration"] = df["combined_ratio_ttm"].diff(1)',
     ["test_v45_features.py"]),
    ("X23_valuation_ttm_avail_no_rolling_max", "src/processing/valuation_multiples.py",
     'ttm_avail = pd.to_datetime(\n        eps_avail_ns.rolling(12, min_periods=12).max(), unit="ns"\n    )',
     'ttm_avail = pd.to_datetime(eps_avail_ns, unit="ns")',
     ["test_valuation_multiples.py"]),
    ("X24_fracdiff_never_qualifies", FE,
     "if adf_pval < adf_alpha and abs(corr_val) >= corr_threshold:",
     "if False and adf_pval < adf_alpha and abs(corr_val) >= corr_threshold:",
     ["test_fracdiff.py"]),
]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    clone = Path(sys.argv[1]).resolve()
    if clone == REPO_ROOT or not (clone / ".git").exists():
        sys.exit("Pass a scratch git clone, not this working tree.")
    only = set(sys.argv[2:])
    only_tests = [t for t in os.environ.get("ONLY_TESTS", "").split(",") if t]
    results: dict[str, str] = {}
    for mid, rel, old, new, tests in MUTATIONS:
        if only and mid not in only:
            continue
        path = clone / rel
        src = path.read_text()
        if src.count(old) != 1:
            results[mid] = f"SITE NOT FOUND ({src.count(old)})"
            print(mid, results[mid], flush=True)
            continue
        path.write_text(src.replace(old, new))
        try:
            chosen = only_tests or list(dict.fromkeys(tests))
            files = [f"tests/{t}" for t in chosen if (clone / "tests" / t).exists()]
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-x", "-q", "-o", "addopts=--tb=line",
                 "-p", "no:cacheprovider", *files],
                cwd=clone, capture_output=True, text=True,
            )
            status = "KILLED" if proc.returncode != 0 else "SURVIVED"
            if status == "SURVIVED" and " skipped" in proc.stdout:
                status = "SURVIVED (with skips)"
            tail = [line for line in proc.stdout.splitlines() if line.strip()][-3:]
        finally:
            path.write_text(src)
        results[mid] = status
        print(mid, status, "|", " / ".join(tail)[:300], flush=True)
    print(json.dumps(results, indent=1))
    core = {k: v for k, v in results.items() if k.startswith("M")}
    print("F28 survivors:", sum(v.startswith("SURVIVED") for v in core.values()), "of", len(core))
    extra = {k: v for k, v in results.items() if k.startswith("X")}
    print("extra survivors:", sum(v.startswith("SURVIVED") for v in extra.values()), "of", len(extra))


if __name__ == "__main__":
    main()
