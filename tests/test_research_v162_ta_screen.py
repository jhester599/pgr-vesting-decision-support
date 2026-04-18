"""Tests for the v162 TA broad-screen helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_candidate_inventory_contains_expected_alpha_vantage_families() -> None:
    from results.research.v162_ta_broad_screen import build_candidate_inventory

    inventory = build_candidate_inventory(["VOO", "BND"])
    families = set(inventory["family"])

    assert {"momentum", "trend", "volatility", "regime", "volume"}.issubset(families)
    assert "ta_ratio_rsi_6m_voo" in set(inventory["feature"])
    assert "ta_bnd_macd_hist_norm" in set(inventory["feature"])


def test_excluded_noisy_families_are_not_model_candidates_by_default() -> None:
    from results.research.v162_ta_broad_screen import build_candidate_inventory

    inventory = build_candidate_inventory(["VOO"])
    model_inventory = inventory.loc[inventory["include_in_model"]]

    assert not model_inventory["family"].isin({"pattern", "hilbert", "price_transform"}).any()
    assert inventory.loc[inventory["family"] == "pattern", "include_in_model"].eq(False).all()


def test_add_and_replacement_modes_generate_distinct_feature_sets() -> None:
    from results.research.v162_ta_broad_screen import build_feature_set_specs

    specs = build_feature_set_specs(
        baseline_features=["mom_6m", "vol_63d", "vix"],
        candidate_feature="ta_ratio_roc_6m_voo",
        redundant_features=["mom_6m", "vol_63d"],
    )
    by_mode = {spec["experiment_mode"]: spec["feature_columns"] for spec in specs}

    assert by_mode["baseline"] == ["mom_6m", "vol_63d", "vix"]
    assert by_mode["add_feature"] == ["mom_6m", "vol_63d", "vix", "ta_ratio_roc_6m_voo"]
    assert by_mode["replace_mom_6m"] == ["vol_63d", "vix", "ta_ratio_roc_6m_voo"]
    assert by_mode["replace_vol_63d"] == ["mom_6m", "vix", "ta_ratio_roc_6m_voo"]


def test_regression_and_classification_records_contain_baseline_deltas() -> None:
    from results.research.v162_ta_broad_screen import attach_baseline_deltas

    records = pd.DataFrame(
        [
            {
                "model_family": "regression",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "baseline",
                "feature": "__baseline__",
                "ic": 0.10,
                "oos_r2": -0.02,
            },
            {
                "model_family": "regression",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "add_feature",
                "feature": "ta_ratio_roc_6m_voo",
                "ic": 0.14,
                "oos_r2": -0.01,
            },
            {
                "model_family": "classification",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "baseline",
                "feature": "__baseline__",
                "balanced_accuracy": 0.50,
                "brier_score": 0.25,
            },
            {
                "model_family": "classification",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "add_feature",
                "feature": "ta_ratio_roc_6m_voo",
                "balanced_accuracy": 0.56,
                "brier_score": 0.23,
            },
        ]
    )
    with_deltas = attach_baseline_deltas(records)
    reg_row = with_deltas.loc[
        (with_deltas["model_family"] == "regression")
        & (with_deltas["experiment_mode"] == "add_feature")
    ].iloc[0]
    cls_row = with_deltas.loc[
        (with_deltas["model_family"] == "classification")
        & (with_deltas["experiment_mode"] == "add_feature")
    ].iloc[0]

    assert abs(float(reg_row["delta_ic"]) - 0.04) < 1e-12
    assert abs(float(reg_row["delta_oos_r2"]) - 0.01) < 1e-12
    assert abs(float(cls_row["delta_balanced_accuracy"]) - 0.06) < 1e-12
    assert abs(float(cls_row["delta_brier_score"]) + 0.02) < 1e-12


def test_duplicate_baselines_do_not_expand_delta_records() -> None:
    from results.research.v162_ta_broad_screen import attach_baseline_deltas

    records = pd.DataFrame(
        [
            {
                "model_family": "regression",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "baseline",
                "feature": "__baseline__",
                "ic": 0.10,
            },
            {
                "model_family": "regression",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "baseline",
                "feature": "__baseline__",
                "ic": 0.10,
            },
            {
                "model_family": "regression",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "add_feature",
                "feature": "ta_ratio_roc_6m_voo",
                "ic": 0.12,
            },
        ]
    )

    with_deltas = attach_baseline_deltas(records)

    assert len(with_deltas) == len(records)
    assert abs(
        with_deltas.loc[
            with_deltas["experiment_mode"] == "add_feature",
            "delta_ic",
        ].iloc[0]
        - 0.02
    ) < 1e-12


def test_classification_positive_count_uses_ba_and_brier() -> None:
    from results.research.v162_ta_broad_screen import summarize_screen

    records = pd.DataFrame(
        [
            {
                "model_family": "classification",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "add_feature",
                "feature": "ta_pgr_obv_detrended",
                "feature_group": "pgr_volume",
                "family": "volume",
                "delta_balanced_accuracy": 0.01,
                "delta_brier_score": 0.02,
            },
            {
                "model_family": "classification",
                "benchmark": "BND",
                "model_type": "ridge",
                "experiment_mode": "add_feature",
                "feature": "ta_pgr_obv_detrended",
                "feature_group": "pgr_volume",
                "family": "volume",
                "delta_balanced_accuracy": -0.01,
                "delta_brier_score": -0.02,
            },
            {
                "model_family": "classification",
                "benchmark": "GLD",
                "model_type": "ridge",
                "experiment_mode": "add_feature",
                "feature": "ta_pgr_obv_detrended",
                "feature_group": "pgr_volume",
                "family": "volume",
                "delta_balanced_accuracy": -0.01,
                "delta_brier_score": 0.02,
            },
        ]
    )

    summary = summarize_screen(records).iloc[0]

    assert summary["positive_benchmark_count"] == 2
    assert summary["feature_group"] == "pgr_volume"


def test_compact_detail_records_removes_stale_expansion() -> None:
    from results.research.v162_ta_broad_screen import compact_detail_records

    records = pd.DataFrame(
        [
            {
                "model_family": "regression",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "baseline",
                "feature": "__baseline__",
                "ic": 0.10,
                "baseline_ic": 0.10,
                "delta_ic": 0.00,
            },
            {
                "model_family": "regression",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "baseline",
                "feature": "__baseline__",
                "ic": 0.10,
                "baseline_ic": 0.10,
                "delta_ic": 0.00,
            },
            {
                "model_family": "regression",
                "benchmark": "VOO",
                "model_type": "ridge",
                "experiment_mode": "add_feature",
                "feature": "ta_ratio_roc_6m_voo",
                "ic": 0.12,
                "baseline_ic": 999.00,
                "delta_ic": 999.00,
            },
        ]
    )

    compact = compact_detail_records(records, benchmarks=("VOO",))

    assert len(compact) == 2
    assert "feature_group" in compact.columns
    assert abs(
        compact.loc[
            compact["experiment_mode"] == "add_feature",
            "delta_ic",
        ].iloc[0]
        - 0.02
    ) < 1e-12


def test_broad_screen_script_can_run_from_file_path() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "results/research/v162_ta_broad_screen.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run v162 broad TA feature screen" in result.stdout
