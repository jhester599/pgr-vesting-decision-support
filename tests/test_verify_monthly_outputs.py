from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.database import db_client


REQUIRED_MONTHLY_FILES = [
    "recommendation.md",
    "diagnostic.md",
    "signals.csv",
    "benchmark_quality.csv",
    "consensus_shadow.csv",
    "classification_shadow.csv",
    "decision_overlays.csv",
    "dashboard.html",
    "monthly_summary.json",
    "run_manifest.json",
]


def _write_required_files(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_MONTHLY_FILES:
        path = out_dir / filename
        if filename.endswith(".json"):
            path.write_text("{}", encoding="utf-8")
        elif filename.endswith(".csv"):
            path.write_text("placeholder\n", encoding="utf-8")
        else:
            path.write_text("placeholder\n", encoding="utf-8")


def _write_valid_monthly_fixture(base_dir: Path, db_path: Path) -> None:
    out_dir = base_dir / "2026-04"
    _write_required_files(out_dir)
    monthly_summary = {
        "as_of_date": "2026-04-18",
        "run_date": "2026-04-18",
        "schema_version": 3,
        "classification_shadow_variants": [
            {"variant": "baseline_shadow", "reporting_only": False},
            {
                "variant": "ta_minimal_replacement",
                "reporting_only": True,
                "probability_actionable_sell": 0.39,
            },
            {
                "variant": "ta_minimal_plus_vwo_pct_b",
                "reporting_only": True,
                "probability_actionable_sell": 0.41,
            },
        ],
    }
    run_manifest = {
        "schema_version": "003_model_retrain_log",
        "git_sha": "abc123",
        "warnings": [],
    }
    (out_dir / "monthly_summary.json").write_text(
        json.dumps(monthly_summary),
        encoding="utf-8",
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "variant": [
                "baseline_shadow",
                "ta_minimal_replacement",
                "ta_minimal_plus_vwo_pct_b",
            ],
            "benchmark": ["VOO", "VOO", "VOO"],
            "classifier_prob_actionable_sell": [0.36, 0.39, 0.41],
        }
    ).to_csv(out_dir / "classification_shadow.csv", index=False)
    pd.DataFrame(
        {
            "as_of_date": ["2026-04-18", "2026-04-18"],
            "run_date": ["2026-04-18", "2026-04-18"],
            "variant": ["ta_minimal_replacement", "ta_minimal_plus_vwo_pct_b"],
            "feature_anchor_date": ["2026-03-31", "2026-03-31"],
            "forecast_horizon_months": [6, 6],
            "mature_on_date": ["2026-09-30", "2026-09-30"],
            "is_horizon_mature": [False, False],
            "probability_actionable_sell": [0.39, 0.41],
            "stance": ["NEUTRAL", "NEUTRAL"],
            "confidence_tier": ["MODERATE", "LOW"],
            "benchmark_count": [10, 10],
            "actual_actionable_sell": [None, None],
            "actual_basket_relative_return": [None, None],
        }
    ).to_csv(base_dir / "ta_shadow_variant_history.csv", index=False)

    conn = db_client.get_connection(str(db_path))
    db_client.initialize_schema(conn)
    db_client.upsert_prices(
        conn,
        [{"ticker": "PGR", "date": "2026-04-17", "close": 250.0}],
    )
    db_client.upsert_fred_macro(
        conn,
        [{"series_id": "T10Y2Y", "month_end": "2026-03-31", "value": 0.5}],
    )
    db_client.upsert_pgr_edgar_monthly(
        conn,
        [{"month_end": "2026-02-28", "combined_ratio": 85.7}],
    )
    conn.close()


def test_verify_monthly_outputs_accepts_complete_artifacts(tmp_path: Path) -> None:
    from scripts.verify_monthly_outputs import verify_monthly_outputs

    base_dir = tmp_path / "monthly_decisions"
    db_path = tmp_path / "pgr.db"
    _write_valid_monthly_fixture(base_dir, db_path)

    result = verify_monthly_outputs(
        base_dir=base_dir,
        db_path=db_path,
        as_of_date="2026-04-18",
    )

    assert result.output_dir == base_dir / "2026-04"
    assert result.ta_variants == [
        "ta_minimal_plus_vwo_pct_b",
        "ta_minimal_replacement",
    ]
    assert result.freshness_status == "OK"
    assert "TA shadow variants: `2`" in "\n".join(result.summary_lines())


def test_verify_monthly_outputs_fails_when_ta_ledger_row_missing(
    tmp_path: Path,
) -> None:
    from scripts.verify_monthly_outputs import verify_monthly_outputs

    base_dir = tmp_path / "monthly_decisions"
    db_path = tmp_path / "pgr.db"
    _write_valid_monthly_fixture(base_dir, db_path)
    pd.DataFrame(
        {
            "as_of_date": ["2026-04-18"],
            "variant": ["ta_minimal_replacement"],
        }
    ).to_csv(base_dir / "ta_shadow_variant_history.csv", index=False)

    with pytest.raises(SystemExit, match="missing TA ledger rows"):
        verify_monthly_outputs(
            base_dir=base_dir,
            db_path=db_path,
            as_of_date="2026-04-18",
        )


def test_verify_monthly_outputs_fails_when_data_freshness_warns(
    tmp_path: Path,
) -> None:
    from scripts.verify_monthly_outputs import verify_monthly_outputs

    base_dir = tmp_path / "monthly_decisions"
    db_path = tmp_path / "pgr.db"
    _write_valid_monthly_fixture(base_dir, db_path)

    with pytest.raises(SystemExit, match="Data freshness warnings"):
        verify_monthly_outputs(
            base_dir=base_dir,
            db_path=db_path,
            as_of_date="2026-04-18",
            reference_date="2026-04-26",
        )
