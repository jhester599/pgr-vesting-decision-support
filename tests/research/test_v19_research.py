"""Tests for the v19 public-macro and traceability helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.research.v19 import (
    BLOCKED_FEATURE_REASONS,
    build_v19_traceability_matrix,
    fetch_bls_series,
    fetch_fredgraph_series,
    fetch_multpl_series,
)


def _mock_text_response(text: str) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.text = text
    response.raise_for_status = MagicMock()
    return response


def _mock_json_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


def test_fetch_fredgraph_series_parses_monthly_csv() -> None:
    csv_text = "\n".join(
        [
            "observation_date,DTWEXBGS",
            "2024-01-15,100.0",
            "2024-01-31,101.0",
            "2024-02-29,102.0",
        ]
    )
    with patch("src.research.v19.requests.get", return_value=_mock_text_response(csv_text)):
        df = fetch_fredgraph_series("DTWEXBGS", observation_start="2024-01-01")

    assert list(df.columns) == ["DTWEXBGS"]
    assert len(df) == 2
    assert float(df.loc["2024-01-31", "DTWEXBGS"]) == 101.0


def test_fetch_bls_series_parses_monthly_payload() -> None:
    payload = {
        "Results": {
            "series": [
                {
                    "data": [
                        {"year": "2024", "period": "M02", "value": "112.5"},
                        {"year": "2024", "period": "M01", "value": "110.0"},
                        {"year": "2024", "period": "M13", "value": "999.0"},
                    ]
                }
            ]
        }
    }
    with patch("src.research.v19.requests.get", return_value=_mock_json_response(payload)):
        df = fetch_bls_series("CUSR0000SETE", observation_start="2024-01-01")

    assert list(df.columns) == ["CUSR0000SETE"]
    assert len(df) == 2
    assert float(df.loc["2024-02-29", "CUSR0000SETE"]) == 112.5


def test_fetch_multpl_series_parses_table() -> None:
    table = pd.DataFrame(
        {
            "Date": ["Feb 1, 2024", "Jan 1, 2024"],
            "Value": ["3.40%", "3.20%"],
        }
    )
    with patch("src.research.v19.pd.read_html", return_value=[table]):
        df = fetch_multpl_series(
            slug="s-p-500-earnings-yield",
            series_id="SP500_EARNINGS_YIELD_MULTPL",
            observation_start="2024-01-01",
        )

    assert list(df.columns) == ["SP500_EARNINGS_YIELD_MULTPL"]
    assert len(df) == 2
    assert float(df.loc["2024-02-29", "SP500_EARNINGS_YIELD_MULTPL"]) == 3.4


def test_traceability_marks_tested_and_blocked_features() -> None:
    inventory = pd.DataFrame(
        [
            {
                "feature_name": "usd_broad_return_3m",
                "category": "benchmark_predictive",
                "replace_or_compete_with": "nfci",
                "definition": "",
                "economic_rationale": "",
                "expected_direction": "",
                "likely_frequency": "",
                "likely_source": "",
                "implementation_difficulty": "",
                "likely_signal_quality": "",
                "why_it_might_outperform_existing_feature": "",
                "key_risks": "",
                "target_model": "both",
                "priority_rank": 1,
                "research_source": "test",
                "status": "queued",
                "notes": "",
            },
            {
                "feature_name": "pgr_cr_vs_peer_cr",
                "category": "PGR_specific",
                "replace_or_compete_with": "combined_ratio_ttm",
                "definition": "",
                "economic_rationale": "",
                "expected_direction": "",
                "likely_frequency": "",
                "likely_source": "",
                "implementation_difficulty": "",
                "likely_signal_quality": "",
                "why_it_might_outperform_existing_feature": "",
                "key_risks": "",
                "target_model": "gbt",
                "priority_rank": 2,
                "research_source": "test",
                "status": "queued",
                "notes": "",
            },
        ]
    )
    phase0 = pd.DataFrame(
        [
            {
                "candidate_name": "ridge_lean_v1",
                "model_type": "ridge",
                "candidate_feature": "usd_broad_return_3m",
                "replace_feature": "nfci",
                "mean_policy_return_sign_delta": 0.01,
                "mean_oos_r2_delta": 0.02,
                "mean_ic_delta": 0.03,
            }
        ]
    )

    trace = build_v19_traceability_matrix(
        inventory,
        feature_columns={"usd_broad_return_3m"},
        phase0_summary=phase0,
        blocked_reasons=BLOCKED_FEATURE_REASONS,
    )

    tested = trace[trace["feature_name"] == "usd_broad_return_3m"].iloc[0]
    blocked = trace[trace["feature_name"] == "pgr_cr_vs_peer_cr"].iloc[0]

    assert tested["evaluation_status"] == "tested"
    assert tested["best_model"] == "ridge_lean_v1"
    assert blocked["evaluation_status"] == "blocked"
