from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.pooled_benchmark_experiments import (
    build_pooled_relative_targets,
    summarize_pooled_experiments,
)


def test_build_pooled_relative_targets_averages_available_members():
    conn = MagicMock()

    def _fake_loader(_conn, ticker, horizon):
        idx = pd.date_range("2024-01-31", periods=2, freq="ME")
        data = {
            "AAA": pd.Series([0.10, 0.20], index=idx),
            "BBB": pd.Series([0.30, 0.50], index=idx),
        }
        return data.get(ticker, pd.Series(dtype=float))

    with patch("scripts.pooled_benchmark_experiments.load_relative_return_matrix", side_effect=_fake_loader):
        pooled = build_pooled_relative_targets(
            conn=conn,
            pool_definitions={"pooled": ["AAA", "BBB"], "empty": ["ZZZ"]},
            target_horizon_months=6,
        )

    assert list(pooled.columns) == ["pooled"]
    assert pooled.iloc[0, 0] == 0.20
    assert pooled.iloc[1, 0] == 0.35


def test_summarize_pooled_experiments_groups_rows():
    detail = pd.DataFrame(
        [
            {
                "pooled_target": "broad_equity",
                "item_type": "model",
                "item_name": "elasticnet",
                "horizon_months": 6,
                "ic": 0.10,
                "hit_rate": 0.55,
                "oos_r2": -0.10,
                "mae": 0.20,
                "gate_status": "FAIL",
            },
            {
                "pooled_target": "broad_equity",
                "item_type": "model",
                "item_name": "elasticnet",
                "horizon_months": 6,
                "ic": 0.20,
                "hit_rate": 0.65,
                "oos_r2": 0.00,
                "mae": 0.10,
                "gate_status": "MARGINAL",
            },
        ]
    )
    summary = summarize_pooled_experiments(detail)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["mean_ic"] == pytest.approx(0.15)
    assert row["mean_hit_rate"] == pytest.approx(0.60)
