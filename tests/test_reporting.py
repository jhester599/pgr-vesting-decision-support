"""
Tests for src/reporting/backtest_report.py and the v2 plot functions in
src/visualization/plots.py.

All plot tests use tmp_path so no PNG files accumulate in the project.
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtest_engine import BacktestEventResult
from src.backtest.vesting_events import VestingEvent, _add_months
from src.reporting.backtest_report import (
    export_backtest_to_csv,
    generate_backtest_table,
    generate_correct_direction_table,
    generate_prediction_table,
    print_backtest_summary,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_result(
    year: int,
    month: int,
    rsu_type: str,
    benchmark: str,
    horizon: int,
    predicted: float,
    realized: float,
) -> BacktestEventResult:
    event_date = date(year, month, 19 if month == 1 else 17)
    event = VestingEvent(
        event_date=event_date,
        rsu_type=rsu_type,
        horizon_6m_end=_add_months(event_date, 6),
        horizon_12m_end=_add_months(event_date, 12),
    )
    signal = "OUTPERFORM" if predicted >= 0 else "UNDERPERFORM"
    real_dir = "OUTPERFORM" if realized >= 0 else "UNDERPERFORM"
    return BacktestEventResult(
        event=event,
        benchmark=benchmark,
        target_horizon=horizon,
        predicted_relative_return=predicted,
        realized_relative_return=realized,
        signal_direction=signal,
        correct_direction=(signal == real_dir),
        predicted_sell_pct=0.50,
        ic_at_event=0.10,
        hit_rate_at_event=0.60,
        n_train_observations=120,
        proxy_fill_fraction=0.0,
    )


@pytest.fixture()
def sample_results() -> list[BacktestEventResult]:
    """12 results: 2 events × 3 benchmarks × 2 horizons."""
    results = []
    events = [
        (2020, 1, "time"),
        (2020, 7, "performance"),
    ]
    benchmarks = ["VTI", "BND", "VGT"]
    for year, month, rsu_type in events:
        for bench in benchmarks:
            for horizon, pred, real in [(6, 0.05, 0.03), (12, -0.02, -0.01)]:
                results.append(
                    _make_result(year, month, rsu_type, bench, horizon, pred, real)
                )
    return results


# ---------------------------------------------------------------------------
# Table generation tests
# ---------------------------------------------------------------------------

class TestGenerateBacktestTable:
    def test_returns_dataframe(self, sample_results):
        df = generate_backtest_table(sample_results, horizon=6)
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self, sample_results):
        df = generate_backtest_table(sample_results, horizon=6)
        # 2 events × 3 benchmarks
        assert df.shape == (2, 3)

    def test_columns_are_benchmark_tickers(self, sample_results):
        df = generate_backtest_table(sample_results, horizon=6)
        assert set(df.columns) == {"VTI", "BND", "VGT"}

    def test_index_is_event_dates(self, sample_results):
        df = generate_backtest_table(sample_results, horizon=6)
        assert all(isinstance(d, date) for d in df.index)

    def test_empty_for_missing_horizon(self, sample_results):
        df = generate_backtest_table(sample_results, horizon=99)
        assert df.empty

    def test_values_are_realized_returns(self, sample_results):
        df = generate_backtest_table(sample_results, horizon=6)
        # All 6M realized values in fixture are 0.03
        assert (df.values[~np.isnan(df.values.astype(float))] == pytest.approx(0.03))

    def test_horizon_filter_works(self, sample_results):
        df6 = generate_backtest_table(sample_results, horizon=6)
        df12 = generate_backtest_table(sample_results, horizon=12)
        # Different realized values for different horizons
        assert not np.allclose(
            df6.values.astype(float), df12.values.astype(float), equal_nan=True
        )


class TestGeneratePredictionTable:
    def test_shape_matches_realized_table(self, sample_results):
        pred = generate_prediction_table(sample_results, horizon=6)
        real = generate_backtest_table(sample_results, horizon=6)
        assert pred.shape == real.shape

    def test_values_are_predictions(self, sample_results):
        pred = generate_prediction_table(sample_results, horizon=6)
        # All 6M predicted values in fixture are 0.05
        assert (pred.values[~np.isnan(pred.values.astype(float))] == pytest.approx(0.05))


class TestGenerateCorrectDirectionTable:
    def test_dtype_is_bool(self, sample_results):
        df = generate_correct_direction_table(sample_results, horizon=6)
        assert df.dtypes.apply(lambda t: t == bool or t == object).all()

    def test_correct_direction_matches_logic(self, sample_results):
        df = generate_correct_direction_table(sample_results, horizon=6)
        # 6M: predicted=0.05 (OUTPERFORM), realized=0.03 (OUTPERFORM) → True
        assert df.values.astype(bool).all()


# ---------------------------------------------------------------------------
# print_backtest_summary tests
# ---------------------------------------------------------------------------

class TestPrintBacktestSummary:
    def test_prints_without_error(self, sample_results, capsys):
        print_backtest_summary(sample_results)
        captured = capsys.readouterr()
        assert "BACKTEST SUMMARY" in captured.out

    def test_prints_hit_rate(self, sample_results, capsys):
        print_backtest_summary(sample_results)
        captured = capsys.readouterr()
        assert "hit rate" in captured.out.lower()

    def test_prints_by_horizon(self, sample_results, capsys):
        print_backtest_summary(sample_results)
        captured = capsys.readouterr()
        assert "6M" in captured.out or "12M" in captured.out

    def test_empty_results_does_not_crash(self, capsys):
        print_backtest_summary([])
        captured = capsys.readouterr()
        assert "No backtest results" in captured.out


# ---------------------------------------------------------------------------
# export_backtest_to_csv tests
# ---------------------------------------------------------------------------

class TestExportBacktestToCsv:
    def test_creates_csv_file(self, sample_results, tmp_path):
        path = str(tmp_path / "backtest.csv")
        export_backtest_to_csv(sample_results, path)
        assert os.path.exists(path)

    def test_csv_has_expected_columns(self, sample_results, tmp_path):
        path = str(tmp_path / "backtest.csv")
        export_backtest_to_csv(sample_results, path)
        df = pd.read_csv(path)
        expected = {
            "event_date", "rsu_type", "year", "benchmark", "target_horizon",
            "predicted_relative_return", "realized_relative_return",
            "signal_direction", "correct_direction", "predicted_sell_pct",
            "ic_at_event", "hit_rate_at_event", "n_train_observations",
            "proxy_fill_fraction",
        }
        assert expected.issubset(set(df.columns))

    def test_row_count_matches(self, sample_results, tmp_path):
        path = str(tmp_path / "backtest.csv")
        export_backtest_to_csv(sample_results, path)
        df = pd.read_csv(path)
        assert len(df) == len(sample_results)

    def test_empty_results_does_not_create_file(self, tmp_path):
        path = str(tmp_path / "empty.csv")
        export_backtest_to_csv([], path)
        assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# Plot function tests
# ---------------------------------------------------------------------------

class TestPlotBacktestHeatmap:
    def test_creates_png(self, sample_results, tmp_path, monkeypatch):
        import src.visualization.plots as plots_mod
        monkeypatch.setattr(plots_mod, "_PLOTS_DIR", str(tmp_path))
        path = plots_mod.plot_backtest_heatmap(sample_results, horizon=6)
        assert path != ""
        assert os.path.exists(path)

    def test_returns_empty_string_for_missing_horizon(self, sample_results, tmp_path, monkeypatch):
        import src.visualization.plots as plots_mod
        monkeypatch.setattr(plots_mod, "_PLOTS_DIR", str(tmp_path))
        path = plots_mod.plot_backtest_heatmap(sample_results, horizon=99)
        assert path == ""


class TestPlotHitRateByBenchmark:
    def test_creates_png(self, sample_results, tmp_path, monkeypatch):
        import src.visualization.plots as plots_mod
        monkeypatch.setattr(plots_mod, "_PLOTS_DIR", str(tmp_path))
        path = plots_mod.plot_hit_rate_by_benchmark(sample_results)
        assert path != ""
        assert os.path.exists(path)

    def test_returns_empty_string_on_no_data(self, tmp_path, monkeypatch):
        import src.visualization.plots as plots_mod
        monkeypatch.setattr(plots_mod, "_PLOTS_DIR", str(tmp_path))
        path = plots_mod.plot_hit_rate_by_benchmark([])
        assert path == ""


class TestPlotPredictedVsRealizedScatter:
    def test_creates_png_for_valid_benchmark(self, sample_results, tmp_path, monkeypatch):
        import src.visualization.plots as plots_mod
        monkeypatch.setattr(plots_mod, "_PLOTS_DIR", str(tmp_path))
        path = plots_mod.plot_predicted_vs_realized_scatter(sample_results, "VTI")
        assert path != ""
        assert os.path.exists(path)

    def test_returns_empty_for_unknown_benchmark(self, sample_results, tmp_path, monkeypatch):
        import src.visualization.plots as plots_mod
        monkeypatch.setattr(plots_mod, "_PLOTS_DIR", str(tmp_path))
        path = plots_mod.plot_predicted_vs_realized_scatter(sample_results, "UNKNOWN")
        assert path == ""


class TestPlotMultiBenchmarkSignals:
    def test_creates_png(self, tmp_path, monkeypatch):
        import src.visualization.plots as plots_mod
        monkeypatch.setattr(plots_mod, "_PLOTS_DIR", str(tmp_path))
        signals = pd.DataFrame(
            {
                "predicted_relative_return": [0.05, -0.03, 0.0],
                "signal": ["OUTPERFORM", "UNDERPERFORM", "NEUTRAL"],
                "ic": [0.12, 0.08, 0.04],
                "hit_rate": [0.60, 0.55, 0.50],
                "top_feature": ["mom_6m", "vol_21d", "pe_ratio"],
            },
            index=pd.Index(["VTI", "BND", "VGT"], name="benchmark"),
        )
        path = plots_mod.plot_multi_benchmark_signals(signals)
        assert path != ""
        assert os.path.exists(path)

    def test_returns_empty_for_empty_df(self, tmp_path, monkeypatch):
        import src.visualization.plots as plots_mod
        monkeypatch.setattr(plots_mod, "_PLOTS_DIR", str(tmp_path))
        path = plots_mod.plot_multi_benchmark_signals(pd.DataFrame())
        assert path == ""
