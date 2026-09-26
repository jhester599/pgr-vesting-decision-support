"""
Tests for v7.3 — Monthly Report Tax Section + Decision Log Fix.

Covers:
  _build_tax_context_lines() — 8 tests
  _append_decision_log() fix — 5 tests
  Total: 13 tests
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.monthly_decision import _append_decision_log, _build_tax_context_lines


# ---------------------------------------------------------------------------
# _build_tax_context_lines() tests
# ---------------------------------------------------------------------------

LOG_HEADER = (
    "| As-Of Date | Run Date | Consensus Signal | Sell % "
    "| Predicted 6M Return | Mean IC | Hit Rate | Notes |"
)
LOG_SEP = "|------------|----------|-----------------|--------|---------------------|---------|----------|-------|"


class TestTaxContextLines:

    def test_returns_list_of_strings(self):
        lines = _build_tax_context_lines(0.04, 0.58)
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_contains_tax_context_header(self):
        lines = _build_tax_context_lines(0.04, 0.58)
        assert "## Tax Context" in lines

    def test_breakeven_row_present(self):
        """The LTCG breakeven is an absolute PGR decline (review F19: sign fixed)."""
        lines = _build_tax_context_lines(0.04, 0.58, stcg_rate=0.37, ltcg_rate=0.20)
        combined = "\n".join(lines)
        # -g (S - L) / (1 - L) with g = 1: -(0.37 - 0.20) / 0.80 = -21.25 %
        assert "-21.25%" in combined
        # g = 0.5: -10.625 % (printed to two decimals)
        assert "| LTCG breakeven PGR return (gain = half the price) | -10.6" in combined

    def test_hold_wins_unless_pgr_falls_verdict(self):
        """Holding to LTCG wins unless PGR's own price falls by more than the breakeven."""
        lines = _build_tax_context_lines(0.04, 0.58, stcg_rate=0.37, ltcg_rate=0.20)
        combined = "\n".join(lines)
        assert "unless PGR's own price falls by more than 21.25%" in combined

    def test_relative_forecast_is_not_compared_with_the_breakeven(self):
        """A large relative forecast no longer triggers 'immediate sale may be warranted'."""
        lines = _build_tax_context_lines(0.30, 0.75, stcg_rate=0.37, ltcg_rate=0.20)
        combined = "\n".join(lines)
        assert "EXCEEDS" not in combined
        assert "relative forecast, not a PGR price forecast" in combined

    def test_negative_prediction_loss_harvest_verdict(self):
        """Negative relative forecast: diversification note, not 'capital-loss harvesting'."""
        lines = _build_tax_context_lines(-0.10, 0.30, stcg_rate=0.37, ltcg_rate=0.20)
        combined = "\n".join(lines)
        assert "capital-loss harvesting" not in combined.lower()
        assert "not a tax loss" in combined
        assert "wash sale" in combined

    def test_custom_rates_reflected(self):
        """Custom STCG/LTCG rates appear in the table."""
        lines = _build_tax_context_lines(0.04, 0.58, stcg_rate=0.40, ltcg_rate=0.15)
        combined = "\n".join(lines)
        assert "40%" in combined
        assert "15%" in combined

    def test_next_vest_dates_present(self):
        """Next vest dates (from config) appear in the output."""
        lines = _build_tax_context_lines(0.04, 0.58)
        combined = "\n".join(lines)
        assert "Next time-based vest" in combined
        assert "Next performance vest" in combined

    def test_prob_outperform_in_table(self):
        """P(outperform) value appears in the output table."""
        lines = _build_tax_context_lines(0.04, 0.62)
        combined = "\n".join(lines)
        assert "62.0%" in combined or "62%" in combined


# ---------------------------------------------------------------------------
# _append_decision_log() fix tests
# ---------------------------------------------------------------------------

def _make_log_file(tmp_path: Path, extra_rows: list[str] | None = None) -> Path:
    """Create a minimal decision_log.md in tmp_path."""
    rows = extra_rows or []
    rows_text = "\n".join(rows)
    content = f"""# PGR Monthly Decision Log

---

## Log

{LOG_HEADER}
{LOG_SEP}
{rows_text}

---

## Column Definitions

| Column | Description |
|--------|-------------|
| As-Of Date | The date the model uses as its information cutoff |
| Notes | Any anomalies, data gaps, or model warnings for this run |

---

## Interpreting the Recommendation

Some text here.
"""
    log_path = tmp_path / "decision_log.md"
    log_path.write_text(content, encoding="utf-8")
    return log_path


def _append(log_path: Path, as_of: date | None = None, **kwargs) -> None:
    """Test helper: call _append_decision_log with a path override."""
    defaults = dict(
        as_of=as_of or date(2026, 4, 20),
        run_date=date(2026, 4, 20),
        consensus="NEUTRAL",
        sell_pct=0.50,
        mean_predicted=0.03,
        mean_ic=0.042,
        mean_hr=0.55,
        dry_run=False,
        _log_path_override=log_path,
    )
    defaults.update(kwargs)
    _append_decision_log(**defaults)


class TestAppendDecisionLog:

    def test_new_row_appended_inside_log_table(self, tmp_path):
        """New row is inserted after the last log-table row, before Column Definitions."""
        log_path = _make_log_file(tmp_path, extra_rows=[
            "| 2026-03-20 | 2026-03-20 | NEUTRAL | 50% | +1.30% | -0.0056 | 53.0% |  |",
        ])
        _append(log_path)

        content = log_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        col_def_idx = next(i for i, l in enumerate(lines) if "## Column Definitions" in l)
        new_row_idx = next(
            (i for i, l in enumerate(lines) if "2026-04-20" in l),
            -1,
        )
        assert new_row_idx != -1, "New row not found in log"
        assert new_row_idx < col_def_idx, (
            f"New row at line {new_row_idx} is after Column Definitions at {col_def_idx}"
        )

    def test_new_row_not_duplicated(self, tmp_path):
        """Calling append twice with identical data only records one row."""
        log_path = _make_log_file(tmp_path)
        _append(log_path)
        _append(log_path)
        content = log_path.read_text(encoding="utf-8")
        # Count lines that look like log data rows containing our date.
        matching_lines = [l for l in content.splitlines()
                          if "2026-04-20" in l and l.strip().startswith("|")]
        assert len(matching_lines) == 1, (
            f"Expected 1 de-duplicated row with '2026-04-20', found {len(matching_lines)}"
        )

    def test_dry_run_duplicate_skipped_when_non_dry_row_already_exists(self, tmp_path):
        """A dry-run rerun should not append a second row with identical metrics."""
        log_path = _make_log_file(tmp_path)
        _append(log_path, dry_run=False)
        _append(log_path, dry_run=True)
        matching_lines = [
            line for line in log_path.read_text(encoding="utf-8").splitlines()
            if "2026-04-20" in line and line.strip().startswith("|")
        ]
        assert len(matching_lines) == 1

    def test_dry_run_flag_written(self, tmp_path):
        """[DRY RUN] label appears in the appended row when dry_run=True."""
        log_path = _make_log_file(tmp_path)
        _append(log_path, consensus="OUTPERFORM", sell_pct=0.25,
                mean_predicted=0.06, mean_ic=0.08, mean_hr=0.60, dry_run=True)
        content = log_path.read_text(encoding="utf-8")
        assert "[DRY RUN]" in content

    def test_column_definitions_not_corrupted(self, tmp_path):
        """After append, the Column Definitions table structure is intact."""
        log_path = _make_log_file(tmp_path)
        _append(log_path)
        content = log_path.read_text(encoding="utf-8")
        assert "## Column Definitions" in content
        assert "| As-Of Date | The date the model" in content

    def test_missing_log_file_is_noop(self, tmp_path):
        """If decision_log.md doesn't exist, no exception is raised."""
        nonexistent = tmp_path / "nonexistent_log.md"
        # Should not raise even though file doesn't exist
        _append_decision_log(
            as_of=date(2026, 4, 20),
            run_date=date(2026, 4, 20),
            consensus="NEUTRAL",
            sell_pct=0.50,
            mean_predicted=0.03,
            mean_ic=0.042,
            mean_hr=0.55,
            dry_run=False,
            _log_path_override=nonexistent,
        )
