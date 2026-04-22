from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.edgar_8k_fetcher import _parse_html_exhibit


def _minimal_exhibit_html(repurchased_cell: str, avg_cost_cell: str) -> str:
    return f"""
    <html>
      <body>
        <table>
          <tr><td>Combined ratio</td><td>97.2</td></tr>
          <tr><td>Common shares repurchased</td><td>obsolete label in summary</td></tr>
        </table>
        <table>
          <tr><td>Common shares outstanding</td><td>585.1</td></tr>
          <tr><td>Common shares repurchased - actual</td><td>{repurchased_cell}</td></tr>
          <tr><td>Average cost per common share</td><td>$ {avg_cost_cell}</td></tr>
          <tr><td>Book value per common share</td><td>$ 28.93</td></tr>
        </table>
      </body>
    </html>
    """


class TestShareRepurchaseParsing:
    """Existing baseline tests — all formats the parser must handle."""

    def test_decimal_millions_format_is_preserved(self):
        """Pre-2023-08 format: decimal already in millions (e.g., "0.21")."""
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("0.21", "123.14"),
            "2023-08-16",
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.21)
        assert parsed["avg_cost_per_share"] == pytest.approx(123.14)

    def test_whole_share_count_is_normalized_to_millions(self):
        """Post-2023-08 large buyback: comma-formatted whole-share count."""
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("46,822", "130.01"),
            "2023-09-15",
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.046822)
        assert parsed["avg_cost_per_share"] == pytest.approx(130.01)

    def test_larger_whole_share_count_is_normalized_to_millions(self):
        """Post-2023-08 large buyback: six-digit comma-formatted whole-share count."""
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("299,855", "163.11"),
            "2023-12-15",
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.299855)
        assert parsed["avg_cost_per_share"] == pytest.approx(163.11)


class TestSmallIntegerThousandsFormat:
    """
    Regression tests for the 1000x normalization bug.

    Root cause: post-2023-08 small buybacks appear in the HTML as plain integers
    representing *thousands* of shares (e.g., "51" = 51,000 shares = 0.051M).
    The original parser divided these by 1,000,000 instead of 1,000, storing
    0.000051 instead of 0.051 — 1000x too small.

    The confirmed bad rows in the committed CSV were:
      2023-12: 0.000274 (should be 0.274)
      2024-09: 0.000077 (should be 0.077)
      2024-10: 0.000195 (should be 0.195) — confirmed by official Nov-2024 release
      2024-11: 0.000051 (should be 0.051) — confirmed by official Nov-2024 release
      2024-12: 0.000079 (should be 0.079)
      2025-08: 0.000087 (should be 0.087) — confirmed by official Sep-2025 release
    """

    def test_small_integer_51_normalized_to_thousands(self):
        """51 in post-2023-08 filing → 0.051M shares (not 0.000051 or 51.0)."""
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("51", "244.21"),
            "2024-12-13",  # filing for Nov-2024 month-end
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.051, rel=1e-4)
        assert parsed["avg_cost_per_share"] == pytest.approx(244.21)
        # Implied buyback: 0.051M × $244.21 = ~$12.5M  (plausible)
        implied_dollars = parsed["shares_repurchased"] * 1_000_000 * parsed["avg_cost_per_share"]
        assert implied_dollars == pytest.approx(12_454_710, rel=0.01)

    def test_small_integer_87_normalized_to_thousands(self):
        """87 in post-2023-08 filing → 0.087M shares (not 0.000087 or 87.0)."""
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("87", "243.31"),
            "2025-09-17",  # filing for Aug-2025 month-end
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.087, rel=1e-4)
        assert parsed["avg_cost_per_share"] == pytest.approx(243.31)

    def test_small_integer_195_normalized_to_thousands(self):
        """195 in post-2023-08 filing → 0.195M shares (not 0.000195 or 195.0)."""
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("195", "253.66"),
            "2024-11-15",  # filing for Oct-2024 month-end
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.195, rel=1e-4)
        assert parsed["avg_cost_per_share"] == pytest.approx(253.66)

    def test_old_format_large_decimal_not_divided_by_1000(self):
        """Pre-2023-08 large buyback in decimal millions must not be divided by 1000."""
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("16.9", "88.00"),
            "2004-11-15",  # filing for Oct-2004 month-end (the big ASR)
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(16.9, rel=1e-4)

    def test_normalized_value_not_divided_again(self):
        """An already-correct value like 0.051 must survive round-trip unchanged."""
        # Filed under old format (pre-2023-08): "0.051" is 0.051M shares directly.
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("0.051", "244.00"),
            "2020-06-15",
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.051, rel=1e-4)

    def test_zero_repurchases_stored_as_zero(self):
        """Zero repurchases (blackout period) must remain zero."""
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("0", "0.00"),
            "2022-05-15",
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.0, abs=1e-9)


class TestCsvDataIntegrity:
    """Data-integrity tests against the committed pgr_edgar_cache.csv."""

    CSV_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "pgr_edgar_cache.csv"
    )

    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(self.CSV_PATH)

    def test_confirmed_bad_rows_corrected(self, df: pd.DataFrame):
        """The three task-confirmed bad rows must now have the right values."""
        row_oct = df[df["report_period"] == "2024-10"].iloc[0]
        row_nov = df[df["report_period"] == "2024-11"].iloc[0]
        row_aug = df[df["report_period"] == "2025-08"].iloc[0]

        # shares_repurchased (millions)
        assert row_oct["shares_repurchased"] == pytest.approx(0.195, rel=1e-4), (
            f"2024-10 shares_repurchased={row_oct['shares_repurchased']} — expected 0.195"
        )
        assert row_nov["shares_repurchased"] == pytest.approx(0.051, rel=1e-4), (
            f"2024-11 shares_repurchased={row_nov['shares_repurchased']} — expected 0.051"
        )
        assert row_aug["shares_repurchased"] == pytest.approx(0.087, rel=1e-4), (
            f"2025-08 shares_repurchased={row_aug['shares_repurchased']} — expected 0.087"
        )

        # avg_cost_per_share unchanged
        assert row_oct["avg_cost_per_share"] == pytest.approx(253.66, rel=1e-4)
        assert row_nov["avg_cost_per_share"] == pytest.approx(244.21, rel=1e-4)
        assert row_aug["avg_cost_per_share"] == pytest.approx(243.31, rel=1e-4)

        # book_value_per_share unchanged
        assert row_oct["book_value_per_share"] == pytest.approx(45.35, rel=1e-4)
        assert row_nov["book_value_per_share"] == pytest.approx(47.43, rel=1e-4)
        assert row_aug["book_value_per_share"] == pytest.approx(59.97, rel=1e-4)

    def test_no_implausibly_tiny_repurchase_with_real_cost(self, df: pd.DataFrame):
        """
        No row with avg_cost_per_share > $50 should have an implied buyback < $10K.
        That would indicate a 1000x scale error (thousands stored as units → /1M).
        """
        real_cost = df[(df["avg_cost_per_share"] > 50) & df["shares_repurchased"].notna()].copy()
        real_cost = real_cost[real_cost["shares_repurchased"] > 0]
        real_cost["implied_dollars"] = (
            real_cost["shares_repurchased"] * 1_000_000 * real_cost["avg_cost_per_share"]
        )
        tiny = real_cost[real_cost["implied_dollars"] < 10_000]
        assert tiny.empty, (
            f"Rows with implied_repurchase_dollars < $10K when avg_cost > $50:\n"
            f"{tiny[['report_period','shares_repurchased','avg_cost_per_share','implied_dollars']].to_string()}"
        )

    def test_no_implausibly_large_share_count(self, df: pd.DataFrame):
        """No monthly share repurchase should exceed 25M shares (PGR historical max ~17M)."""
        bad = df[df["shares_repurchased"] > 25]
        assert bad.empty, (
            f"Rows with shares_repurchased > 25M:\n"
            f"{bad[['report_period','shares_repurchased']].to_string()}"
        )

    def test_all_nonzero_repurchases_have_plausible_dollar_amount(self, df: pd.DataFrame):
        """
        Every active (non-zero) repurchase row must imply a dollar amount in [$1K, $750M].

        Floor $1K: below this is almost certainly a scale error (thousands stored as units).
        Ceiling $750M: above this is implausible for normal-course buybacks.

        Oct-2004 is explicitly excluded — it was a one-time Dutch auction tender offer
        (~$1.49B) that is well outside normal-course repurchase activity.
        """
        active = df[
            df["shares_repurchased"].notna()
            & (df["shares_repurchased"] > 0)
            & df["avg_cost_per_share"].notna()
            & (df["report_period"] != "2004-10")  # Dutch auction outlier
        ].copy()
        active["implied_M"] = active["shares_repurchased"] * active["avg_cost_per_share"]
        bad = active[(active["implied_M"] < 0.001) | (active["implied_M"] > 750)]
        assert bad.empty, (
            f"Rows with implied buyback outside [$1K, $750M]:\n"
            f"{bad[['report_period','shares_repurchased','avg_cost_per_share','implied_M']].to_string()}"
        )
