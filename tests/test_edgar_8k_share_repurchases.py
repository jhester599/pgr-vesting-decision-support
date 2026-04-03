from __future__ import annotations

import os
import sys

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
    def test_decimal_millions_format_is_preserved(self):
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("0.21", "123.14"),
            "2023-08-16",
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.21)
        assert parsed["avg_cost_per_share"] == pytest.approx(123.14)

    def test_whole_share_count_is_normalized_to_millions(self):
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("46,822", "130.01"),
            "2023-09-15",
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.046822)
        assert parsed["avg_cost_per_share"] == pytest.approx(130.01)

    def test_larger_whole_share_count_is_normalized_to_millions(self):
        parsed = _parse_html_exhibit(
            _minimal_exhibit_html("299,855", "163.11"),
            "2023-12-15",
        )

        assert parsed is not None
        assert parsed["shares_repurchased"] == pytest.approx(0.299855)
        assert parsed["avg_cost_per_share"] == pytest.approx(163.11)
