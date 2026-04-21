from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.edgar_8k_fetcher import _parse_html_exhibit, _validate_parsed_record


def _broad_exhibit_html() -> str:
    return """
    <html><body>
      <table>
        <tr><td>Combined ratio</td><td>97.2</td></tr>
        <tr><td>Average diluted equivalent common shares</td><td>587.5</td></tr>
        <tr><td>Net premiums written</td><td>$</td><td>4743.8</td></tr>
        <tr><td>Net premiums earned</td><td>$</td><td>4599.2</td></tr>
        <tr><td>Net income</td><td>$</td><td>147.6</td></tr>
        <tr><td>Per share available to common shareholders</td><td>$</td><td>0.25</td></tr>
      </table>
      <table>
        <tr><td>Policies in Force</td></tr>
        <tr><td>Agency - auto</td><td>8377.0</td></tr>
        <tr><td>Direct - auto</td><td>11150.8</td></tr>
        <tr><td>Special lines</td><td>5934.3</td></tr>
        <tr><td>Total Personal Lines</td><td>25462.1</td></tr>
        <tr><td>Commercial Lines</td><td>1105.6</td></tr>
        <tr><td>Property business</td><td>3007.2</td></tr>
        <tr><td>Companywide Total</td><td>29574.9</td></tr>
      </table>
      <table>
        <tr><td>Net premiums written</td><td>1710.1</td><td>2099.1</td><td>3809.2</td><td>683.0</td><td>251.5</td><td>4743.8</td></tr>
        <tr><td>Net premiums earned</td><td>1666.0</td><td>1956.7</td><td>3622.7</td><td>764.5</td><td>212.0</td><td>4599.2</td></tr>
        <tr><td>Loss/LAE ratio</td><td>78.2</td><td>79.1</td><td>78.7</td><td>93.2</td><td>69.9</td><td>80.7</td></tr>
        <tr><td>Expense ratio</td><td>18.5</td><td>12.1</td><td>15.0</td><td>20.1</td><td>28.1</td><td>16.5</td></tr>
        <tr><td>Combined ratio</td><td>96.7</td><td>91.2</td><td>93.7</td><td>113.3</td><td>98.0</td><td>97.2</td></tr>
      </table>
      <table>
        <tr><td>Investment income</td><td>171.0</td></tr>
        <tr><td>Total net realized gains (losses) on securities</td><td>(80.6)</td></tr>
        <tr><td>Fees and other revenues</td><td>69.0</td></tr>
        <tr><td>Service revenues</td><td>26.3</td></tr>
        <tr><td>Total revenues</td><td>4784.9</td></tr>
        <tr><td>Losses and loss adjustment expenses</td><td>3720.8</td></tr>
        <tr><td>Policy acquisition costs</td><td>361.5</td></tr>
        <tr><td>Other underwriting expenses</td><td>457.8</td></tr>
        <tr><td>Interest expense</td><td>23.3</td></tr>
        <tr><td>Total expenses</td><td>4595.5</td></tr>
        <tr><td>Income before income taxes</td><td>189.4</td></tr>
        <tr><td>Provision for income taxes</td><td>41.8</td></tr>
        <tr><td>Total comprehensive income</td><td>$</td><td>114.0</td></tr>
      </table>
      <table>
        <tr><td>Basic</td><td>$</td><td>0.25</td></tr>
        <tr><td>Diluted</td><td>$</td><td>0.25</td></tr>
        <tr><td>Diluted</td><td>$</td><td>0.19</td></tr>
        <tr><td>Average common shares outstanding - Basic</td><td>584.8</td></tr>
        <tr><td>Total average equivalent common shares - Diluted</td><td>587.5</td></tr>
      </table>
      <table>
        <tr><td>Fixed-income securities</td><td>0.2%</td></tr>
        <tr><td>Common stocks</td><td>(1.5)%</td></tr>
        <tr><td>Total portfolio</td><td>0.1%</td></tr>
        <tr><td>Pretax annualized investment income book yield</td><td>3.3%</td></tr>
      </table>
      <table>
        <tr><td>Total investments2</td><td>61430.5</td></tr>
        <tr><td>Total assets</td><td>84990.6</td></tr>
        <tr><td>Unearned premiums</td><td>20514.7</td></tr>
        <tr><td>Loss and loss adjustment expense reserves</td><td>33345.4</td></tr>
        <tr><td>Debt</td><td>6887.2</td></tr>
        <tr><td>Total liabilities</td><td>67569.7</td></tr>
        <tr><td>Shareholders' equity</td><td>17420.9</td></tr>
        <tr><td>Common shares outstanding</td><td>585.1</td></tr>
        <tr><td>Common shares repurchased - actual</td><td>46,822</td></tr>
        <tr><td>Average cost per common share</td><td>$</td><td>130.01</td></tr>
        <tr><td>Book value per common share</td><td>$</td><td>28.93</td></tr>
        <tr><td>Net income</td><td>10.4</td><td>%</td></tr>
        <tr><td>Comprehensive income</td><td>8.4</td><td>%</td></tr>
        <tr><td>Net unrealized pretax gains (losses) on fixed-maturity securities</td><td>$</td><td>(3339.2)</td></tr>
        <tr><td>Debt-to-total capital ratio</td><td>28.3</td><td>%</td></tr>
        <tr><td>Fixed-income portfolio duration</td><td>2.9</td></tr>
        <tr><td>Weighted average credit quality</td><td>AA-</td></tr>
      </table>
    </body></html>
    """


def test_parse_html_exhibit_extracts_broader_current_fields():
    parsed = _parse_html_exhibit(_broad_exhibit_html(), "2023-09-15")

    assert parsed is not None
    assert parsed["pif_total"] == pytest.approx(29574.9)
    assert parsed["pif_special_lines"] == pytest.approx(5934.3)
    assert parsed["npw_agency"] == pytest.approx(1710.1)
    assert parsed["npe_direct"] == pytest.approx(1956.7)
    assert parsed["loss_lae_ratio"] == pytest.approx(80.7)
    assert parsed["investment_income"] == pytest.approx(171.0)
    assert parsed["total_revenues"] == pytest.approx(4784.9)
    assert parsed["policy_acquisition_costs"] == pytest.approx(361.5)
    assert parsed["total_comprehensive_income"] == pytest.approx(114.0)
    assert parsed["avg_shares_basic"] == pytest.approx(584.8)
    assert parsed["comprehensive_eps_diluted"] == pytest.approx(0.19)
    assert parsed["total_investments"] == pytest.approx(61430.5)
    assert parsed["loss_lae_reserves"] == pytest.approx(33345.4)
    assert parsed["common_shares_outstanding"] == pytest.approx(585.1)
    assert parsed["shares_repurchased"] == pytest.approx(0.046822)
    assert parsed["roe_net_income_trailing_12m"] == pytest.approx(10.4)
    assert parsed["debt_to_total_capital"] == pytest.approx(28.3)
    assert parsed["fte_return_common_stocks"] == pytest.approx(-1.5)
    assert parsed["investment_book_yield"] == pytest.approx(0.033)
    assert parsed["weighted_avg_credit_quality"] == "AA-"


def _quarterly_exhibit_html() -> str:
    """8-K exhibit in item 2.02 (quarterly earnings) format.

    The ratio rows have 7 columns: Agency, Direct, PLTotal, Commercial, Property,
    CompanyTotal_CurrentQuarter, CompanyTotal_PriorYear.  The parser must return
    the 6th value (index -2), not the 7th (index -1, prior year).
    """
    return """
    <html><body>
      <table>
        <tr><td>Companywide Total</td><td>31500.0</td></tr>
      </table>
      <table>
        <tr><td>Net premiums written</td><td>1800.0</td><td>2200.0</td><td>4000.0</td><td>720.0</td><td>260.0</td><td>4980.0</td></tr>
        <tr><td>Loss/LAE ratio</td><td>72.1</td><td>70.5</td><td>71.2</td><td>88.4</td><td>95.3</td><td>73.5</td><td>81.3</td></tr>
        <tr><td>Expense ratio</td><td>17.2</td><td>12.5</td><td>14.6</td><td>19.8</td><td>27.4</td><td>16.4</td><td>15.9</td></tr>
        <tr><td>Combined ratio</td><td>89.3</td><td>83.0</td><td>85.8</td><td>108.2</td><td>122.7</td><td>89.9</td><td>97.2</td></tr>
      </table>
      <table>
        <tr><td>Book value per common share</td><td>$</td><td>54.82</td></tr>
      </table>
    </body></html>
    """


def test_parse_quarterly_exhibit_extracts_current_quarter_ratios():
    parsed = _parse_html_exhibit(_quarterly_exhibit_html(), "2026-04-15", item_code="2.02")
    assert parsed is not None
    # 7-value rows: nums[-2] = current-quarter total, nums[-1] = prior-year total
    assert parsed["combined_ratio"] == pytest.approx(89.9)
    assert parsed["loss_lae_ratio"] == pytest.approx(73.5)
    assert parsed["expense_ratio"] == pytest.approx(16.4)
    assert parsed["book_value_per_share"] == pytest.approx(54.82)
    assert parsed["filing_type"] == "quarterly_earnings"


def test_parse_quarterly_exhibit_text_mode_cr_fallback():
    """CR label and values in separate rows — table scanner misses it, text-mode finds it."""
    html = """
    <html><body>
      <table>
        <tr><td>Companywide Total</td><td>32000.0</td></tr>
      </table>
      <table>
        <tr><td>Combined ratio</td></tr>
        <tr><td>Agency</td><td>Direct</td><td>PLTotal</td><td>Commercial</td><td>Property</td><td>Total</td><td>PY Total</td></tr>
        <tr><td></td><td>89.3</td><td>83.0</td><td>85.8</td><td>108.2</td><td>122.7</td><td>90.9</td><td>97.2</td></tr>
      </table>
      <table>
        <tr><td>Book value per common share</td><td>$</td><td>54.82</td></tr>
      </table>
    </body></html>
    """
    parsed = _parse_html_exhibit(html, "2026-04-15", item_code="2.02")
    assert parsed is not None
    # Table scanner sees no nums in the "combined ratio" row; text-mode fallback
    # finds 7 valid values in vicinity and picks vals[-2] = 90.9
    assert parsed["combined_ratio"] == pytest.approx(90.9)


def test_validate_parsed_record_accepts_pif_in_thousands():
    record = {
        "combined_ratio": 97.2,
        "loss_lae_ratio": 80.7,
        "expense_ratio": 16.5,
        "pif_total": 29574.9,
        "eps_basic": 0.25,
        "net_premiums_written": 4743.8,
        "npw_agency": 1710.1,
        "npw_direct": 2099.1,
        "npw_commercial": 683.0,
        "npw_property": 251.5,
    }
    validated = _validate_parsed_record(record, "2023-09-15", "000008066123000045")
    assert validated["pif_total"] == pytest.approx(29574.9)
