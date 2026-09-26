"""Shared rendering helpers for the monthly decision reports.

``recommendation_report`` writes ``recommendation.md`` and
``diagnostic_report`` writes ``diagnostic.md`` and the calibration plot;
both use the labels here.
"""

from __future__ import annotations

from datetime import date

from src.reporting.decision_rendering import (
    build_executive_summary_lines as render_executive_summary_lines,
)
from src.reporting.decision_rendering import (
    build_vest_decision_lines as render_vest_decision_lines,
)

# Footer of recommendation.md and diagnostic.md. The reports still name the
# pre-phase-5 script so that the refactor leaves them byte-identical (review
# 2026-09-25, section 5, phase 5); the entry point is now
# cli/monthly_decision.py, and run_manifest.json records that.
REPORT_GENERATED_BY: str = "scripts/monthly_decision.py"

# ETF short descriptions (shown in per-benchmark tables). Stripped of provider
# names ("Vanguard", "SPDR", "iShares") and generic suffixes ("Fund", "ETF")
# per reporting guidelines.
ETF_DESCRIPTIONS: dict[str, str] = {
    "VTI":  "Total Stock Market",
    "VOO":  "S&P 500",
    "VGT":  "Information Technology",
    "VHT":  "Health Care",
    "VFH":  "Financials",
    "VIS":  "Industrials",
    "VDE":  "Energy",
    "VPU":  "Utilities",
    "KIE":  "S&P Insurance",
    "VXUS": "Total International Stock",
    "VEA":  "Developed Markets ex-US",
    "VWO":  "Emerging Markets",
    "VIG":  "Dividend Appreciation",
    "SCHD": "US Dividend Equity",
    "BND":  "Total Bond Market",
    "BNDX": "Total International Bond",
    "VCIT": "Intermediate-Term Corporate Bond",
    "VMBS": "Mortgage-Backed Securities",
    "VNQ":  "Real Estate",
    "GLD":  "Gold Shares",
    "DBC":  "DB Commodity Index",
}


MODEL_VERSION_LABEL = (
    "v11.1 (lean 2-model ensemble: Ridge + GBT, v18 feature sets, "
    "8-benchmark PRIMARY_FORECAST_UNIVERSE, inverse-variance weighting, "
    "prequential post-ensemble shrinkage and realised-only health metrics "
    "(review 2026-09-25 WP7); C(8,2) CPCV with 7 paths is diagnostic only; "
    "ElasticNet+BayesianRidge retired after v18/v20 research showed Ridge+GBT "
    "outperforms on IC, hit rate, and obs/feature ratio)"
)


def build_executive_summary_lines(
    as_of: date,
    consensus: str,
    confidence_tier: str,
    mean_predicted: float,
    sell_pct: float,
    recommendation_mode: dict[str, str | float],
    aggregate_health: dict | None,
    previous_summary: dict | None,
    next_vest_summary: dict | None,
) -> list[str]:
    """Compatibility wrapper around the extracted summary-rendering helper."""
    return render_executive_summary_lines(
        as_of=as_of,
        consensus=consensus,
        confidence_tier=confidence_tier,
        mean_predicted=mean_predicted,
        sell_pct=sell_pct,
        recommendation_mode=recommendation_mode,
        aggregate_health=aggregate_health,
        previous_summary=previous_summary,
        next_vest_summary=next_vest_summary,
    )


def build_vest_decision_lines(
    next_vest_summary: dict | None,
    recommendation_mode: dict[str, str | float],
    sell_pct: float,
) -> list[str]:
    """Compatibility wrapper around the extracted vest-section helper."""
    return render_vest_decision_lines(
        next_vest_summary=next_vest_summary,
        recommendation_mode=recommendation_mode,
        sell_pct=sell_pct,
    )
