"""Write x14 indicator candidate synthesis artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.x14_indicator_synthesis import (
    build_x14_recommendation,
    select_indicator_candidate,
    summarize_horizon_evidence,
)

OUTPUT_DIR = Path("results") / "research"
SUMMARY_PATH = OUTPUT_DIR / "x14_indicator_synthesis_summary.json"
MEMO_PATH = OUTPUT_DIR / "x14_indicator_synthesis_memo.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required x14 input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_x13_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build raw-vs-adjusted comparison rows from x13 ranked output."""
    frame = pd.DataFrame(rows)
    comparisons: list[dict[str, Any]] = []
    if frame.empty:
        return comparisons
    for horizon in sorted(frame["horizon_months"].unique()):
        sub = frame[frame["horizon_months"] == horizon]
        raw = sub[sub["target_variant"] == "raw"].sort_values("rank").iloc[0]
        adjusted = sub[sub["target_variant"] == "adjusted"].sort_values("rank").iloc[0]
        delta = float(adjusted["implied_price_mae"] - raw["implied_price_mae"])
        comparisons.append(
            {
                "horizon_months": int(horizon),
                "raw_model_name": raw["model_name"],
                "adjusted_model_name": adjusted["model_name"],
                "adjusted_beats_raw": bool(delta < 0.0),
                "mae_delta": delta,
                "raw_mae": float(raw["implied_price_mae"]),
                "adjusted_mae": float(adjusted["implied_price_mae"]),
            }
        )
    return comparisons


def write_memo(
    comparison: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    candidate: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    """Write human-readable x14 memo."""
    lines = [
        "# x14 Indicator Synthesis Memo",
        "",
        "## Scope",
        "",
        "x14 translates the adjusted decomposition research into a possible",
        "research-only indicator candidate for monthly report/dashboard",
        "discussion. It does not change production or shadow outputs.",
        "The synthesis explicitly reads x3, x5, x8, x11, x12, and x13",
        "research artifacts.",
        "",
        "## Raw vs Adjusted Comparison",
        "",
    ]
    for row in comparison:
        verdict = "beat" if row["adjusted_beats_raw"] else "did not beat"
        lines.append(
            f"- {row['horizon_months']}m adjusted `{row['adjusted_model_name']}` "
            f"{verdict} raw `{row['raw_model_name']}` "
            f"(delta MAE {row['mae_delta']:.3f})."
        )
    lines.extend(["", "## Horizon Evidence", ""])
    for row in evidence:
        lines.append(
            f"- {row['horizon_months']}m: x12 adjusted gate="
            f"`{row['x12_supports_adjusted_target_family']}`, x13 adjusted gate="
            f"`{row['x13_adjusted_beats_raw']}`, x5 anchor="
            f"`{row['x5_leader_model_name']}`, x3 return leader="
            f"`{row['x3_return_leader_model_name']}`, x3 log-return leader="
            f"`{row['x3_log_return_leader_model_name']}`."
        )
    lines.extend(["", "## Recommendation", ""])
    lines.append(f"- Status: `{recommendation['status']}`.")
    lines.append(f"- Rationale: {recommendation['rationale']}")
    lines.extend(["", "## Candidate", ""])
    if candidate["status"] == "candidate":
        lines.append(
            f"- Proposed indicator: `{candidate['signal_family']}` at "
            f"{candidate['horizon_months']}m using `{candidate['model_name']}`."
        )
        lines.append(
            "- Intended use: show as a research-only structural medium-term "
            "readout on the monthly report/dashboard, pending a later plan."
        )
    else:
        lines.append(f"- No candidate nominated. {candidate['reason']}")
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    x3 = _load_json(OUTPUT_DIR / "x3_direct_return_summary.json")
    x5 = _load_json(OUTPUT_DIR / "x5_decomposition_summary.json")
    x8 = _load_json(OUTPUT_DIR / "x8_synthesis_summary.json")
    x11 = _load_json(OUTPUT_DIR / "x11_capital_synthesis_summary.json")
    x12 = _load_json(OUTPUT_DIR / "x12_bvps_target_audit_summary.json")
    x13 = _load_json(OUTPUT_DIR / "x13_adjusted_decomposition_summary.json")
    comparison = build_x13_comparison(x13["ranked_rows"])
    evidence = summarize_horizon_evidence(
        x3_rows=x3["ranked_rows"],
        x5_rows=x5["ranked_rows"],
        x12_rows=x12["ranked_rows"],
        x13_comparison=comparison,
    )
    candidate = select_indicator_candidate(
        evidence_rows=evidence,
        x13_comparison=comparison,
        x8_status=x8["shadow_readiness"]["status"],
        x11_status=x11["recommendation"]["status"],
    )
    recommendation = build_x14_recommendation(
        candidate_status=candidate["status"],
        x8_status=x8["shadow_readiness"]["status"],
        x11_status=x11["recommendation"]["status"],
    )
    payload = {
        "version": "x14",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "evidence_artifacts": [
            "x3_direct_return_summary.json",
            "x5_decomposition_summary.json",
            "x8_synthesis_summary.json",
            "x11_capital_synthesis_summary.json",
            "x12_bvps_target_audit_summary.json",
            "x13_adjusted_decomposition_summary.json",
        ],
        "x13_comparison": comparison,
        "horizon_evidence": evidence,
        "indicator_candidate": candidate,
        "recommendation": recommendation,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(comparison, evidence, candidate, recommendation)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
