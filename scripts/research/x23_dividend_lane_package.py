"""Run x23 dividend-lane synthesis and package artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.x23_dividend_lane_package import build_x23_summary

OUTPUT_DIR = Path("results") / "research"
SUMMARY_PATH = OUTPUT_DIR / "x23_dividend_lane_package.json"
MEMO_PATH = OUTPUT_DIR / "x23_research_memo.md"
PROMPT_PATH = OUTPUT_DIR / "x23_peer_review_prompt.md"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_artifacts(summary: dict[str, Any]) -> None:
    """Write x23 JSON, memo, and peer-review prompt."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_memo(summary)
    write_peer_review_prompt(summary)


def write_memo(summary: dict[str, Any]) -> None:
    """Write a compact x23 memo."""
    lane = summary["dividend_lane_shape"]
    recommendation = summary["recommendation"]
    lines = [
        "# x23 Research Memo",
        "",
        "## Scope",
        "",
        "x23 packages the dividend-policy rebuild so the broader x-series can",
        "see what this lane currently is and is not.",
        "",
        "## Findings",
        "",
        f"- Occurrence status: `{lane['occurrence_status']}`.",
        f"- Best size target scale: `{lane['best_size_target_scale']}`.",
        f"- Best size row: `{lane['best_size_feature_set']}` / `{lane['best_size_model_name']}`.",
        f"- Recommendation: `{recommendation['status']}`.",
        "",
        "## Interpretation",
        "",
        f"- {recommendation['rationale']}",
        "- The clean next use of this lane is as a research-only annual size",
        "  watch, not as a standalone occurrence/size deployment path.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_peer_review_prompt(summary: dict[str, Any]) -> None:
    """Write a deep-research prompt for external challenge/peer review."""
    lane = summary["dividend_lane_shape"]
    prompt = f"""# x23 Dividend Lane Peer Review Prompt

Please review the current research-only dividend lane for PGR.

Known findings:
- Post-policy regime anchored at December 1, 2018.
- Annual prediction timestamp is November month-end.
- Overlapping post-policy OOS years are currently one-class on occurrence.
- Best surviving size target is `{lane['best_size_target_scale']}`.
- Best current row is `{lane['best_size_feature_set']}` / `{lane['best_size_model_name']}`.

Please challenge:
1. whether the post-policy occurrence problem should be modeled at all yet
2. whether normalizing special-dividend excess by current BVPS is economically sound
3. whether there are stronger but still disciplined insurer-specific capital-return features available from the repo's existing data sources
4. whether a simpler annual baseline should dominate the current ridge result
5. what evidence would justify promoting this lane into a future dashboard/monthly-indicator discussion

Constraints:
- strict temporal validation only
- no K-Fold CV
- research-only; do not suggest production wiring
- prefer low-complexity, small-sample-safe models
"""
    PROMPT_PATH.write_text(prompt, encoding="utf-8")


def main() -> None:
    summary = build_x23_summary(
        {
            "x20": _read_json(OUTPUT_DIR / "x20_dividend_policy_synthesis.json"),
            "x21": _read_json(OUTPUT_DIR / "x21_dividend_target_scales_summary.json"),
            "x22": _read_json(OUTPUT_DIR / "x22_dividend_size_baselines_summary.json"),
        }
    )
    write_artifacts(summary)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")
    print(f"Wrote {PROMPT_PATH}")


if __name__ == "__main__":
    main()
