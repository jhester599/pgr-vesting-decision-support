"""Build the x24 research-only indicator contract and resume docs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.x24_indicator_contract import (
    build_x24_indicator_contract,
    build_x24_peer_review_prompt,
)

RESULTS_DIR = Path("results") / "research"
DOCS_DIR = Path("docs") / "research"
CONTRACT_PATH = RESULTS_DIR / "x24_indicator_contract.json"
MEMO_PATH = RESULTS_DIR / "x24_research_memo.md"
BUNDLE_PROMPT_PATH = RESULTS_DIR / "x24_bundle_peer_review_prompt.md"
STRUCTURAL_PROMPT_PATH = RESULTS_DIR / "x24_structural_peer_review_prompt.md"
RESUME_PATH = DOCS_DIR / "x_series_resume_2026-04-24.md"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_contract_artifacts(contract: dict[str, Any]) -> None:
    """Write x24 contract JSON and memo."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(
        json.dumps(contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# x24 Research Memo",
        "",
        "## Scope",
        "",
        "x24 packages the surviving x-series signals into one research-only",
        "indicator bundle for later monthly-report/dashboard discussion.",
        "",
        "## Bundle",
        "",
        f"- Structural signal: `{contract['structural_signal']['indicator_name']}`.",
        f"- Structural model: `{contract['structural_signal']['model_name']}`.",
        f"- Dividend signal: `{contract['dividend_signal']['indicator_name']}`.",
        f"- Dividend target scale: `{contract['dividend_signal']['target_scale']}`.",
        f"- Bundle status: `{contract['status']}`.",
        "",
        "## Interpretation",
        "",
        "- x24 is packaging only. It does not wire anything into production,",
        "  monthly outputs, or shadow artifacts.",
        "- The structural side remains a 6m research watch anchored on no-change",
        "  P/B. The dividend side remains a research-only annual size watch.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_additional_prompts(
    *,
    contract: dict[str, Any],
    x8_summary: dict[str, Any],
    x11_summary: dict[str, Any],
    x16_package: dict[str, Any],
    x23_package: dict[str, Any],
) -> None:
    """Write x24 peer-review prompts."""
    BUNDLE_PROMPT_PATH.write_text(
        build_x24_peer_review_prompt(
            contract=contract,
            x8_summary=x8_summary,
            x11_summary=x11_summary,
            x23_package=x23_package,
        ),
        encoding="utf-8",
    )
    structural_prompt = f"""# x24 Structural P/B Peer Review Prompt

Please review the current structural x-series path for PGR.

Known state:
- Packaged structural indicator: `{x16_package['indicator_name']}`
- Horizon: `{x16_package['horizon_months']}m`
- Current model: `{x16_package['model_name']}`
- Current P/B policy: `{x16_package['pb_anchor_policy']}`
- x11 recommendation: `{x11_summary.get('recommendation', {}).get('status')}`

Please challenge:
1. whether the 6m structural path is the right horizon to carry forward
2. whether no-change P/B is still the correct anchor or merely the least-bad placeholder
3. which insurer-specific valuation features from existing repo data are most promising for a disciplined P/B leg revisit
4. whether a better packaged structural signal would be BVPS-only, P/B-only regime state, or the existing recombined decomposition

Constraints:
- strict temporal validation only
- no K-Fold CV
- no production wiring
- prefer low-complexity, robust modeling under small samples
"""
    STRUCTURAL_PROMPT_PATH.write_text(structural_prompt, encoding="utf-8")


def write_resume_doc(
    *,
    contract: dict[str, Any],
    x8_summary: dict[str, Any],
    x11_summary: dict[str, Any],
    x16_package: dict[str, Any],
    x23_package: dict[str, Any],
) -> None:
    """Write a clear x1-x24 resume document."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    current_branch = _git(["git", "branch", "--show-current"])
    master_head = _git(["git", "rev-parse", "origin/master"])
    branch_head = _git(["git", "rev-parse", "HEAD"])
    lines = [
        "# X-Series Resume Summary (2026-04-24)",
        "",
        "## Purpose",
        "",
        "This document is the restart point for the separate x-series PGR",
        "research lane. It summarizes x1 through x24, records the current repo",
        "state, links useful peer-review prompts, and recommends the next",
        "research steps.",
        "",
        "## Current Repo State",
        "",
        f"- Current working branch when this document was written: `{current_branch}`.",
        f"- Current branch HEAD: `{branch_head}`.",
        f"- `origin/master` HEAD at write time: `{master_head}`.",
        "- Important: the x17-x24 chain currently lives on the stacked branch",
        "  lineage above `codex/x17-persistent-bvps` and is not yet reflected in",
        "  `master` in this local/repo state. Before future x-series work, verify",
        "  whether the latest stacked branch has been merged to `master` or open a",
        "  catch-up PR directly to `master`.",
        "",
        "## X1-X24 Summary",
        "",
        "### x1-x6: Lane Setup And First Baselines",
        "",
        "- `x1`: created the separate x-series lane, target utilities, feature",
        "  inventory, and data sufficiency audit. Key constraint: only 22 annual",
        "  special-dividend snapshots.",
        "- `x2`: multi-horizon absolute direction classification baseline. Result:",
        "  did not clear the base-rate gate.",
        "- `x3`: direct forward-return benchmark. Result: mostly baseline-heavy;",
        "  only the 12m drift path clearly cleared the no-change gate.",
        "- `x4`: BVPS forecasting leg. Result: strongest early lane; beat no-change",
        "  BVPS across all four horizons.",
        "- `x5`: BVPS x P/B decomposition benchmark. Result: useful structurally,",
        "  but the stable anchor remained `no_change_pb`.",
        "- `x6`: annual special-dividend two-stage sidecar. Result: low-confidence,",
        "  small-sample annual research only.",
        "",
        "### x7-x12: Targeted Follow-Up And Synthesis",
        "",
        "- `x7`: targeted TA replacement follow-up. Result: `ta_minimal_plus_vwo_pct_b`",
        "  cleared 2/4 horizons, but broad TA expansion was not justified.",
        f"- `x8`: cross-lane synthesis. Result: shadow readiness remained `{x8_summary.get('shadow_readiness', {}).get('status')}`.",
        "- `x9`: BVPS bridge features, interactions, and stronger baselines.",
        "  Result: improved BVPS at 1m and 3m, but not 6m/12m.",
        "- `x10`: capital-enhanced dividend lane using x9 features. Result:",
        "  improved x6 EV MAE but remained low-confidence.",
        f"- `x11`: synthesis of x9/x10. Result: `{x11_summary.get('recommendation', {}).get('status')}`.",
        "- `x12`: dividend-adjusted BVPS target audit. Result: adjustment helped",
        "  3m/6m, but not 1m/12m.",
        "",
        "### x13-x17: Structural Packaging And Persistent BVPS",
        "",
        "- `x13`: adjusted decomposition follow-up. Result: only the 6m adjusted",
        "  structural path clearly survived.",
        "- `x14`: indicator synthesis. Result: narrowed to one bounded 6m",
        "  structural candidate.",
        "- `x15`: P/B regime overlay research. Result: no overlay beat the",
        "  no-change P/B anchor.",
        f"- `x16`: packaged the structural indicator `{x16_package['indicator_name']}`.",
        "- `x17`: persistent BVPS research. Result: persistent BVPS helped medium",
        "  horizons (3m/6m), supporting separation of capital creation from payout policy.",
        "",
        "### x18-x23: Dividend Policy Rebuild",
        "",
        "- `x18`: rebuilt dividend labels around the December 2018 policy break and",
        "  a December-February payout window.",
        "- `x19`: post-policy-only dividend model using persistent-BVPS capital",
        "  features. Result: better than x10 on the overlapping post-policy years,",
        "  but the sample was only 3 OOS years.",
        "- `x20`: synthesized x19 vs x10. Result: occurrence is one-class on the",
        "  overlap sample, so only size-target experiments are identifiable.",
        "- `x21`: target-scale comparison for dividend size. Result:",
        "  `special_dividend_excess / current_bvps` was best.",
        "- `x22`: baseline challenge on dividend size. Result: the",
        "  `to_current_bvps` size target survived baseline challengers.",
        f"- `x23`: packaged the dividend lane. Result: `{x23_package.get('recommendation', {}).get('status')}` with occurrence still `{x23_package.get('recommendation', {}).get('occurrence_status')}`.",
        "",
        "### x24: Research Indicator Bundle",
        "",
        f"- `x24` bundles `{contract['structural_signal']['indicator_name']}` and",
        f"  `{contract['dividend_signal']['indicator_name']}` into one research-only",
        "  indicator contract for future monthly-report/dashboard discussion.",
        "",
        "## Strongest Findings",
        "",
        "- BVPS forecasting is still the strongest core x-series leg.",
        "- The structural 6m path is the cleanest price-adjacent packaged signal,",
        "  but it still depends on `no_change_pb` as the valuation anchor.",
        "- The dividend lane became much clearer after the policy-regime rebuild:",
        "  occurrence is still underidentified post-policy, but dividend size has a",
        "  credible research-only path via `special_dividend_excess / current_bvps`.",
        "- The x-series remains research-only overall; it is not ready for shadow or production wiring.",
        "",
        "## What Did Not Work",
        "",
        "- Broad absolute classification did not establish a robust edge over base rate.",
        "- Direct return prediction was mostly baseline-heavy.",
        "- P/B overlays and regime tricks did not beat the plain no-change anchor.",
        "- Post-policy dividend occurrence currently has too little label variation to support a practical classifier.",
        "",
        "## Surviving Packaged Signals",
        "",
        f"- Structural watch: `{contract['structural_signal']['indicator_name']}`",
        f"  (`{contract['structural_signal']['model_name']}`, {contract['structural_signal']['horizon_months']}m).",
        f"- Dividend watch: `{contract['dividend_signal']['indicator_name']}`",
        f"  (`{contract['dividend_signal']['target_scale']}`, {contract['dividend_signal']['timing']}).",
        "",
        "## Recommended Next Steps",
        "",
        "1. Merge/catch up the stacked x17-x24 branch chain to `master` cleanly.",
        "2. Start `x25` as a research-only monthly indicator contract/output for the",
        "   x24 bundle, without touching production or shadow artifacts.",
        "3. Start `x26` as a structural P/B revisit focused on disciplined",
        "   insurer-specific valuation features from existing data.",
        "4. Keep the dividend lane annual and size-focused until occurrence gains",
        "   meaningful post-policy label variation.",
        "5. Use the peer-review prompts below before any attempt to promote the x-series into a reporting surface.",
        "",
        "## Peer Review / Deep Research Prompts",
        "",
        f"- Dividend lane prompt: [x23 peer review](/C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/x23_peer_review_prompt.md)",
        f"- Holistic bundle prompt: [x24 bundle peer review](/C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/x24_bundle_peer_review_prompt.md)",
        f"- Structural P/B prompt: [x24 structural peer review](/C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/x24_structural_peer_review_prompt.md)",
        f"- Earlier structural prompt: [x16 peer review](/C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/x16_peer_review_prompt.md)",
        "",
        "## Resume Advice",
        "",
        "When restarting later, read this file first, then `x24_research_memo.md`,",
        "`x23_research_memo.md`, `x16_research_memo.md`, and verify whether the",
        "x17-x24 chain is on `master`. If not, resolve that before doing any new x-series branch work.",
    ]
    RESUME_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    x8_summary = _read_json(RESULTS_DIR / "x8_synthesis_summary.json")
    x11_summary = _read_json(RESULTS_DIR / "x11_capital_synthesis_summary.json")
    x16_package = _read_json(RESULTS_DIR / "x16_indicator_package.json")
    x23_package = _read_json(RESULTS_DIR / "x23_dividend_lane_package.json")
    contract = build_x24_indicator_contract(
        x16_package=x16_package,
        x23_package=x23_package,
        x8_summary=x8_summary,
    )
    write_contract_artifacts(contract)
    write_additional_prompts(
        contract=contract,
        x8_summary=x8_summary,
        x11_summary=x11_summary,
        x16_package=x16_package,
        x23_package=x23_package,
    )
    write_resume_doc(
        contract=contract,
        x8_summary=x8_summary,
        x11_summary=x11_summary,
        x16_package=x16_package,
        x23_package=x23_package,
    )
    print(f"Wrote {CONTRACT_PATH}")
    print(f"Wrote {MEMO_PATH}")
    print(f"Wrote {BUNDLE_PROMPT_PATH}")
    print(f"Wrote {STRUCTURAL_PROMPT_PATH}")
    print(f"Wrote {RESUME_PATH}")


if __name__ == "__main__":
    main()
