"""Replay past monthly decisions with the current code (read-only dry runs).

Review 2026-09-25, step 5 (WP7) re-baseline. For each as-of date this runs

    python scripts/monthly_decision.py --dry-run --as-of <date> --skip-fred

in a subprocess, then collects the decision and health fields from the dry-run
artifacts under ``results/dry_run/monthly_decisions/YYYY-MM/`` into one CSV row.

Dry runs open the DB read-only (review step 1), and this script checks that
the DB file's sha256 is unchanged after every run. Run it from a checkout
whose ``data/pgr_financials.db`` is a copy, not the committed DB.

Usage:
    python scripts/replay_monthly_decisions.py --committed-dates --out replay.csv
    python scripts/replay_monthly_decisions.py --as-of 2026-09-21 --out replay.csv

``--committed-dates`` replays the as-of date of every committed production
month in ``artifacts/monthly_decisions/`` (from its run_manifest.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_DIR = REPO_ROOT / config.MONTHLY_DECISIONS_DIR
DRY_RUN_DIR = REPO_ROOT / config.DRY_RUN_MONTHLY_DECISIONS_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def committed_as_of_dates(base_dir: Path = COMMITTED_DIR) -> list[date]:
    """As-of dates of the committed production months, oldest first."""
    dates: list[date] = []
    for manifest_path in sorted(base_dir.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_classification", "production") != "production":
            continue
        as_of = manifest.get("as_of_date")
        if as_of:
            dates.append(date.fromisoformat(str(as_of)))
    return sorted(dates)


def _gate_status(model_health: dict, name: str) -> str | None:
    for gate in model_health.get("gates", []):
        if gate.get("name") == name:
            return gate.get("status")
    return None


def collect_row(as_of: date, dry_run_dir: Path = DRY_RUN_DIR) -> dict[str, object]:
    """Read one dry run's artifacts into a flat row."""
    out_dir = dry_run_dir / as_of.strftime("%Y-%m")
    summary = json.loads((out_dir / "monthly_summary.json").read_text(encoding="utf-8"))
    rec = summary.get("recommendation", {})
    health = summary.get("model_health", {}) or {}
    cpcv = health.get("cpcv", {}) or {}
    calibration = health.get("calibration", {}) or {}
    row: dict[str, object] = {
        "as_of": as_of.isoformat(),
        "recommendation_mode": rec.get("recommendation_mode"),
        "sell_pct": rec.get("recommended_sell_pct"),
        "consensus": rec.get("signal"),
        "confidence_tier": rec.get("confidence_tier"),
        "mean_predicted": rec.get("predicted_6m_relative_return"),
        "mean_ic_reported": rec.get("mean_ic"),
        "mean_hit_rate_reported": rec.get("mean_hit_rate"),
        "aggregate_oos_r2": rec.get("aggregate_oos_r2"),
        "pooled_ic": rec.get("aggregate_nw_ic"),
        "calibrated_prob_outperform": rec.get("prob_outperform_calibrated"),
        "equal_weight_mean_ic": health.get("equal_weight_mean_ic"),
        "quality_weighted_mean_ic": health.get("quality_weighted_mean_ic"),
        "pooled_ic_p_value": health.get("pooled_ic_p_value"),
        "hit_rate": health.get("hit_rate"),
        "constant_rule_hit_rate": health.get("constant_rule_hit_rate"),
        "pesaran_timmermann_p_value": health.get("pesaran_timmermann_p_value"),
        "clark_west_p_value": health.get("clark_west_p_value"),
        "shrinkage_alpha": health.get("shrinkage_alpha"),
        "prequential_ece": calibration.get("prequential_ece"),
        "conformal_trailing_coverage": health.get("conformal_trailing_coverage"),
        "cpcv_verdict": cpcv.get("verdict"),
        "cpcv_positive_paths": cpcv.get("positive_paths"),
        "cpcv_n_paths": cpcv.get("n_paths"),
        "gate_oos_r2": _gate_status(health, "oos_r2"),
        "gate_mean_ic": _gate_status(health, "mean_ic"),
        "gate_directional_skill": _gate_status(health, "directional_skill"),
        "gate_cpcv_completed": _gate_status(health, "cpcv_completed"),
    }
    shadow_path = out_dir / "consensus_shadow.csv"
    if shadow_path.exists():
        shadow = pd.read_csv(shadow_path)
        for variant in ("equal_weight", "quality_weighted"):
            match = shadow[shadow["variant"] == variant]
            if not match.empty:
                row[f"{variant}_consensus"] = match.iloc[0]["consensus"]
                row[f"{variant}_mode"] = match.iloc[0]["recommendation_mode"]
                row[f"{variant}_sell_pct"] = match.iloc[0]["recommended_sell_pct"]
    return row


def replay(as_of_dates: list[date], python: str = sys.executable) -> pd.DataFrame:
    """Run one dry run per as-of date and collect the rows."""
    db_path = REPO_ROOT / config.DB_PATH
    rows: list[dict[str, object]] = []
    for as_of in as_of_dates:
        before = _sha256(db_path)
        print(f"[replay] {as_of}: dry run (DB sha256 {before[:12]}...)", flush=True)
        subprocess.run(
            [
                python,
                "scripts/monthly_decision.py",
                "--dry-run",
                "--as-of",
                as_of.isoformat(),
                "--skip-fred",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        after = _sha256(db_path)
        if after != before:
            raise RuntimeError(f"DB changed during the {as_of} dry run: {before} -> {after}")
        rows.append(collect_row(as_of))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", action="append", default=[], metavar="YYYY-MM-DD")
    parser.add_argument(
        "--committed-dates",
        action="store_true",
        help="Replay every committed production month's as-of date.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip the dry runs and only read existing dry-run artifacts.",
    )
    parser.add_argument("--out", required=True, help="CSV path for the collected rows.")
    args = parser.parse_args()

    as_of_dates = [date.fromisoformat(value) for value in args.as_of]
    if args.committed_dates:
        as_of_dates = sorted(set(as_of_dates) | set(committed_as_of_dates()))
    if not as_of_dates:
        parser.error("Give --as-of dates or --committed-dates.")

    if args.collect_only:
        frame = pd.DataFrame([collect_row(as_of) for as_of in as_of_dates])
    else:
        frame = replay(as_of_dates)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"[replay] wrote {len(frame)} rows to {args.out}")


if __name__ == "__main__":
    main()
