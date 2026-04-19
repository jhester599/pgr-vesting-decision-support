"""Verify monthly decision artifacts after workflow generation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.database import db_client


REQUIRED_MONTHLY_ARTIFACTS = [
    "recommendation.md",
    "diagnostic.md",
    "signals.csv",
    "benchmark_quality.csv",
    "consensus_shadow.csv",
    "classification_shadow.csv",
    "decision_overlays.csv",
    "dashboard.html",
    "monthly_summary.json",
    "run_manifest.json",
]


@dataclass(frozen=True)
class MonthlyVerificationResult:
    """Structured result for monthly artifact verification."""

    output_dir: Path
    schema_version: str | int | None
    git_sha: str | None
    warning_count: int
    freshness_status: str
    ta_variants: list[str]

    def summary_lines(self) -> list[str]:
        """Return GitHub-step-summary markdown lines."""
        return [
            "## Monthly Decision Report",
            "",
            f"- Output folder: `{self.output_dir}`",
            f"- Schema version: `{self.schema_version}`",
            f"- Git SHA: `{self.git_sha}`",
            f"- Warnings: `{self.warning_count}`",
            f"- Data freshness: `{self.freshness_status}`",
            f"- TA shadow variants: `{len(self.ta_variants)}`",
        ]


def _monthly_dir(base_dir: Path, as_of_date: str | None) -> Path:
    """Return the monthly output directory to verify."""
    if as_of_date is not None:
        month_dir = base_dir / as_of_date[:7]
        if not month_dir.is_dir():
            raise SystemExit(f"Monthly output folder not found: {month_dir}")
        return month_dir
    candidates = sorted(path for path in base_dir.iterdir() if path.is_dir())
    if not candidates:
        raise SystemExit(f"No monthly output folders found under {base_dir}")
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _required_ta_variants(monthly_summary: dict[str, Any]) -> list[str]:
    """Return reporting-only TA variants expected in the ledger."""
    variants = monthly_summary.get("classification_shadow_variants", [])
    if not isinstance(variants, list):
        raise SystemExit("monthly_summary.json classification_shadow_variants must be a list")
    result = sorted(
        str(item["variant"])
        for item in variants
        if isinstance(item, dict)
        and str(item.get("variant", "")).startswith("ta_")
        and item.get("reporting_only") is True
    )
    if not result:
        raise SystemExit("monthly_summary.json contains no reporting-only TA variants")
    return result


def _verify_ta_artifacts(
    *,
    base_dir: Path,
    output_dir: Path,
    as_of_date: str,
    ta_variants: list[str],
) -> None:
    """Verify TA variants appear in shadow detail and durable ledger."""
    classification_shadow = pd.read_csv(output_dir / "classification_shadow.csv")
    if "variant" not in classification_shadow.columns:
        raise SystemExit("classification_shadow.csv is missing variant column")
    detail_variants = set(classification_shadow["variant"].astype(str))
    missing_detail = sorted(set(ta_variants) - detail_variants)
    if missing_detail:
        raise SystemExit(
            "classification_shadow.csv missing TA variants: "
            + ", ".join(missing_detail)
        )

    ledger_path = base_dir / "ta_shadow_variant_history.csv"
    if not ledger_path.exists():
        raise SystemExit(f"TA shadow ledger missing: {ledger_path}")
    ledger_df = pd.read_csv(ledger_path)
    if not {"as_of_date", "variant"}.issubset(ledger_df.columns):
        raise SystemExit("TA shadow ledger must include as_of_date and variant columns")
    ledger_rows = ledger_df[ledger_df["as_of_date"].astype(str) == as_of_date]
    ledger_variants = set(ledger_rows["variant"].astype(str))
    missing_ledger = sorted(set(ta_variants) - ledger_variants)
    if missing_ledger:
        raise SystemExit(
            "missing TA ledger rows for "
            f"{as_of_date}: {', '.join(missing_ledger)}"
        )


def _verify_data_freshness(db_path: Path, reference_date: date) -> str:
    """Verify monitored data feeds are fresh for the monthly run."""
    conn = db_client.get_connection(str(db_path))
    try:
        db_client.initialize_schema(conn)
        report = db_client.check_data_freshness(conn, reference_date)
    finally:
        conn.close()
    if report["overall_status"] != "OK":
        raise SystemExit(
            "Data freshness warnings: " + "; ".join(report["warnings"])
        )
    return str(report["overall_status"])


def verify_monthly_outputs(
    *,
    base_dir: Path = Path("results") / "monthly_decisions",
    db_path: Path = Path(config.DB_PATH),
    as_of_date: str | None = None,
    reference_date: str | None = None,
) -> MonthlyVerificationResult:
    """Verify monthly report artifacts, freshness, and TA shadow ledger rows."""
    output_dir = _monthly_dir(base_dir, as_of_date)
    missing = [
        str(output_dir / filename)
        for filename in REQUIRED_MONTHLY_ARTIFACTS
        if not (output_dir / filename).exists()
    ]
    if missing:
        raise SystemExit("Missing monthly artifacts: " + ", ".join(missing))

    monthly_summary = _load_json(output_dir / "monthly_summary.json")
    manifest = _load_json(output_dir / "run_manifest.json")
    summary_as_of = str(monthly_summary.get("as_of_date") or as_of_date or "")
    if not summary_as_of:
        raise SystemExit("monthly_summary.json is missing as_of_date")
    if as_of_date is not None and summary_as_of != as_of_date:
        raise SystemExit(
            f"monthly_summary.json as_of_date {summary_as_of} != requested {as_of_date}"
        )

    ta_variants = _required_ta_variants(monthly_summary)
    _verify_ta_artifacts(
        base_dir=base_dir,
        output_dir=output_dir,
        as_of_date=summary_as_of,
        ta_variants=ta_variants,
    )
    freshness_reference = reference_date or str(monthly_summary.get("run_date") or summary_as_of)
    freshness_status = _verify_data_freshness(
        db_path=db_path,
        reference_date=date.fromisoformat(freshness_reference),
    )

    warnings = manifest.get("warnings", [])
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    return MonthlyVerificationResult(
        output_dir=output_dir,
        schema_version=manifest.get("schema_version") or monthly_summary.get("schema_version"),
        git_sha=str(manifest.get("git_sha")) if manifest.get("git_sha") is not None else None,
        warning_count=warning_count,
        freshness_status=freshness_status,
        ta_variants=ta_variants,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("results") / "monthly_decisions")
    parser.add_argument("--db-path", type=Path, default=Path(config.DB_PATH))
    parser.add_argument("--as-of", default=None, help="Monthly as-of date to verify")
    parser.add_argument("--reference-date", default=None, help="Freshness reference date")
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    result = verify_monthly_outputs(
        base_dir=args.base_dir,
        db_path=args.db_path,
        as_of_date=args.as_of,
        reference_date=args.reference_date,
    )
    text = "\n".join(result.summary_lines()) + "\n"
    if args.summary_path is not None:
        args.summary_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
