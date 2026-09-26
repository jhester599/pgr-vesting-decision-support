"""Monthly decision report generator for PGR RSU vesting decisions (entry point).

The logic lives in ``pgr_vds.decision`` (``pipeline.main`` runs it); this file
only parses the command line. Needs ``pip install -e .``.

Usage:
    python cli/monthly_decision.py [--as-of YYYY-MM-DD] [--dry-run] [--skip-fred]

Options:
    --as-of YYYY-MM-DD  Override the as-of date; must not be later than today.
                        Use for back-dated runs and testing.
    --dry-run           Read-only run: no HTTP calls, no DB writes, no ledger
                        appends. Artifacts go to
                        results/dry_run/monthly_decisions/YYYY-MM/.
    --skip-fred         Skip the FRED data fetch step. Useful when
                        FRED_API_KEY is not set or during testing.
"""

from __future__ import annotations

import argparse
import sys

from pgr_vds.decision.pipeline import main as run_monthly_decision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="PGR v3.0 monthly decision report generator."
    )
    parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        help=(
            "Override the as-of date; must not be later than today (default: today, "
            "or from the 20th the last business day on or before the 20th)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Read-only run: no HTTP calls, no DB writes, no ledger appends. "
            "Artifacts go to results/dry_run/monthly_decisions/YYYY-MM/."
        ),
    )
    parser.add_argument(
        "--skip-fred",
        action="store_true",
        help="Skip the FRED data fetch step.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the monthly decision from the command line."""
    # UTF-8 console output (the report text has non-ASCII symbols). This used
    # to happen as a side effect of importing
    # results/research/v46_classification.py (review 2026-09-25, F30).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    run_monthly_decision(
        as_of_date_str=args.as_of,
        dry_run=args.dry_run,
        skip_fred=args.skip_fred,
    )


if __name__ == "__main__":
    main()
