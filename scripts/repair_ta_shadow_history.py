"""Deterministic R4 monitoring repair using external DB and ledger copies.

Run from an installed checkout:
``python -m scripts.repair_ta_shadow_history``.
All four paths must be outside Git checkouts; inputs are never overwritten.
This command reads cached targets only and does not run a decision or model.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from src.models.classification_monitoring import attach_matured_ta_outcomes

REPO_ROOT = Path(__file__).resolve().parents[1]
MONITORING_COLUMNS = (
    "mature_on_date",
    "is_horizon_mature",
    "actual_actionable_sell",
    "actual_basket_relative_return",
)


def _external_path(path: Path) -> Path:
    """Reject checkout-local paths and resolved symlink targets."""
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT) or any(
        (parent / ".git").exists() for parent in resolved.parents
    ):
        raise ValueError("Repair paths must be outside repository checkouts")
    return resolved


def repair_copies(
    *,
    db_copy: Path,
    ledger_copy: Path,
    output: Path,
    row_diff: Path,
    as_of: date,
) -> None:
    """Write a repaired copy and cell diff without modifying inputs."""
    db_copy, ledger_copy, output, row_diff = (
        _external_path(path)
        for path in (db_copy, ledger_copy, output, row_diff)
    )
    if len({db_copy, ledger_copy, output, row_diff}) != 4:
        raise ValueError("Inputs, output and row diff must be distinct paths")
    before = pd.read_csv(ledger_copy)
    if before.duplicated(["as_of_date", "variant"]).any():
        raise ValueError(
            "Duplicate issuance keys require explicit manual reconciliation"
        )
    conn = sqlite3.connect(db_copy.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        after = attach_matured_ta_outcomes(conn, before, as_of=as_of)
    finally:
        conn.close()
    changes: list[dict[str, object]] = []
    for idx, row in after.iterrows():
        for column in MONITORING_COLUMNS:
            old = before.at[idx, column] if column in before else pd.NA
            new = row[column]
            if (pd.isna(old) and pd.isna(new)) or (
                not pd.isna(old) and not pd.isna(new) and old == new
            ):
                continue
            changes.append(
                {
                    "as_of_date": row["as_of_date"],
                    "variant": row["variant"],
                    "column": column,
                    "before": old,
                    "after": new,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    row_diff.parent.mkdir(parents=True, exist_ok=True)
    after.to_csv(output, index=False)
    pd.DataFrame(
        changes,
        columns=[
            "as_of_date",
            "variant",
            "column",
            "before",
            "after",
        ],
    ).to_csv(row_diff, index=False)


def main() -> None:
    """Parse explicit copy paths and evaluation date; no default live paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-copy", type=Path, required=True)
    parser.add_argument("--ledger-copy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-diff", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    repair_copies(
        db_copy=args.db_copy,
        ledger_copy=args.ledger_copy,
        output=args.output,
        row_diff=args.row_diff,
        as_of=args.as_of,
    )


if __name__ == "__main__":
    main()
