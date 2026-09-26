"""Production artifact paths (review 2026-09-25, section 5, phase 1).

Everything the scheduled workflows write and commit lives under
``artifacts/``; research output stays under ``results/``. Paths are relative
to the repository root, like ``DB_PATH``: the entry points run from the root,
and callers that need an absolute path join them to it.
"""

from __future__ import annotations

import os

__all__ = [
    "ARTIFACTS_DIR",
    "MONTHLY_DECISIONS_DIR",
    "DECISION_LOG_PATH",
    "SHADOW_REVIEWS_DIR",
    "CHARTS_DIR",
    "OPS_DIR",
    "FETCH_STATUS_PATH",
    "DRY_RUN_MONTHLY_DECISIONS_DIR",
]

ARTIFACTS_DIR: str = "artifacts"

# One folder per month (YYYY-MM) plus the append-only log and shadow ledgers.
MONTHLY_DECISIONS_DIR: str = os.path.join(ARTIFACTS_DIR, "monthly_decisions")
DECISION_LOG_PATH: str = os.path.join(MONTHLY_DECISIONS_DIR, "decision_log.md")

# v14 shadow-review memos (2025-11 .. 2026-04); no current writer.
SHADOW_REVIEWS_DIR: str = os.path.join(ARTIFACTS_DIR, "shadow_reviews")

# Monthly PGR charts, rewritten by the monthly decision workflow.
CHARTS_DIR: str = os.path.join(ARTIFACTS_DIR, "charts")

# Run logs written by workflows.
OPS_DIR: str = os.path.join(ARTIFACTS_DIR, "ops")
FETCH_STATUS_PATH: str = os.path.join(OPS_DIR, "fetch_status.md")

# `monthly_decision.py --dry-run` output: gitignored, never committed.
DRY_RUN_MONTHLY_DECISIONS_DIR: str = os.path.join("results", "dry_run", "monthly_decisions")
