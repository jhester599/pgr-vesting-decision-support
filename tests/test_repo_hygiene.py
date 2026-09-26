"""Regression tests for review 2026-09-25 finding F29 (repository hygiene)."""

from __future__ import annotations

import sqlite3
import subprocess
import tomllib
from pathlib import Path

import pytest

from src.database import db_client

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
COMMITTED_DB = REPO_ROOT / "data" / "pgr_financials.db"


def _is_ignored(rel_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", rel_path],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "rel_path",
    [
        "data/processed/pgr_edgar_cache.csv",
        "data/processed/pgr_combined_ratio_manual.csv",
        "data/processed/pgr_valuation_monthly.csv",
    ],
)
def test_gitignore_negations_reinclude_reference_csvs(rel_path: str) -> None:
    assert not _is_ignored(rel_path)


@pytest.mark.parametrize(
    "rel_path",
    [
        "data/processed/feature_matrix.parquet",
        "data/processed/position_lots.csv",
        "data/pgr_financials.db-wal",
        "data/pgr_financials.db-shm",
        "results/dry_run/monthly_decisions/2026-04/recommendation.md",
    ],
)
def test_gitignore_ignores_caches_sidecars_and_dry_run_output(rel_path: str) -> None:
    assert _is_ignored(rel_path)


def test_committed_db_is_tracked_not_ignored() -> None:
    assert not _is_ignored("data/pgr_financials.db")


def test_pytest_config_does_not_add_quiet_flag() -> None:
    # pytest config moved from pytest.ini to pyproject.toml (review section 5, phase 0).
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = pyproject["tool"]["pytest"]["ini_options"].get("addopts", "").split()
    assert "-q" not in addopts and "-qq" not in addopts


def _db_commit_steps() -> list[tuple[str, list[str]]]:
    """Return (workflow, run-block lines) for every step that commits the DB.

    Parsed as text (steps start at ``- name:``) to avoid a YAML dependency.
    """
    steps: list[tuple[str, list[str]]] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        current: list[str] = []
        blocks: list[list[str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("- name:"):
                blocks.append(current)
                current = []
            current.append(line.strip())
        blocks.append(current)
        for block in blocks:
            if any(l.startswith("git add") and "pgr_financials.db" in l for l in block):
                steps.append((path.name, block))
    return steps


def test_every_db_commit_step_finalizes_db_before_git_add() -> None:
    steps = _db_commit_steps()
    assert steps, "expected at least one workflow that commits the DB"
    offenders: list[str] = []
    for name, lines in steps:
        first_add = next(i for i, line in enumerate(lines) if line.startswith("git add"))
        finalize = [
            i for i, line in enumerate(lines) if line.startswith("python scripts/finalize_db.py")
        ]
        if not finalize or finalize[0] > first_add:
            offenders.append(name)
    assert offenders == [], f"workflows commit the DB without finalize_db.py first: {offenders}"


def _journal_header_bytes(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:20]
    return header[18], header[19]


def test_finalize_for_commit_checkpoints_wal_and_sets_delete_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "wal.db"
    conn = db_client.get_connection(str(db_path))  # read-write -> WAL mode
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(100)])
    conn.commit()
    assert _journal_header_bytes(db_path) == (2, 2)  # WAL
    conn.close()

    # Re-open read-write so there is WAL content to fold in.
    conn = db_client.get_connection(str(db_path))
    conn.execute("INSERT INTO t VALUES (100)")
    conn.commit()
    conn.close()

    assert db_client.finalize_for_commit(str(db_path)) == "delete"
    assert _journal_header_bytes(db_path) == (1, 1)  # rollback journal
    wal = Path(f"{db_path}-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    check = sqlite3.connect(db_path)
    try:
        assert check.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 101
    finally:
        check.close()


def test_committed_db_is_not_in_wal_mode() -> None:
    """The committed DB must be self-contained (checkpointed, DELETE journal)."""
    assert _journal_header_bytes(COMMITTED_DB) == (1, 1)
