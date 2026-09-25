"""Regression tests for review 2026-09-25 finding F14: ``--dry-run`` is read-only.

Before the fix, ``monthly_decision.py --dry-run`` rewrote the DB's
``model_performance_log`` row, added a ``model_retrain_log`` row, overwrote the
committed ``results/monthly_decisions/YYYY-MM/`` artifacts and appended to
``decision_log.md`` and the shadow ledgers. ``weekly_fetch.py --dry-run``
logged API requests and seeded splits.

Each test runs ``main(dry_run=True)`` in a temp copy of the repo layout (DB
copy plus the committed ``results/monthly_decisions`` tree) and asserts that
the DB, every copied file, and every tracked file in the real checkout are
byte-identical afterwards.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import config

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_DB = REPO_ROOT / "data" / "pgr_financials.db"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tree(root: Path, exclude: Path | None = None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if exclude is not None and exclude in path.parents:
            continue
        hashes[str(path.relative_to(root))] = _sha256(path)
    return hashes


def _hash_tracked_repo_files() -> dict[str, str]:
    """Hash every tracked file in the real checkout (catches absolute-path writes)."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    hashes: dict[str, str] = {}
    for rel in filter(None, listing.split("\0")):
        path = REPO_ROOT / rel
        if path.is_file():
            hashes[rel] = _sha256(path)
    return hashes


def _changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    keys = set(before) | set(after)
    return sorted(k for k in keys if before.get(k) != after.get(k))


@pytest.fixture()
def temp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp working directory laid out like the repo, with a DB copy."""
    root = tmp_path / "repo"
    (root / "data" / "processed").mkdir(parents=True)
    shutil.copy2(COMMITTED_DB, root / "data" / "pgr_financials.db")
    shutil.copytree(
        REPO_ROOT / "results" / "monthly_decisions",
        root / "results" / "monthly_decisions",
    )
    monkeypatch.chdir(root)
    monkeypatch.setattr(config, "DB_PATH", str(root / "data" / "pgr_financials.db"))
    for key in ("AV_API_KEY", "FRED_API_KEY", "FMP_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "AV_API_KEY", None, raising=False)
    monkeypatch.setattr(config, "FRED_API_KEY", None, raising=False)
    return root


def test_weekly_fetch_dry_run_leaves_db_and_files_unchanged(temp_repo: Path) -> None:
    from scripts import weekly_fetch

    db_path = temp_repo / "data" / "pgr_financials.db"
    db_before = _sha256(db_path)
    repo_before = _hash_tracked_repo_files()

    weekly_fetch.main(dry_run=True, skip_fred=True)

    assert _sha256(db_path) == db_before, "weekly_fetch --dry-run modified the DB"
    assert _changed(repo_before, _hash_tracked_repo_files()) == []


def test_weekly_fetch_dry_run_uses_read_only_connection(
    temp_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import weekly_fetch
    from src.database import db_client

    calls: list[bool] = []
    real_get_connection = db_client.get_connection

    def _spy(db_path=None, read_only: bool = False):
        calls.append(read_only)
        return real_get_connection(db_path, read_only=read_only)

    monkeypatch.setattr(weekly_fetch.db_client, "get_connection", _spy)
    weekly_fetch.main(dry_run=True, skip_fred=True)
    assert calls == [True]


@pytest.mark.slow
@pytest.mark.integration
def test_monthly_decision_dry_run_leaves_db_and_tracked_files_unchanged(
    temp_repo: Path,
) -> None:
    """Full ``main(dry_run=True)`` on a DB copy: nothing tracked may change."""
    from scripts import monthly_decision

    db_path = temp_repo / "data" / "pgr_financials.db"
    dry_run_root = temp_repo / "results" / "dry_run"
    db_before = _sha256(db_path)
    results_before = _hash_tree(temp_repo / "results")
    repo_before = _hash_tracked_repo_files()

    monthly_decision.main(as_of_date_str="2026-04-02", dry_run=True, skip_fred=True)

    assert _sha256(db_path) == db_before, "monthly_decision --dry-run modified the DB"
    changed_results = _changed(
        results_before,
        _hash_tree(temp_repo / "results", exclude=dry_run_root),
    )
    assert changed_results == [], f"dry run rewrote committed artifacts: {changed_results}"
    assert _changed(repo_before, _hash_tracked_repo_files()) == []

    out_dir = dry_run_root / "monthly_decisions" / "2026-04"
    assert (out_dir / "recommendation.md").exists()
    assert (out_dir / "signals.csv").exists()
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["artifact_classification"] == "dry_run"
    assert "nan_live_features" in manifest


def test_read_only_connection_rejects_writes(tmp_path: Path) -> None:
    from src.database import db_client

    db_path = tmp_path / "ro.db"
    conn = db_client.get_connection(str(db_path))
    db_client.initialize_schema(conn)
    conn.close()
    before = _sha256(db_path)

    ro = db_client.get_connection(str(db_path), read_only=True)
    try:
        assert ro.execute("SELECT COUNT(*) FROM pgr_edgar_monthly").fetchone()[0] == 0
        with pytest.raises(Exception, match="readonly"):
            db_client.upsert_splits(
                ro,
                [
                    {
                        "ticker": "PGR",
                        "split_date": "2006-05-19",
                        "split_ratio": 4.0,
                        "numerator": 4.0,
                        "denominator": 1.0,
                    }
                ],
            )
    finally:
        ro.close()
    assert _sha256(db_path) == before


def test_read_only_connection_requires_existing_db(tmp_path: Path) -> None:
    from src.database import db_client

    missing = tmp_path / "missing" / "nope.db"
    with pytest.raises(FileNotFoundError):
        db_client.get_connection(str(missing), read_only=True)
    assert not missing.parent.exists()
