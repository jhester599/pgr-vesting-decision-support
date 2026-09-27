"""Review 2026-09-25, step 9 (WP12, F28): test-suite hygiene.

- The repo guard (``tests/repo_guard.py``, installed by ``conftest.py``)
  fails tests that write inside the repository or open the committed
  database, and redirects ``config.DB_PATH`` and the feature-matrix cache.
- Stored-artifact tests carry ``@pytest.mark.artifact`` and run in their own
  CI job.
- ``tests/integration/pipeline/test_engine_smoke.py`` uses seeds that do not depend on
  ``PYTHONHASHSEED``.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tests import repo_guard

REPO_ROOT = repo_guard.REPO_ROOT
_WRITE = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


# ---------------------------------------------------------------------------
# Repo guard: the classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("event", "args", "allow", "refused"),
    [
        ("open", (str(REPO_ROOT / "data/processed/feature_matrix.parquet"), "wb", _WRITE), False, True),
        ("open", ("data/processed/feature_matrix.parquet", "w", _WRITE), False, True),
        ("open", (str(REPO_ROOT / "results/x.csv"), None, _WRITE), False, True),
        ("open", (str(REPO_ROOT / "results/x.csv"), "r", os.O_RDONLY), False, False),
        ("open", (str(REPO_ROOT / "src/__pycache__/m.pyc"), "wb", _WRITE), False, False),
        ("open", (str(REPO_ROOT / ".pytest_cache/v/x"), "w", _WRITE), False, False),
        ("open", (str(repo_guard.COMMITTED_DB), "rb", os.O_RDONLY), False, True),
        ("open", (str(repo_guard.COMMITTED_DB), "rb", os.O_RDONLY), True, False),
        ("open", (str(repo_guard.COMMITTED_DB), "r+b", os.O_RDWR), True, True),
        ("sqlite3.connect", (str(repo_guard.COMMITTED_DB),), False, True),
        ("sqlite3.connect", (str(repo_guard.COMMITTED_DB),), True, True),
        ("sqlite3.connect", (f"{repo_guard.COMMITTED_DB.as_uri()}?mode=ro",), False, True),
        ("sqlite3.connect", (f"{repo_guard.COMMITTED_DB.as_uri()}?mode=ro",), True, False),
        ("sqlite3.connect", (":memory:",), False, False),
        ("sqlite3.connect", (str(REPO_ROOT / "data" / "other.db"),), True, True),
        ("os.remove", (str(REPO_ROOT / "results" / "x.csv"), -1), False, True),
        ("os.rename", ("/tmp/a", str(REPO_ROOT / "results" / "x.csv"), -1, -1), False, True),
        ("open", ("/tmp/elsewhere.txt", "w", _WRITE), False, False),
        ("open", (3, "w", _WRITE), False, False),
    ],
)
def test_classify_access(event: str, args: tuple, allow: bool, refused: bool) -> None:
    assert (repo_guard.classify_access(event, args, allow) is not None) is refused


def test_existing_directory_mkdir_is_not_a_write() -> None:
    assert repo_guard.classify_access("os.mkdir", (str(REPO_ROOT / "data"), 0o777, -1), False) is None
    assert repo_guard.classify_access("os.mkdir", (str(REPO_ROOT / "data" / "new_dir"), 0o777, -1), False)


@pytest.mark.parametrize("allow", [False, True])
def test_committed_db_spellings_obey_read_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow: bool,
) -> None:
    root = tmp_path / "repo with spaces"
    root.mkdir()
    db = root / "data" / "pgr_financials.db"
    monkeypatch.chdir(root)
    monkeypatch.setattr(repo_guard, "REPO_ROOT", root)
    monkeypatch.setattr(repo_guard, "COMMITTED_DB", db)
    native = str(db).swapcase() if os.name == "nt" else str(db)
    for raw in (str(db), native, "data/pgr_financials.db"):
        assert repo_guard.classify_access("sqlite3.connect", (raw,), allow)
        assert bool(
            repo_guard.classify_access(
                "open",
                (raw, "rb", os.O_RDONLY),
                allow,
            )
        ) is (not allow)
        assert repo_guard.classify_access(
            "open", (raw, "r+b", os.O_RDWR), allow
        )
    uris = (
        db.as_uri(),
        db.as_uri().replace("file:///", "file://localhost/"),
        "file:data/pgr_financials.db",
        "file:" + db.as_posix(),
    )
    assert "%20" in uris[0]
    for uri in uris:
        for query in ("", "?mode=rw", "?mode=ro", "?mode=ro&immutable=1"):
            refused = repo_guard.classify_access(
                "sqlite3.connect",
                (uri + query,),
                allow,
            )
            assert bool(refused) is (not (allow and "mode=ro" in query)), (
                uri + query
            )


def test_file_uri_preserves_authority_and_platform_drive_semantics() -> None:
    path, read_only = repo_guard._as_path(
        "file://server/share/a%20b.db?mode=ro"
    )
    assert path == Path("//server/share/a b.db")
    assert read_only
    path, _ = repo_guard._as_path("file:///C:/data/a%20b.db")
    expected = "C:/data/a b.db" if os.name == "nt" else "/C:/data/a b.db"
    assert path == Path(expected)


@pytest.mark.parametrize(
    "raw",
    [
        ":memory:",
        "file::memory:",
        "file::memory:?mode=ro",
        "file:cache?mode=memory",
    ],
)
def test_memory_db_is_allowed(raw: str) -> None:
    assert repo_guard.classify_access("sqlite3.connect", (raw,), False) is None


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize("last_mode", ["ro", "rw"])
def test_repeated_sqlite_modes_cannot_hide_committed_db_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow: bool,
    last_mode: str,
) -> None:
    db = tmp_path / "synthetic.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE sample (x INTEGER)")
        conn.execute("INSERT INTO sample VALUES (23)")
        conn.commit()
    finally:
        conn.close()
    uri = db.as_uri() + f"?mode=memory&mode={last_mode}"
    # Independently show this URI actually reads a disk DB, not memory.
    conn = sqlite3.connect(uri, uri=True)
    try:
        assert conn.execute("SELECT x FROM sample").fetchone() == (23,)
    finally:
        conn.close()
    monkeypatch.setattr(repo_guard, "COMMITTED_DB", db)
    assert repo_guard.classify_access("sqlite3.connect", (uri,), allow)


@pytest.mark.parametrize("raw", [":memory:", "file:cache?mode=memory"])
def test_native_file_writes_do_not_get_sqlite_memory_exemption(
    raw: str,
) -> None:
    # These are ordinary native filenames on POSIX, not SQLite connections.
    assert repo_guard.classify_access("open", (raw, "w", _WRITE), False)


def test_external_db_paths_are_allowed(tmp_path: Path) -> None:
    external = tmp_path / "external db.sqlite"
    for raw in (str(external), external.as_uri() + "?mode=ro&immutable=1"):
        assert (
            repo_guard.classify_access("sqlite3.connect", (raw,), False)
            is None
        )


# ---------------------------------------------------------------------------
# Repo guard: end to end on the probe tests
# ---------------------------------------------------------------------------

_EXPECTED_PROBES = {
    "test_probe_write_inside_repo": "failed",
    "test_probe_swallowed_write_inside_repo": "failed",
    "test_probe_mkdir_inside_repo": "failed",
    "test_probe_unmarked_read_only_db_open": "failed",
    "test_probe_unmarked_relative_db_open": "failed",
    "test_probe_unmarked_native_case_db_open": "failed",
    "test_probe_swallowed_read_only_db_open": "failed",
    "test_probe_unmarked_plain_file_read_of_db": "failed",
    "test_probe_artifact_read_write_db_open": "failed",
    "test_probe_artifact_read_write_uri_db_open": "failed",
    "test_probe_artifact_read_only_db_open": "passed",
    "test_probe_artifact_committed_db_copy": "passed",
    "test_probe_default_paths_are_in_tmp": "passed",
    "test_probe_writes_outside_repo_are_allowed": "passed",
}


def _probe_outcomes(output: str) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for line in output.splitlines():
        line = line.replace("\\", "/")
        match = re.match(
            r"^(PASSED|FAILED|ERROR) tests/guard_probe\.py::(\w+)", line
        )
        if match:
            status = "passed" if match.group(1) == "PASSED" else "failed"
            # A teardown error after a pass still fails the test.
            if outcomes.get(match.group(2)) != "failed":
                outcomes[match.group(2)] = status
    return outcomes


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_probe_parser_preserves_teardown_failure(separator: str) -> None:
    output = "\n".join(
        f"{status} tests{separator}guard_probe.py::{name}"
        for status, name in (
            ("PASSED", "test_probe_swallowed_write_inside_repo"),
            ("ERROR", "test_probe_swallowed_write_inside_repo"),
            ("PASSED", "test_probe_swallowed_write_inside_repo"),
            ("PASSED", "test_probe_writes_outside_repo_are_allowed"),
        )
    )
    assert _probe_outcomes(output) == {
        "test_probe_swallowed_write_inside_repo": "failed",
        "test_probe_writes_outside_repo_are_allowed": "passed",
    }


@pytest.mark.skipif(not repo_guard.COMMITTED_DB.exists(), reason="committed DB not present")
def test_guard_fails_exactly_the_probes_that_touch_the_repo(tmp_path: Path) -> None:
    before = {p: p.stat().st_mtime_ns for p in repo_guard.COMMITTED_DB.parent.glob("pgr_financials.db*")}
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/guard_probe.py", "-rA", "-q",
            "-p", "no:cacheprovider", "-o", "addopts=", f"--basetemp={tmp_path / 'probe'}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    outcomes = _probe_outcomes(proc.stdout)
    assert proc.returncode == 1, proc.stdout[-4000:] + proc.stderr
    assert outcomes == _EXPECTED_PROBES, proc.stdout[-4000:]
    assert "repo guard" in proc.stdout
    # Nothing was created and the committed DB (and its sidecars) is untouched.
    # If the guard ever regresses, remove what the probes wrote before failing.
    residue = [
        path
        for path in (REPO_ROOT / "results" / "guard_probe_wp12.txt", REPO_ROOT / "results" / "guard_probe_wp12_dir")
        if path.exists()
    ]
    for path in residue:
        path.rmdir() if path.is_dir() else path.unlink()
    assert residue == []
    after = {p: p.stat().st_mtime_ns for p in repo_guard.COMMITTED_DB.parent.glob("pgr_financials.db*")}
    assert after == before


def test_hypothesis_storage_is_outside_the_repo() -> None:
    from hypothesis.configuration import storage_directory

    assert REPO_ROOT not in storage_directory("examples", intent_to_write=False).resolve().parents


# ---------------------------------------------------------------------------
# Artifact tests run in their own CI job
# ---------------------------------------------------------------------------

def test_ci_runs_artifact_tests_in_a_separate_job() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r'python -m pytest -q -m "not artifact and not research"', ci)
    assert re.search(r'python -m pytest -q -m artifact', ci)
    jobs = re.findall(r"^  ([a-z_-]+):\s*$", ci, flags=re.MULTILINE)
    assert "artifacts" in jobs and "test" in jobs


def test_artifact_marker_is_registered() -> None:
    import tomllib

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("artifact:") for m in markers)


# ---------------------------------------------------------------------------
# Deterministic seeds in tests/integration/pipeline/test_engine_smoke.py
# ---------------------------------------------------------------------------

def test_integration_prices_do_not_depend_on_the_hash_seed() -> None:
    """``hash(ticker)`` is salted per process, so the synthetic prices (and
    the WFO results on them) changed on every run."""
    code = (
        "import hashlib, json;"
        "from tests.integration.pipeline.test_engine_smoke import _generate_prices;"
        "rows = _generate_prices('VTI')[:50];"
        "print(hashlib.sha256(json.dumps(rows).encode()).hexdigest())"
    )
    digests = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120
        )
        assert out.returncode == 0, out.stderr[-2000:]
        digests.add(out.stdout.strip().splitlines()[-1])
    assert len(digests) == 1
