"""Exact offline input and runtime pins for the v200 research baseline."""

from __future__ import annotations

import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import sqlite3
import subprocess
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_CODE_COMMIT = "c3b4798b90a43f9dbd616981861e9ff01fa2de72"
INPUT_DB_COMMIT = "ed7997f6f540f664e59dd44a4079616d74a8e8cb"
INPUT_DB_SHA256 = (
    "38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6"
)
PARENT_SEED_COMMIT = "aae0be883309bdc068ee9afbb78b0aa9a4b52b8d"
PARENT_SEED_SHA256 = (
    "7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d"
)
CORE_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "matplotlib",
    "xgboost",
    "pyarrow",
    "requests",
    "lxml",
    "beautifulsoup4",
    "python-dotenv",
    "skfolio",
    "PyPortfolioOpt",
    "mapie",
    "PyYAML",
)


def _full_commit(value: Any, field: str) -> str:
    """Reject mutable or abbreviated Git references."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(
            f"{field} must contain a full 40-character Git commit."
        )
    return value


def _sha256(value: Any, field: str) -> str:
    """Require an exact, lowercase SHA256 pin."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must contain a full lowercase SHA256.")
    return value


def _relative_path(value: Any, field: str) -> str:
    """Validate a portable repository-relative path without traversal."""
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{field} must be a nonempty repository-relative path."
        )
    windows = PureWindowsPath(value)
    portable = PurePosixPath(value.replace("\\", "/"))
    if (
        windows.drive
        or windows.root
        or portable.is_absolute()
        or ".." in portable.parts
        or ":" in value
        or "\x00" in value
        or portable == PurePosixPath(".")
    ):
        raise ValueError(f"{field} must be a repository-relative path.")
    return portable.as_posix()


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without opening a database connection."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(path: str | Path) -> str:
    """Hash source text as Git LF bytes, permitting only CRLF equivalence.

    CSV, parquet, model and other artifact bytes always use sha256_file.
    Windows automatic text checkout conversion is not a source-code change.
    """
    source = Path(path)
    if source.suffix not in {".py", ".md", ".yaml", ".toml"}:
        raise ValueError("Git line-ending equivalence is for source text only")
    return hashlib.sha256(
        source.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()


def _verify_file(path: Path, expected_sha: str) -> None:
    """Fail before consuming any file whose exact bytes differ."""
    expected = _sha256(expected_sha, "expected_sha")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected}, "
            f"observed {actual}."
        )


def read_immutable(path: str | Path, expected_sha: str) -> sqlite3.Connection:
    """Verify a DB, then open a read-only immutable SQLite connection.

    No journal, WAL, migration or cache is created by this function. The caller
    owns the connection and must close it. Research runners use external DB
    copies; immutable reads also cannot write to an existing tracked DB.
    """
    database = Path(path).resolve()
    _verify_file(database, expected_sha)
    connection = sqlite3.connect(
        database.as_uri() + "?mode=ro&immutable=1",
        uri=True,
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def export_git_blob(
    commit: str,
    relative_path: str,
    destination: str | Path,
    expected_sha: str,
) -> Path:
    """Export a hash-verified Git blob outside the repository data tree.

    Existing different files are preserved. Validation of both the Git ref and
    blob bytes precedes writing. Only the explicitly named destination is
    replaced, atomically, after validation.
    """
    pinned_commit = _full_commit(commit, "commit")
    source_path = _relative_path(relative_path, "relative_path")
    expected = _sha256(expected_sha, "expected_sha")
    target = Path(destination).resolve()
    data_root = (REPO_ROOT / "data").resolve()
    if target == data_root or target.is_relative_to(data_root):
        raise ValueError("Git exports cannot write anywhere under repo data/.")
    result = subprocess.run(
        ["git", "show", f"{pinned_commit}:{source_path}"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    actual = hashlib.sha256(result.stdout).hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for Git blob: expected {expected}, "
            f"observed {actual}."
        )
    if target.exists():
        if sha256_file(target) != expected:
            raise FileExistsError(
                f"Preserving different existing file: {target}"
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="v200-verified-",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(result.stdout)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    _verify_file(target, expected)
    return target


def verify_baseline_lock(lock_path: str | Path) -> dict[str, Any]:
    """Validate D7's clean lock and every consumed file's SHA256.

    The authoritative DB pin is its full Git commit and SHA256. An optional
    ``input_db.path`` is an execution convenience that is verified if present.
    ``input_files`` paths are relative to the repository root; they cannot pin
    a mutable external cache by pathname alone.
    """
    with Path(lock_path).open(encoding="utf-8") as stream:
        lock = json.load(stream)
    if not isinstance(lock, dict) or lock.get("status") != "clean":
        raise ValueError("A clean baseline lock is required before fitting.")
    if (
        type(lock.get("schema_version")) is not int
        or lock["schema_version"] != 1
    ):
        raise ValueError("Unsupported baseline lock schema_version.")
    baseline = _full_commit(
        lock.get("baseline_code_commit"), "baseline_code_commit"
    )
    if baseline != BASELINE_CODE_COMMIT:
        raise ValueError(
            "baseline_code_commit does not match the approved R3b pin."
        )
    database = lock.get("input_db")
    if not isinstance(database, dict):
        raise ValueError("input_db must pin the approved repaired database.")
    db_commit = _full_commit(database.get("git_commit"), "input_db.git_commit")
    db_sha = _sha256(database.get("sha256"), "input_db.sha256")
    db_relative = _relative_path(
        database.get("relative_path"), "input_db.relative_path"
    )
    if (db_commit, db_sha, db_relative) != (
        INPUT_DB_COMMIT,
        INPUT_DB_SHA256,
        "data/pgr_financials.db",
    ):
        raise ValueError(
            "input_db does not match the approved repair's exact pins."
        )
    seed = lock.get("parent_seed")
    if not isinstance(seed, dict):
        raise ValueError("parent_seed must preserve the original input pins.")
    seed_commit = _full_commit(
        seed.get("git_commit"), "parent_seed.git_commit"
    )
    seed_sha = _sha256(seed.get("db_sha256"), "parent_seed.db_sha256")
    if (seed_commit, seed_sha) != (PARENT_SEED_COMMIT, PARENT_SEED_SHA256):
        raise ValueError("parent_seed does not match the original user pins.")
    repairs = lock.get("repairs")
    if not isinstance(repairs, list) or not repairs:
        raise ValueError("repair IDs R2-lite and R3b are required.")
    repair_ids: set[str] = set()
    for repair in repairs:
        if not isinstance(repair, dict) or not isinstance(
            repair.get("id"), str
        ):
            raise ValueError(
                "Each repair must have an ID and full Git commit."
            )
        _full_commit(repair.get("commit"), "repairs.commit")
        if repair["id"] in repair_ids:
            raise ValueError("Duplicate repair IDs are not allowed.")
        repair_ids.add(repair["id"])
    if not {"R2-lite", "R3b"}.issubset(repair_ids):
        raise ValueError("repair IDs R2-lite and R3b are required.")
    exceptions = lock.get("accepted_exceptions")
    if not isinstance(exceptions, list) or not any(
        isinstance(item, dict)
        and item.get("id") == "VWO_MARCH_2026_GAP"
        and isinstance(item.get("description"), str)
        and item["description"].strip()
        for item in exceptions
    ):
        raise ValueError("The accepted VWO_MARCH_2026_GAP must be disclosed.")
    files = lock.get("input_files")
    if not isinstance(files, list) or not files:
        raise ValueError("Consumed input_files must have exact SHA256 pins.")
    seen_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each input_files entry needs a path and SHA256.")
        relative = _relative_path(item.get("path"), "input_files.path")
        if relative in seen_paths:
            raise ValueError("Duplicate input_files paths are not allowed.")
        seen_paths.add(relative)
        path = (REPO_ROOT / relative).resolve()
        if not path.is_relative_to(REPO_ROOT.resolve()):
            raise ValueError(
                "input_files.path must resolve inside the repository."
            )
        basis = item.get("hash_basis", "exact_bytes")
        if basis == "git_source_lf":
            expected = _sha256(item.get("sha256"), "source sha256")
            if source_sha256(path) != expected:
                raise ValueError(
                    f"SHA256 mismatch for source text: {relative}"
                )
        elif basis == "exact_bytes":
            _verify_file(path, item.get("sha256"))
        else:
            raise ValueError("Unknown input SHA256 basis")
    if database.get("path") is not None:
        _verify_file(Path(database["path"]).resolve(), db_sha)
    return lock


def runtime_lock() -> dict[str, Any]:
    """Record exact dependency versions and the installed namespace path."""
    specification = util.find_spec("pgr_vds")
    if specification is None or specification.origin is None:
        raise RuntimeError(
            "Install pgr_vds before recording research provenance."
        )
    packages: dict[str, str] = {}
    for package in CORE_DEPENDENCIES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"Required research dependency missing: {package}"
            ) from error
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "used_core_dependencies": list(CORE_DEPENDENCIES),
        "pgr_vds_path": str(Path(specification.origin).resolve()),
    }


def verify_runtime(lock: dict[str, Any]) -> None:
    """Reject silent Python or core dependency changes before a later run.

    Platform and install path are recorded for attribution, but can differ
    between machines. Python and every declared used dependency must match.
    """
    if not isinstance(lock, dict):
        raise ValueError("Runtime lock must be a mapping.")
    current = runtime_lock()
    for key in ("python", "implementation"):
        if lock.get(key) != current[key]:
            raise ValueError(
                f"Runtime mismatch for {key}: {lock.get(key)} "
                f"!= {current[key]}"
            )
    if lock.get("used_core_dependencies") != current["used_core_dependencies"]:
        raise ValueError(
            "Runtime lock must freeze every used core dependency."
        )
    packages = lock.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("Runtime lock must contain exact package versions.")
    for package in CORE_DEPENDENCIES:
        if packages.get(package) != current["packages"][package]:
            raise ValueError(
                f"Runtime mismatch for {package}: {packages.get(package)} != "
                f"{current['packages'][package]}"
            )
