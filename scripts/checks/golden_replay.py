"""Golden-output check: replay one monthly dry run on two commits and compare.

Review 2026-09-25, section 5, phase 5 (the monolith split) had to be a pure
refactor. This script proves it the same way for any later refactor:

1. check out ``--base`` and ``--head`` into temporary git worktrees;
2. copy the same DB file (``--db``, a copy, never the committed DB) into
   both, and record its sha256;
3. run the monthly decision dry run (``--as-of``, ``--skip-fred``) in each,
   with ``PYTHONPATH`` pointing at that worktree so its own code runs;
4. compare the month folders: every output file must be byte-identical, and
   ``run_manifest.json`` may differ only in the fields that describe the run
   itself (``git_sha``, ``run_timestamp_utc``, plus any ``--allow-field``);
5. check that the DB copy's sha256 is unchanged after both runs.

The entry point is ``cli/monthly_decision.py`` when the commit has it, else
the pre-phase-5 ``scripts/monthly_decision.py``.

Usage:
    python scripts/checks/golden_replay.py --base master --head HEAD \\
        --db /tmp/copy/pgr_financials.db --as-of 2026-09-21 \\
        --allow-field script_name

``compare_month_folders`` is importable for tests. Exit code 0 when the
outputs match, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# run_manifest.json fields that record the run, not its result.
RUN_FIELDS: frozenset[str] = frozenset({"git_sha", "run_timestamp_utc"})
MANIFEST = "run_manifest.json"
ENTRY_POINTS: tuple[str, ...] = ("cli/monthly_decision.py", "scripts/monthly_decision.py")
DRY_RUN_DIR = Path("results") / "dry_run" / "monthly_decisions"


def sha256(path: Path) -> str:
    """sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry_point(checkout: Path) -> str:
    """The monthly decision entry point of a checkout (relative path)."""
    for rel in ENTRY_POINTS:
        if (checkout / rel).is_file():
            return rel
    raise FileNotFoundError(f"no monthly decision entry point in {checkout}")


def _files(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def compare_month_folders(
    before: Path,
    after: Path,
    allowed_manifest_fields: frozenset[str] = RUN_FIELDS,
) -> tuple[list[str], list[str]]:
    """Compare two dry-run month folders.

    Returns ``(identical, problems)``: the files that are byte-identical, and
    one message per difference. ``run_manifest.json`` counts as identical
    when it differs only in ``allowed_manifest_fields``; those differences are
    not problems.
    """
    identical: list[str] = []
    problems: list[str] = []
    before_files, after_files = _files(before), _files(after)
    for rel in sorted(before_files - after_files):
        problems.append(f"missing after: {rel}")
    for rel in sorted(after_files - before_files):
        problems.append(f"new after: {rel}")
    for rel in sorted(before_files & after_files):
        a, b = before / rel, after / rel
        if a.read_bytes() == b.read_bytes():
            identical.append(rel)
            continue
        if rel != MANIFEST:
            problems.append(f"differs: {rel}")
            continue
        left = json.loads(a.read_text(encoding="utf-8"))
        right = json.loads(b.read_text(encoding="utf-8"))
        changed = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        unexpected = [key for key in changed if key not in allowed_manifest_fields]
        if unexpected:
            problems.append(f"differs: {rel} fields {unexpected}")
    return identical, problems


def _run_dry_run(checkout: Path, as_of: str) -> Path:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # The checkout's own `src`, `config` and `pgr_vds` come before any
    # installed copy of the package.
    env["PYTHONPATH"] = os.pathsep.join([str(checkout), str(checkout / "src")])
    subprocess.run(
        [sys.executable, entry_point(checkout), "--dry-run", "--as-of", as_of, "--skip-fred"],
        cwd=checkout,
        env=env,
        check=True,
    )
    return checkout / DRY_RUN_DIR / as_of[:7]


def _worktree(ref: str, path: Path) -> None:
    subprocess.run(
        ["git", "worktree", "add", "--detach", "--force", str(path), ref],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="Commit before the change (e.g. master).")
    parser.add_argument("--head", default="HEAD", help="Commit after the change.")
    parser.add_argument("--db", required=True, help="A copy of data/pgr_financials.db.")
    parser.add_argument("--as-of", required=True, help="As-of date, YYYY-MM-DD.")
    parser.add_argument(
        "--allow-field",
        action="append",
        default=[],
        help="Extra run_manifest.json field allowed to differ (repeatable).",
    )
    parser.add_argument("--keep", action="store_true", help="Keep the worktrees.")
    args = parser.parse_args(argv)

    db = Path(args.db).resolve()
    if db == (REPO_ROOT / "data" / "pgr_financials.db").resolve():
        parser.error("--db must be a copy, not the committed DB")
    db_sha = sha256(db)
    print(f"[golden] DB copy sha256 {db_sha}")

    work = Path(tempfile.mkdtemp(prefix="golden-replay-"))
    folders: dict[str, Path] = {}
    try:
        for label, ref in (("base", args.base), ("head", args.head)):
            checkout = work / label
            _worktree(ref, checkout)
            shutil.copyfile(db, checkout / "data" / "pgr_financials.db")
            print(f"[golden] {label} ({ref}): {entry_point(checkout)}", flush=True)
            folders[label] = _run_dry_run(checkout, args.as_of)
            after_sha = sha256(checkout / "data" / "pgr_financials.db")
            if after_sha != db_sha:
                print(f"[golden] {label}: DB changed during the dry run ({after_sha})")
                return 1
        identical, problems = compare_month_folders(
            folders["base"],
            folders["head"],
            RUN_FIELDS | frozenset(args.allow_field),
        )
        for rel in identical:
            print(f"[golden] identical: {rel}")
        for message in problems:
            print(f"[golden] {message}")
        print(f"[golden] {len(identical)} identical, {len(problems)} problems")
        return 1 if problems else 0
    finally:
        if not args.keep:
            for label in folders.keys() | {"base", "head"}:
                if (work / label).exists():
                    subprocess.run(
                        ["git", "worktree", "remove", "--force", str(work / label)],
                        cwd=REPO_ROOT,
                        check=False,
                        capture_output=True,
                    )
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
