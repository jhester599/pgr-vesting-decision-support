"""
Shared pytest fixtures for the PGR Vesting Decision Support test suite.
"""

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis.configuration import set_hypothesis_home_dir

# Ensure the project root is on sys.path so src.* imports resolve correctly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests import repo_guard  # noqa: E402

# Hypothesis keeps its example database in ``.hypothesis/`` under the working
# directory unless told otherwise; keep it out of the repository (F28).
set_hypothesis_home_dir(Path(tempfile.gettempdir()) / "pgr-vds-hypothesis")
repo_guard.install()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the optional fast-mode switch for local iteration."""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Skip tests marked slow for faster local feedback.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip slow tests when the caller explicitly opts into fast mode."""
    if not config.getoption("--fast"):
        return

    skip_slow = pytest.mark.skip(reason="skipped by --fast")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _edgar_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test an EDGAR User-Agent (review 2026-09-25, F26).

    EDGAR calls now fail without one. Tests never reach EDGAR (requests are
    mocked); tests of the missing-agent error delete it themselves.
    """
    if not os.getenv("EDGAR_USER_AGENT"):
        monkeypatch.setenv("EDGAR_USER_AGENT", "pytest suite pytest@example.invalid")


# ---------------------------------------------------------------------------
# Repository isolation (review 2026-09-25, F28, step 9)
# ---------------------------------------------------------------------------

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> Iterator[None]:
    """Start the repo guard before any fixture of the test is set up."""
    repo_guard.start(item.nodeid, item.get_closest_marker("artifact") is not None)
    return (yield)


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    """Stop the guard after teardown; fail the test if it touched the repo.

    The guard already raised ``PermissionError`` at each refused access;
    this catches tests that swallowed it.
    """
    result = yield
    violations = repo_guard.stop()
    if violations:
        listed = "\n  ".join(dict.fromkeys(violations))
        pytest.fail(f"repo guard: {item.nodeid} touched the repository:\n  {listed}", pytrace=False)
    return result


@pytest.fixture(autouse=True)
def _isolate_repo_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the default DB, the raw-response cache and the feature-matrix
    cache at ``tmp_path``.

    ``config.DB_PATH`` names a file that does not exist, so code that falls
    back to the default database gets an empty one instead of the committed
    ``data/pgr_financials.db``. Tests that need committed data are marked
    ``artifact`` and use ``committed_db_copy``.
    """
    import config
    from src.processing import feature_engineering

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "pgr_financials.db"))
    # API clients create ``data/raw`` before writing their caches.
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(config, "DATA_RAW_DIR", str(raw_dir))
    monkeypatch.setattr(config, "REQUEST_COUNTS_FILE", str(raw_dir / ".request_counts.json"))
    monkeypatch.setattr(
        feature_engineering, "_PROCESSED_PATH", str(tmp_path / "feature_matrix.parquet")
    )


@pytest.fixture
def committed_db_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the committed database, set as every default DB path.

    Only ``artifact`` tests may read the committed file, so only they can
    use this fixture. Code under test can open the copy read-write (WAL
    mode, migrations) without touching the repository.
    """
    import config
    from src.research import v37_utils

    target = tmp_path / "committed_copy" / "pgr_financials.db"
    target.parent.mkdir()
    shutil.copyfile(repo_guard.COMMITTED_DB, target)
    monkeypatch.setattr(config, "DB_PATH", str(target))
    monkeypatch.setattr(v37_utils, "DB_PATH", target)
    return target
