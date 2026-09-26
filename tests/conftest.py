"""
Shared pytest fixtures for the PGR Vesting Decision Support test suite.
"""

import os
import sys

import pytest

# Ensure the project root is on sys.path so src.* imports resolve correctly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
