"""Run a production entry point with the network mocked (CI smoke tests).

Review 2026-09-25, F26: CI's "dry run" of ``edgar_8k_fetcher.py`` still
downloaded the EDGAR submissions list and filings, and the other smoke runs
could reach Alpha Vantage or FRED. This wrapper runs one script with:

- every socket connection refused (``NetworkBlockedError``), so any real
  request fails the smoke test instead of reaching an API;
- ``requests`` answered from canned responses: the SEC submissions index
  returns an empty filing list, so the 8-K fetcher exercises its code path
  without a download. Any other URL raises ``NetworkBlockedError``.

Usage:
    python scripts/ci_offline_smoke.py scripts/edgar_8k_fetcher.py --dry-run
"""

from __future__ import annotations

import json
import runpy
import socket
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from requests.models import Response

REPO_ROOT = Path(__file__).resolve().parent.parent


class NetworkBlockedError(RuntimeError):
    """Raised when a smoke-tested script tries to reach the network."""


def _canned_response(url: str) -> Response | None:
    parsed = urlparse(url)
    if parsed.netloc == "data.sec.gov" and parsed.path.startswith("/submissions/"):
        payload: dict[str, Any] = {
            "cik": "80661",
            "name": "PROGRESSIVE CORP/OH/",
            "filings": {
                "recent": {
                    "accessionNumber": [],
                    "filingDate": [],
                    "reportDate": [],
                    "form": [],
                    "primaryDocument": [],
                    "items": [],
                },
                "files": [],
            },
        }
        response = Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode("utf-8")
        response.headers["Content-Type"] = "application/json"
        response.url = url
        response.encoding = "utf-8"
        return response
    return None


def install_network_guard() -> list[str]:
    """Block sockets and serve canned ``requests`` responses; returns the URL log."""
    requested: list[str] = []

    def _blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise NetworkBlockedError(f"network disabled in CI smoke test: connect{args[1:]}")

    def _blocked_create_connection(address: Any, *args: Any, **kwargs: Any) -> None:
        raise NetworkBlockedError(f"network disabled in CI smoke test: {address}")

    def _mock_send(self: requests.Session, request: requests.PreparedRequest, **kwargs: Any) -> Response:
        url = str(request.url)
        requested.append(url)
        response = _canned_response(url)
        if response is None:
            raise NetworkBlockedError(f"network disabled in CI smoke test: {request.method} {url}")
        response.request = request
        return response

    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.create_connection = _blocked_create_connection  # type: ignore[assignment]
    requests.Session.send = _mock_send  # type: ignore[method-assign]
    return requested


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    script = Path(args[0])
    if not script.is_absolute():
        script = (Path.cwd() / script) if (Path.cwd() / script).exists() else REPO_ROOT / script
    requested = install_network_guard()
    sys.argv = [str(script), *args[1:]]
    sys.path.insert(0, str(REPO_ROOT))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code:
            return code
    for url in requested:
        print(f"[ci-offline-smoke] served canned response: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
