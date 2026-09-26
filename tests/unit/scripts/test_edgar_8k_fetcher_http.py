"""Focused tests for the script-level EDGAR 8-K HTTP helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from scripts import edgar_8k_fetcher


def test_get_uses_shared_edgar_headers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _mock_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        response = MagicMock()
        response.raise_for_status = MagicMock()
        return response

    monkeypatch.setattr(edgar_8k_fetcher.time, "sleep", lambda *_args, **_kwargs: None)

    with patch(
        "scripts.edgar_8k_fetcher.config.build_edgar_headers",
        return_value={"User-Agent": "Unit Test"},
    ) as mock_headers:
        with patch("scripts.edgar_8k_fetcher.requests.get", side_effect=_mock_get):
            response = edgar_8k_fetcher._get(
                "https://data.sec.gov/submissions/test.json"
            )

    assert response is not None
    mock_headers.assert_called_once_with()
    assert captured["url"] == "https://data.sec.gov/submissions/test.json"
    assert captured["headers"] == {"User-Agent": "Unit Test"}
    assert captured["timeout"] == 30


def test_get_retries_transient_http_errors(monkeypatch) -> None:
    attempts = {"count": 0}

    transient = requests.HTTPError("server error")
    transient.response = MagicMock(status_code=503)

    success_response = MagicMock()
    success_response.raise_for_status = MagicMock()

    def _mock_get(url, headers=None, timeout=None):
        del url, headers, timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            failing_response = MagicMock()
            failing_response.status_code = 503
            failing_response.raise_for_status.side_effect = transient
            return failing_response
        return success_response

    monkeypatch.setattr(edgar_8k_fetcher.time, "sleep", lambda *_args, **_kwargs: None)

    with patch("scripts.edgar_8k_fetcher.requests.get", side_effect=_mock_get):
        response = edgar_8k_fetcher._get(
            "https://data.sec.gov/submissions/test.json",
            retries=2,
        )

    assert attempts["count"] == 2
    assert response is success_response
