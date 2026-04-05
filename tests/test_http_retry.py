from __future__ import annotations

from unittest.mock import MagicMock

import config
from src.ingestion.http_utils import build_retry_session


def test_build_retry_session_mounts_retry_adapter() -> None:
    session = build_retry_session(total_retries=4, backoff_factor=2.0)
    https_adapter = session.adapters["https://"]
    retry = https_adapter.max_retries

    assert retry.total == 4
    assert retry.connect == 4
    assert retry.read == 4
    assert retry.status == 4
    assert retry.backoff_factor == 2.0
    assert 429 in retry.status_forcelist
    assert 503 in retry.status_forcelist


def test_av_client_uses_retry_session(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "AV_API_KEY", "test-key")
    monkeypatch.setattr(config, "DATA_RAW_DIR", str(tmp_path))
    monkeypatch.setattr(config, "REQUEST_COUNTS_FILE", str(tmp_path / "counts.json"))

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"Global Quote": {"05. price": "100.0"}}

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    monkeypatch.setattr("src.ingestion.av_client.build_retry_session", lambda: mock_session)

    from src.ingestion.av_client import get

    result = get("GLOBAL_QUOTE", {"symbol": "PGR"}, cache_hours=0)

    assert result["Global Quote"]["05. price"] == "100.0"
    assert mock_session.get.call_count == 1


def test_fred_loader_uses_retry_session(monkeypatch) -> None:
    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "observations": [{"date": "2024-01-31", "value": "0.35"}]
    }

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    monkeypatch.setattr("src.ingestion.fred_loader.build_retry_session", lambda: mock_session)

    from src.ingestion.fred_loader import fetch_fred_series

    df = fetch_fred_series("T10Y2Y")

    assert list(df.columns) == ["T10Y2Y"]
    assert mock_session.get.call_count == 1


def test_edgar_client_uses_retry_session(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.ingestion.edgar_client._cache_path", lambda: str(tmp_path / "companyfacts.json"))
    monkeypatch.setattr("src.ingestion.edgar_client._is_cache_valid", lambda *args, **kwargs: False)
    monkeypatch.setenv("EDGAR_USER_AGENT", "Retry Test retry@example.com")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"facts": {}}

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    monkeypatch.setattr("src.ingestion.edgar_client.build_retry_session", lambda: mock_session)

    from src.ingestion.edgar_client import fetch_companyfacts

    result = fetch_companyfacts(force_refresh=True)

    assert result == {"facts": {}}
    assert mock_session.get.call_count == 1
