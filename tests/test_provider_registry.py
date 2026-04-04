from __future__ import annotations

import pytest

import config
from src.database import db_client
from src.ingestion.provider_registry import get_provider_limit, get_provider_spec


def test_get_provider_spec_returns_alpha_vantage_metadata() -> None:
    provider = get_provider_spec("av")
    assert provider.name == "av"
    assert provider.daily_limit == config.AV_DAILY_LIMIT
    assert provider.enforce_limit is True


def test_get_provider_spec_raises_for_unknown_provider() -> None:
    with pytest.raises(KeyError):
        get_provider_spec("unknown-provider")


def test_db_request_logging_accepts_non_limited_provider(tmp_path) -> None:
    conn = db_client.get_connection(str(tmp_path / "provider.db"))
    db_client.initialize_schema(conn)

    db_client.log_api_request(conn, "edgar", endpoint="companyfacts")

    count = db_client.get_api_request_count(conn, "edgar")
    conn.close()

    assert count == 1
    assert get_provider_limit("edgar") is None
