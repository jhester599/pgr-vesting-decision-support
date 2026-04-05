"""Shared HTTP helpers for provider clients."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config


def build_retry_session(
    total_retries: int = config.HTTP_RETRY_TOTAL,
    backoff_factor: float = config.HTTP_RETRY_BACKOFF_FACTOR,
) -> requests.Session:
    """Return a requests session with retry/backoff for transient failures."""
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
