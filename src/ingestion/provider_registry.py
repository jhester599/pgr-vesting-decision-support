"""Explicit provider metadata and rate-limit semantics."""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class ProviderSpec:
    """Configuration and operating semantics for one data provider."""

    name: str
    daily_limit: int | None
    enforce_limit: bool
    cache_enabled: bool
    logs_to_db: bool
    description: str


_PROVIDERS: dict[str, ProviderSpec] = {
    "av": ProviderSpec(
        name="av",
        daily_limit=config.AV_DAILY_LIMIT,
        enforce_limit=True,
        cache_enabled=True,
        logs_to_db=True,
        description="Alpha Vantage market data provider with a hard free-tier daily limit.",
    ),
    "fmp": ProviderSpec(
        name="fmp",
        daily_limit=config.FMP_DAILY_LIMIT,
        enforce_limit=True,
        cache_enabled=True,
        logs_to_db=True,
        description="Legacy Financial Modeling Prep provider retained for backward compatibility only.",
    ),
    "fred": ProviderSpec(
        name="fred",
        daily_limit=None,
        enforce_limit=False,
        cache_enabled=False,
        logs_to_db=False,
        description="Federal Reserve Economic Data provider with no enforced repo-side daily limit.",
    ),
    "edgar": ProviderSpec(
        name="edgar",
        daily_limit=None,
        enforce_limit=False,
        cache_enabled=False,
        logs_to_db=True,
        description="SEC EDGAR public data source with polite request pacing instead of a repo daily cap.",
    ),
}


def get_provider_spec(name: str) -> ProviderSpec:
    """Return provider metadata or raise a clear error for unknown providers."""
    key = name.strip().lower()
    if key not in _PROVIDERS:
        known = ", ".join(sorted(_PROVIDERS))
        raise KeyError(f"Unknown provider '{name}'. Known providers: {known}.")
    return _PROVIDERS[key]


def get_provider_limit(name: str) -> int | None:
    """Return the configured daily limit for a provider, if any."""
    return get_provider_spec(name).daily_limit
