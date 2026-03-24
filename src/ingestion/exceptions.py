"""Shared exceptions for the ingestion layer."""

from __future__ import annotations


class FMPEndpointDeprecatedError(RuntimeError):
    """Raised when FMP returns a 403 Legacy Endpoint response.

    FMP deprecated all v3 REST endpoints on 2025-08-31 for accounts that were
    not grandfathered into legacy access.  Callers should catch this to surface
    a clear, actionable error rather than a generic HTTP 403.

    Resolution options:
      1. Upgrade to an FMP paid plan (Starter or higher) that includes the
         ``/stable/income-statement`` and ``/stable/key-metrics`` endpoints.
      2. Replace FMP with an alternative free fundamental data source
         (e.g. SEC EDGAR XBRL API, SimFin, or yfinance for limited data).
    """


class AVRateLimitError(RuntimeError):
    """Raised when Alpha Vantage returns a server-side rate-limit response.

    Distinct from ``RuntimeError`` (local DB budget exhaustion) so callers can
    handle AV throttling gracefully — return partial results and stop the batch
    cleanly — rather than treating it as a hard failure.

    Triggered by the ``"Note"`` or ``"Information"`` keys in the AV JSON
    response, both of which indicate the free-tier daily call limit has been
    reached on AV's servers.
    """
