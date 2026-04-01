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
    """Raised when the Alpha Vantage hard daily quota (``"Note"`` key) is hit.

    Distinct from ``RuntimeError`` (local DB budget exhaustion) so callers can
    handle AV throttling gracefully — return partial results and stop the batch
    cleanly — rather than treating it as a hard failure.

    Triggered only by the ``"Note"`` key in the AV JSON response, which
    indicates the 25-calls/day free-tier hard limit has been reached.  When
    this is raised, the caller must stop the batch and defer remaining tickers
    to the next run.

    See also: :class:`AVRateLimitAdvisory` for the softer ``"Information"``
    advisory, which does not consume a quota slot.
    """


class AVRateLimitAdvisory(RuntimeError):
    """Raised when Alpha Vantage returns an ``"Information"`` advisory message.

    The ``"Information"`` key signals a soft advisory — AV nudges free-tier
    sessions that have used ~22–23 of their 25 daily calls in rapid succession.
    No usable time-series data is returned for the current ticker, but the
    daily quota is **not** exhausted.  Callers should log a warning, mark this
    specific ticker as skipped (``None``), and continue to the next ticker in
    the batch.

    This is distinct from :class:`AVRateLimitError` (``"Note"`` key), which
    signals a hard quota exhaustion and requires stopping the entire batch.

    Observed: 2026-03-27 — PGR dividend call returned ``"Information"`` with
    2/25 daily quota slots remaining; the remaining calls succeeded in the
    next scheduled run.
    """
