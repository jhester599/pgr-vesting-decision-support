"""Shared exceptions for the ingestion layer."""

from __future__ import annotations


class AVRateLimitError(RuntimeError):
    """Raised when Alpha Vantage returns a server-side rate-limit response.

    Distinct from ``RuntimeError`` (local DB budget exhaustion) so callers can
    handle AV throttling gracefully — return partial results and stop the batch
    cleanly — rather than treating it as a hard failure.

    Triggered by the ``"Note"`` or ``"Information"`` keys in the AV JSON
    response, both of which indicate the free-tier daily call limit has been
    reached on AV's servers.
    """
