"""Shared logging configuration for production entry points."""

from __future__ import annotations

import logging


_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once for production scripts."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=_LOG_FORMAT)
        return

    root_logger.setLevel(level)
