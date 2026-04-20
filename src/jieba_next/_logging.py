"""Logging helpers for jieba_next."""

from __future__ import annotations

import logging
import sys
import warnings

default_logger = logging.getLogger("jieba_next")
default_logger.addHandler(logging.NullHandler())


def configure_logging(level: int | str = logging.INFO, *, stream=None) -> None:
    """Configure a basic stream handler for jieba_next (idempotent)."""
    if not any(isinstance(h, logging.StreamHandler) for h in default_logger.handlers):
        handler = logging.StreamHandler(stream or sys.stderr)
        formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        default_logger.addHandler(handler)
    default_logger.setLevel(level)


def enable_default_logging() -> None:
    """Enable INFO level logging with a basic handler if none configured."""
    configure_logging(logging.INFO)


def set_log_level(log_level) -> None:
    """Set logging level for jieba_next's default logger."""
    configure_logging(log_level)


def setLogLevel(log_level) -> None:
    """Deprecated. Use :func:`set_log_level`.

    Retained for compatibility with jieba/jieba_fast.
    """
    warnings.warn(
        "setLogLevel is deprecated, use set_log_level instead",
        DeprecationWarning,
        stacklevel=2,
    )
    default_logger.setLevel(log_level)


__all__ = [
    "configure_logging",
    "default_logger",
    "enable_default_logging",
    "setLogLevel",
    "set_log_level",
]
