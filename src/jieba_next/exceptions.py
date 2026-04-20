"""Exception types for jieba_next."""

from __future__ import annotations


class JiebaError(Exception):
    """Base exception for jieba_next."""


class DictionaryFormatError(JiebaError):
    """Raised when dictionary file has invalid formatting."""


class DictionaryNotFoundError(JiebaError):
    """Raised when specified dictionary path does not exist."""


__all__ = [
    "DictionaryFormatError",
    "DictionaryNotFoundError",
    "JiebaError",
]
