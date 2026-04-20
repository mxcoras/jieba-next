"""Cache directory management for prefix-dict marshal caches."""

from __future__ import annotations

import os
import tempfile
from hashlib import md5
from pathlib import Path

from ._compat import user_cache_dir
from ._logging import default_logger

DEFAULT_DICT = None
_CACHE_ENV_VAR = "JIEBA_NEXT_CACHE_DIR"

_cache_dir_override: Path | None = None


def set_cache_dir(path: str | Path) -> None:
    """Override the cache directory used for prefix dict caches."""
    global _cache_dir_override
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    _cache_dir_override = p
    default_logger.debug("Cache directory overridden: %s", p)


def get_cache_dir() -> Path:
    """Return the directory used to store marshal caches, creating it if needed."""
    if _cache_dir_override is not None:
        return _cache_dir_override
    env_dir = os.environ.get(_CACHE_ENV_VAR)
    if env_dir:
        p = Path(env_dir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    base = Path(user_cache_dir("jieba-next"))
    try:
        base.mkdir(parents=True, exist_ok=True)
        return base
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "jieba-next-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def cache_file_name(abs_path: str | Path | None) -> str:
    """Return the deterministic cache file name for a given dictionary path."""
    if not abs_path or abs_path == DEFAULT_DICT:
        return "jieba-next.cache"
    abs_str = str(abs_path)
    return f"jieba-next.u{md5(abs_str.encode('utf-8', 'replace')).hexdigest()}.cache"


def resolve_cache_file(
    abs_path: str | Path | None, *, cache_dir: str | Path | None = None
) -> Path:
    """Resolve the full cache file path for a given dictionary path.

    ``cache_dir`` can be used to override the module-level default (useful in
    tests that need hermetic state).
    """
    base = Path(cache_dir) if cache_dir is not None else get_cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / cache_file_name(abs_path)


# Backwards-compatible private aliases
_cache_file_name = cache_file_name
_resolve_cache_file = resolve_cache_file


__all__ = [
    "DEFAULT_DICT",
    "_CACHE_ENV_VAR",
    "_cache_file_name",
    "_resolve_cache_file",
    "cache_file_name",
    "get_cache_dir",
    "resolve_cache_file",
    "set_cache_dir",
]
