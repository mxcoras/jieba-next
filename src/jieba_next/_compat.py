"""Cross-platform compatibility helpers."""

from __future__ import annotations

import errno
import os
import shutil
import sys
import tempfile
from pathlib import Path


def replace_file(src: str | Path, dest: str | Path) -> None:
    """Replace ``dest`` with ``src``, handling cross-device moves safely."""
    src_path = Path(src)
    dest_path = Path(dest)
    try:
        src_path.replace(dest_path)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            shutil.copy2(src_path, dest_path)
            src_path.unlink()
        else:
            raise


def user_cache_dir(app_name: str = "jieba-next") -> str:
    """Return a per-user cache directory path cross-platform without external deps.

    Rough logic:
    * Windows: ``%LOCALAPPDATA%/<AppName>`` or ``%APPDATA%`` fallback.
    * macOS: ``~/Library/Caches/<AppName>``
    * Linux/Unix: ``$XDG_CACHE_HOME/<AppName>`` or ``~/.cache/<AppName>``
    """
    name = app_name or "jieba-next"
    if os.name == "nt":  # Windows
        base = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or tempfile.gettempdir()
        )
        return str(Path(base) / name)
    if sys.platform == "darwin":  # macOS
        return str(Path.home() / "Library" / "Caches" / name)
    xdg = os.environ.get("XDG_CACHE_HOME")  # Linux / other Unix
    if xdg:
        return str(Path(xdg) / name)
    return str(Path.home() / ".cache" / name)


# Backwards-compatible single-underscore aliases (used by older internal callers)
_replace_file = replace_file
_user_cache_dir = user_cache_dir


__all__ = [
    "_replace_file",
    "_user_cache_dir",
    "replace_file",
    "user_cache_dir",
]
