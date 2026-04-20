"""Dictionary loading and prefix-dict parsing helpers."""

from __future__ import annotations

from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from .exceptions import DictionaryFormatError, DictionaryNotFoundError

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_DICT_NAME = "dict.txt"


def open_dict_resource(dictionary: str | Path | None) -> TextIO:
    """Open dictionary path or built-in resource and return a text stream.

    Returns a file-like object opened for reading in UTF-8 mode. The caller is
    responsible for closing the stream.
    """
    if dictionary is None:
        dict_path = _pkg_files("jieba_next").joinpath(DEFAULT_DICT_NAME)
        return Path(str(dict_path)).open(encoding="utf-8")
    p = Path(dictionary).expanduser()
    if not p.is_file():
        raise DictionaryNotFoundError(f"dictionary file does not exist: {p}")
    return p.open(encoding="utf-8")


def parse_prefix_dict(
    lines: Iterable[str], *, source_name: str = "stream"
) -> tuple[dict[str, int], int]:
    """Parse prefix-dict lines into ``(freq_map, total)``.

    This does **not** close any underlying file; the caller owns the stream.

    Each line must contain at least ``word freq`` separated by a single space.
    Extra fields (e.g. POS tag) are ignored.
    """
    lfreq: dict[str, int] = {}
    ltotal: int = 0
    for lineno, line in enumerate(lines, 1):
        parts = line.strip().split(" ")
        if len(parts) < 2 or not parts[1].isdigit():
            raise DictionaryFormatError(
                f"invalid dictionary entry in {source_name} at Line {lineno}: {line}"
            )
        word, freq_str = parts[:2]
        freq = int(freq_str)
        lfreq[word] = freq
        ltotal += freq
        for ch in range(len(word)):
            wfrag = word[: ch + 1]
            if wfrag not in lfreq:
                lfreq[wfrag] = 0
    return lfreq, ltotal


__all__ = [
    "DEFAULT_DICT_NAME",
    "open_dict_resource",
    "parse_prefix_dict",
]
