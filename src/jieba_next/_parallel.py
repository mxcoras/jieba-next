"""Parallel segmentation mode (multiprocessing based).

Encapsulates the legacy ``enable_parallel``/``disable_parallel`` API. The
module keeps a single :class:`multiprocessing.pool.Pool` instance keyed by the
module-level state, mirroring the original jieba behaviour.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.pool import Pool

    from .tokenizer import Tokenizer

# Exposed for tests / code that used to read ``jieba_next.pool`` directly.
pool: Pool | None = None


def _lcut(sentence: str) -> list[str]:
    from . import dt

    return dt.lcut(sentence)


def _lcut_all(sentence: str) -> list[str]:
    from . import dt

    return dt.lcut(sentence, True)


def _lcut_no_hmm(sentence: str) -> list[str]:
    from . import dt

    return dt.lcut(sentence, False, False)


def _lcut_for_search(sentence: str) -> list[str]:
    from . import dt

    return dt.lcut_for_search(sentence)


def _lcut_for_search_no_hmm(sentence: str) -> list[str]:
    from . import dt

    return dt.lcut_for_search(sentence, False)


def _pcut(sentence: str, cut_all: bool = False, HMM: bool = True) -> Iterator[str]:
    parts = sentence.splitlines(True)
    if cut_all:
        result = pool.map(_lcut_all, parts)
    elif HMM:
        result = pool.map(_lcut, parts)
    else:
        result = pool.map(_lcut_no_hmm, parts)
    for r in result:
        yield from r


def _pcut_for_search(sentence: str, HMM: bool = True) -> Iterator[str]:
    parts = sentence.splitlines(True)
    if HMM:
        result = pool.map(_lcut_for_search, parts)
    else:
        result = pool.map(_lcut_for_search_no_hmm, parts)
    for r in result:
        yield from r


def enable_parallel(processnum: int | None = None) -> None:
    """Switch module-level ``cut``/``cut_for_search`` to a parallel version.

    Only works against the module-level default tokenizer ``jieba_next.dt``;
    custom :class:`Tokenizer` instances are not supported (same limitation as
    the original jieba).
    """
    global pool
    from multiprocessing import Pool, cpu_count

    import jieba_next

    if os.name == "nt":
        raise NotImplementedError("jieba: parallel mode only supports posix system")

    jieba_next.dt.check_initialized()
    if processnum is None:
        processnum = cpu_count()
    if pool is not None:
        pool.close()
        pool.join()
    pool = Pool(processnum)
    jieba_next.cut = _pcut
    jieba_next.cut_for_search = _pcut_for_search


def disable_parallel() -> None:
    global pool
    import jieba_next

    if pool is not None:
        pool.close()
        pool.join()
        pool = None
    jieba_next.cut = jieba_next.dt.cut
    jieba_next.cut_for_search = jieba_next.dt.cut_for_search


__all__ = [
    "disable_parallel",
    "enable_parallel",
    "pool",
]
