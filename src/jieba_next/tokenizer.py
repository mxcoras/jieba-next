"""Core ``Tokenizer`` class.

Factored out of the historical ``jieba_next.__init__`` god-module so unit tests
can import the class directly and exercise small helpers.
"""

from __future__ import annotations

import marshal
import os
import tempfile
import threading
import time
import warnings
from math import log
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from . import finalseg, jieba_next_rust
from ._cache import DEFAULT_DICT, resolve_cache_file
from ._compat import replace_file
from ._dict import open_dict_resource, parse_prefix_dict
from ._logging import default_logger
from ._patterns import (
    re_eng,
    re_han_cut_all,
    re_han_default,
    re_skip_cut_all,
    re_skip_default,
    re_userdict,
)
from .exceptions import DictionaryNotFoundError

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping, Sequence


# Shared write-lock registry keyed by resolved dictionary path. Preserved for
# backwards compatibility with code that monkey-patches or inspects it.
DICT_WRITING: dict[object, threading.RLock] = {}


class Tokenizer:
    def __init__(
        self,
        dictionary: str | Path | None = DEFAULT_DICT,
        *,
        cache_dir: str | Path | None = None,
    ):
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            self.dictionary = Path(dictionary).resolve()
        self.FREQ: dict[str, int] = {}
        self.total: int = 0
        self.user_word_tag_tab: dict[str, str] = {}
        self.initialized: bool = False
        self.tmp_dir = None
        self.cache_file: str | Path | None = None
        self._rust_prefix = None  # fast trie for DAG+DP
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    def __repr__(self) -> str:
        return f"<Tokenizer dictionary={self.dictionary!r}>"

    # ------------------------------------------------------------------
    # Dictionary loading
    # ------------------------------------------------------------------

    def gen_pfdict(self, f: TextIO) -> tuple[dict[str, int], int]:
        """Parse a prefix-dict stream into ``(freq_map, total)``.

        Kept as a method for backwards compatibility. Internally delegates to
        :func:`jieba_next._dict.parse_prefix_dict`. Unlike the historical
        implementation this method no longer closes ``f``; callers own the
        lifetime of the stream.
        """
        source_name = getattr(f, "name", "stream")
        return parse_prefix_dict(f, source_name=source_name)

    def get_dict_file(self) -> TextIO:
        return open_dict_resource(self.dictionary)

    # ------------------------------------------------------------------
    # initialize() helpers
    # ------------------------------------------------------------------

    def _resolve_cache_file(self, abs_path: str | Path | None) -> Path:
        if self.cache_file:
            return Path(self.cache_file)
        return resolve_cache_file(abs_path, cache_dir=self._cache_dir)

    def _load_from_cache(self, cache_file: Path, abs_path: str | Path | None) -> bool:
        """Try to load ``FREQ`` / ``total`` from ``cache_file``. Returns success."""
        if not cache_file.is_file():
            return False
        if abs_path != DEFAULT_DICT:
            try:
                if cache_file.stat().st_mtime <= Path(abs_path).stat().st_mtime:
                    return False
            except OSError:
                return False
        default_logger.debug("Loading model cache: %s", cache_file)
        try:
            with cache_file.open("rb") as cf:
                self.FREQ, self.total = marshal.load(cf)
        except Exception:
            default_logger.warning("Failed to load cache, rebuilding.")
            return False
        return True

    def _build_from_dict(self) -> None:
        """Rebuild ``FREQ`` / ``total`` from the configured dictionary file."""
        with self.get_dict_file() as stream:
            self.FREQ, self.total = self.gen_pfdict(stream)

    def _persist_cache(self, cache_file: Path) -> None:
        """Best-effort write of ``(FREQ, total)`` to ``cache_file`` atomically."""
        tmpdir = cache_file.parent
        try:
            fd, fpath = tempfile.mkstemp(dir=tmpdir)
            with os.fdopen(fd, "wb") as temp_cache_file:
                marshal.dump((self.FREQ, self.total), temp_cache_file)
            replace_file(fpath, cache_file)
        except Exception:
            default_logger.exception("Failed persisting cache file")

    def _ensure_rust_prefix(self) -> None:
        """Rebuild Rust ``PrefixDict`` from current FREQ/total if missing."""
        if self.initialized and self._rust_prefix is None:
            try:
                self._rust_prefix = jieba_next_rust.PrefixDict(
                    self.FREQ, float(self.total)
                )
            except Exception:
                self._rust_prefix = None

    def initialize(self, dictionary: str | Path | None = None) -> None:
        if dictionary:
            abs_path = Path(dictionary).resolve()
            if self.dictionary == abs_path and self.initialized:
                return
            self.dictionary = abs_path
            self.initialized = False
        else:
            abs_path = self.dictionary

        with self.lock:
            try:
                with DICT_WRITING[abs_path]:
                    pass
            except KeyError:
                pass
            if self.initialized:
                return

            default_logger.debug(
                "Building prefix dict from %s ...", abs_path or "the default dictionary"
            )
            t1 = time.time()
            cache_file = self._resolve_cache_file(abs_path)

            if not self._load_from_cache(cache_file, abs_path):
                wlock = DICT_WRITING.get(abs_path, threading.RLock())
                DICT_WRITING[abs_path] = wlock
                with wlock:
                    self._build_from_dict()
                    default_logger.debug("Writing model cache: %s", cache_file)
                    self._persist_cache(cache_file)

                try:
                    del DICT_WRITING[abs_path]
                except KeyError:
                    pass

            self.initialized = True
            self._ensure_rust_prefix()
            default_logger.info(
                "Loaded prefix dict in %.3fs (entries=%d)",
                time.time() - t1,
                len(self.FREQ),
            )
            default_logger.debug("Prefix dict built successfully")

    def check_initialized(self) -> None:
        if not self.initialized:
            self.initialize()

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def calc(
        self,
        sentence: str,
        DAG: dict[int, list[int]],
        route: MutableMapping[int, tuple[float, int]],
    ) -> None:
        N: int = len(sentence)
        route[N] = (0, 0)
        logtotal = log(self.total)
        for idx in range(N - 1, -1, -1):
            route[idx] = max(
                (
                    log(self.FREQ.get(sentence[idx : x + 1]) or 1)
                    - logtotal
                    + route[x + 1][0],
                    x,
                )
                for x in DAG[idx]
            )

    def get_DAG(self, sentence: str) -> dict[int, list[int]]:
        self.check_initialized()
        DAG: dict[int, list[int]] = {}
        N: int = len(sentence)
        for k in range(N):
            tmplist: list[int] = []
            i = k
            frag = sentence[k]
            while i < N and frag in self.FREQ:
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k : i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    # ------------------------------------------------------------------
    # Private cut strategies (single-underscore so tests can import them)
    # ------------------------------------------------------------------

    def _cut_all(self, sentence: str) -> Iterator[str]:
        dag = self.get_DAG(sentence)
        old_j = -1
        for k, L in dag.items():
            if len(L) == 1 and k > old_j:
                yield sentence[k : L[0] + 1]
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k : j + 1]
                        old_j = j

    def _route_from_rust(self, sentence: str) -> list[int]:
        self.check_initialized()
        if self._rust_prefix is None:
            self._ensure_rust_prefix()
        if self._rust_prefix is not None:
            return list(self._rust_prefix.get_dag_and_calc(sentence))
        route: list[int] = []
        jieba_next_rust._get_DAG_and_calc(self.FREQ, sentence, route, float(self.total))
        return route

    def _cut_dag_no_hmm(self, sentence: str) -> Iterator[str]:
        route = self._route_from_rust(sentence)
        x = 0
        N = len(sentence)
        buf = ""
        while x < N:
            y = route[x] + 1
            l_word = sentence[x:y]
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ""
                yield l_word
                x = y
        if buf:
            yield buf

    def _cut_dag(self, sentence: str) -> Iterator[str]:
        route = self._route_from_rust(sentence)
        x = 0
        buf = ""
        N = len(sentence)
        while x < N:
            y = route[x] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ""
                    else:
                        if not self.FREQ.get(buf):
                            yield from finalseg.cut(buf)
                        else:
                            yield from buf
                        buf = ""
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                yield from finalseg.cut(buf)
            else:
                yield from buf

    # ------------------------------------------------------------------
    # Public cut API
    # ------------------------------------------------------------------

    def cut(
        self, sentence: str, cut_all: bool = False, HMM: bool = True
    ) -> Iterator[str]:
        """Segment a sentence into words.

        Parameters:
            sentence: The ``str`` to be segmented.
            cut_all: Model type. ``True`` for full pattern, ``False`` for the
                default accurate pattern.
            HMM: Whether to use the Hidden Markov Model for unknown words.
        """
        if cut_all:
            re_han = re_han_cut_all
            re_skip = re_skip_cut_all
            cut_block = self._cut_all
        else:
            re_han = re_han_default
            re_skip = re_skip_default
            cut_block = self._cut_dag if HMM else self._cut_dag_no_hmm
        blocks = re_han.split(sentence)
        for blk in blocks:
            if not blk:
                continue
            if re_han.match(blk):
                yield from cut_block(blk)
            else:
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:
                        yield from x
                    else:
                        yield x

    def cut_for_search(self, sentence: str, HMM: bool = True) -> Iterator[str]:
        """Finer segmentation for search engines."""
        for w in self.cut(sentence, HMM=HMM):
            if len(w) > 2:
                for i in range(len(w) - 1):
                    gram2 = w[i : i + 2]
                    if self.FREQ.get(gram2):
                        yield gram2
            if len(w) > 3:
                for i in range(len(w) - 2):
                    gram3 = w[i : i + 3]
                    if self.FREQ.get(gram3):
                        yield gram3
            yield w

    def lcut(self, *args, **kwargs) -> list[str]:
        return list(self.cut(*args, **kwargs))

    def lcut_for_search(self, *args, **kwargs) -> list[str]:
        return list(self.cut_for_search(*args, **kwargs))

    # ------------------------------------------------------------------
    # Deprecated aliases (pre-existing public API)
    # ------------------------------------------------------------------

    def _lcut(self, *args, **kwargs) -> list[str]:
        warnings.warn(
            "_lcut is deprecated, use lcut instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.lcut(*args, **kwargs)

    def _lcut_for_search(self, *args, **kwargs) -> list[str]:
        warnings.warn(
            "_lcut_for_search is deprecated, use lcut_for_search instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.lcut_for_search(*args, **kwargs)

    def _lcut_no_hmm(self, sentence: str) -> list[str]:
        warnings.warn(
            "_lcut_no_hmm is deprecated, use lcut(HMM=False) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence: str) -> list[str]:
        warnings.warn(
            "_lcut_all is deprecated, use lcut(cut_all=True) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence: str) -> list[str]:
        warnings.warn(
            "_lcut_for_search_no_hmm is deprecated, "
            "use lcut_for_search(HMM=False) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.lcut_for_search(sentence, False)

    # ------------------------------------------------------------------
    # User dictionary management
    # ------------------------------------------------------------------

    def load_userdict(self, f: str | Path | TextIO) -> None:
        """Load personalized dict to improve detect rate.

        Parameter:
            - f : A plain text file contains words and their ocurrences.
                  Can be a file-like object, or the path of the dictionary file,
                  whose encoding must be utf-8.

        Structure of dict file::

            word1 freq1 word_type1
            word2 freq2 word_type2
            ...

        Word type may be ignored.
        """
        self.check_initialized()
        own_stream = False
        if isinstance(f, (str, Path)):
            f = Path(f).open(encoding="utf-8")
            own_stream = True

        try:
            for ln in f:
                line = ln.strip()
                if not line:
                    continue
                match = re_userdict.match(line)
                if match is None:  # pragma: no cover - defensive
                    continue
                word, freq, tag = match.groups()
                if freq is not None:
                    freq = freq.strip()
                if tag is not None:
                    tag = tag.strip()
                self.add_word(word, freq, tag)
        finally:
            if own_stream:
                f.close()

    def add_word(
        self, word: str, freq: int | None = None, tag: str | None = None
    ) -> None:
        """Add a word to the in-memory dictionary.

        ``freq`` and ``tag`` can be omitted; ``freq`` defaults to a calculated
        value that ensures the word can be cut out.
        """
        self.check_initialized()
        freq = int(freq) if freq is not None else self.suggest_freq(word, False)
        self.FREQ[word] = freq
        self.total += freq
        if tag:
            self.user_word_tag_tab[word] = tag
        for ch in range(len(word)):
            wfrag = word[: ch + 1]
            if wfrag not in self.FREQ:
                self.FREQ[wfrag] = 0
        if freq == 0:
            finalseg.add_force_split(word)
        # PrefixDict becomes stale; rebuild lazily next access
        self._rust_prefix = None

    def del_word(self, word: str) -> None:
        """Convenience function for deleting a word."""
        self.add_word(word, 0)
        self._rust_prefix = None

    def suggest_freq(self, segment: str | Sequence[str], tune: bool = False) -> int:
        """Suggest word frequency to force characters to be joined or split.

        Parameters:
            segment: The segments that the word is expected to be cut into. If
                the word should be treated as a whole, pass a ``str``.
            tune: If ``True``, tune the word frequency via :meth:`add_word`.

        Note that HMM may affect the final result. If the result does not
        change, set ``HMM=False``.
        """
        self.check_initialized()
        ftotal = float(self.total)
        freq = 1.0
        if isinstance(segment, str):
            word = segment
            for seg in self.cut(word, HMM=False):
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = max(int(freq * self.total) + 1, self.FREQ.get(word, 1))
        else:
            segment = tuple(map(str, segment))
            word = "".join(segment)
            for seg in segment:
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = min(int(freq * self.total), self.FREQ.get(word, 0))
        if tune:
            self.add_word(word, freq)
        return int(freq)

    def tokenize(
        self, unicode_sentence: str, mode: str = "default", HMM: bool = True
    ) -> Iterator[tuple[str, int, int]]:
        """Tokenize a sentence and yield ``(word, start, end)`` tuples.

        Parameters:
            unicode_sentence: The ``str`` to be segmented.
            mode: ``"default"`` or ``"search"`` (finer segmentation).
            HMM: Whether to use the Hidden Markov Model.
        """
        start = 0
        if mode == "default":
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                yield (w, start, start + width)
                start += width
        else:
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                if len(w) > 2:
                    for i in range(len(w) - 1):
                        gram2 = w[i : i + 2]
                        if self.FREQ.get(gram2):
                            yield (gram2, start + i, start + i + 2)
                if len(w) > 3:
                    for i in range(len(w) - 2):
                        gram3 = w[i : i + 3]
                        if self.FREQ.get(gram3):
                            yield (gram3, start + i, start + i + 3)
                yield (w, start, start + width)
                start += width

    def set_dictionary(self, dictionary_path: str | Path) -> None:
        with self.lock:
            abs_path = Path(dictionary_path).resolve()
            if not abs_path.is_file():
                raise DictionaryNotFoundError(
                    f"jieba_next: file does not exist: {abs_path}"
                )
            self.dictionary = abs_path
            self.initialized = False
            self._rust_prefix = None


__all__ = [
    "DICT_WRITING",
    "Tokenizer",
]
