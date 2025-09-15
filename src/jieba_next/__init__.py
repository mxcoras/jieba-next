from __future__ import annotations

import logging
import marshal
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import warnings
from hashlib import md5
from importlib import resources as _resources
from importlib.metadata import version as _pkg_version
from importlib.resources import files as _pkg_files
from math import log
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from . import finalseg, jieba_next_functions

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping, Sequence

if os.name == "nt":
    from shutil import move as _replace_file
else:
    _replace_file = os.rename


__license__ = "MIT"

try:
    __version__ = _pkg_version("jieba-next")
except Exception:  # fallback when package metadata unavailable (editable install)
    __version__ = "0.0.0"


DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"
DICT_WRITING = {}
_CACHE_ENV_VAR = "JIEBA_NEXT_CACHE_DIR"

_cache_dir_override: Path | None = None


default_logger = logging.getLogger(__name__)
default_logger.addHandler(logging.NullHandler())


pool = None

re_userdict = re.compile("^(.+?)( [0-9]+)?( [a-z]+)?$", re.UNICODE)
re_eng = re.compile("[a-zA-Z0-9]", re.UNICODE)

# \u4E00-\u9FD5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# \r\n|\s : whitespace characters. Will not be handled.
re_han_default = re.compile("([\u4e00-\u9fd5a-zA-Z0-9+#&\\._%]+)", re.UNICODE)
re_skip_default = re.compile("(\r\n|\\s)", re.UNICODE)
re_han_cut_all = re.compile("([\u4e00-\u9fd5]+)", re.UNICODE)
re_skip_cut_all = re.compile("[^a-zA-Z0-9+#\n]", re.UNICODE)


def _user_cache_dir(app_name: str, app_author: str | None = None) -> str:
    """Return a per-user cache directory path cross-platform without external deps.

    Rough logic:
    * Windows: %LOCALAPPDATA%/<AppName> or %APPDATA% fallback.
    * macOS: ~/Library/Caches/<AppName>
    * Linux/Unix: $XDG_CACHE_HOME/<AppName> or ~/.cache/<AppName>
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


def setLogLevel(log_level):
    """Deprecated. Use set_log_level.

    Retained for compatibility with jieba/jieba_fast.
    """
    warnings.warn(
        "setLogLevel is deprecated, use set_log_level instead",
        DeprecationWarning,
        stacklevel=2,
    )
    default_logger.setLevel(log_level)


def set_log_level(log_level):
    """Set logging level for jieba_next's default logger."""
    configure_logging(log_level)


class JiebaError(Exception):
    """Base exception for jieba_next."""


class DictionaryFormatError(JiebaError):
    """Raised when dictionary file has invalid formatting."""


class DictionaryNotFoundError(JiebaError):
    """Raised when specified dictionary path does not exist."""


def set_cache_dir(path: str | Path) -> None:
    """Override the cache directory used for prefix dict caches."""
    global _cache_dir_override
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    _cache_dir_override = p
    default_logger.debug("Cache directory overridden: %s", p)


def get_cache_dir() -> Path:
    if _cache_dir_override is not None:
        return _cache_dir_override
    env_dir = os.environ.get(_CACHE_ENV_VAR)
    if env_dir:
        p = Path(env_dir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    if _user_cache_dir is not None:
        base = Path(_user_cache_dir("jieba-next", "jieba-next"))
        base.mkdir(parents=True, exist_ok=True)
        return base
    # fallback
    fallback = Path(tempfile.gettempdir()) / "jieba-next-cache"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _cache_file_name(abs_path: str | Path | None) -> str:
    if not abs_path or abs_path == DEFAULT_DICT:
        return "jieba-next.cache"
    abs_str = str(abs_path)
    return f"jieba-next.u{md5(abs_str.encode('utf-8', 'replace')).hexdigest()}.cache"


def _resolve_cache_file(abs_path: str | Path | None) -> Path:
    return get_cache_dir() / _cache_file_name(abs_path)


def open_dict_resource(dictionary: str | Path | None) -> TextIO:
    """Open dictionary path or built-in resource and return a text stream."""
    if dictionary in (None, DEFAULT_DICT):
        dict_path = _pkg_files(__package__).joinpath(DEFAULT_DICT_NAME)
        return Path(dict_path).open(encoding="utf-8")
    p = Path(dictionary).expanduser()
    if not p.is_file():
        raise DictionaryNotFoundError(f"dictionary file does not exist: {p}")
    return p.open(encoding="utf-8")


class Tokenizer:
    def __init__(self, dictionary: str | Path | None = DEFAULT_DICT):
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            self.dictionary = Path(dictionary).resolve()
        self.FREQ = {}
        self.total = 0
        self.user_word_tag_tab = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None

    def __repr__(self):
        return f"<Tokenizer dictionary={self.dictionary!r}>"

    def gen_pfdict(self, f: TextIO) -> tuple[dict[str, int], int]:
        lfreq: dict[str, int] = {}
        ltotal: int = 0
        f_name = getattr(f, "name", "stream")
        for lineno, line in enumerate(f, 1):
            line_parts = line.strip().split(" ")
            if len(line_parts) < 2 or not line_parts[1].isdigit():
                raise DictionaryFormatError(
                    f"invalid dictionary entry in {f_name} at Line {lineno}: {line}"
                )
            word, freq = line_parts[:2]
            freq = int(freq)
            lfreq[word] = freq
            ltotal += freq
            for ch in range(len(word)):
                wfrag = word[: ch + 1]
                if wfrag not in lfreq:
                    lfreq[wfrag] = 0
        f.close()
        return lfreq, ltotal

    def initialize(self, dictionary: str | Path | None = None) -> None:
        if dictionary:
            abs_path = Path(dictionary).resolve()
            if self.dictionary == abs_path and self.initialized:
                return
            else:
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
            cache_file = (
                Path(self.cache_file)
                if self.cache_file
                else _resolve_cache_file(abs_path)
            )
            tmpdir = cache_file.parent

            load_from_cache_fail = True
            if cache_file.is_file() and (
                abs_path == DEFAULT_DICT
                or cache_file.stat().st_mtime > Path(abs_path).stat().st_mtime
            ):
                default_logger.debug("Loading model cache: %s", cache_file)
                try:
                    with cache_file.open("rb") as cf:
                        self.FREQ, self.total = marshal.load(cf)
                    load_from_cache_fail = False
                except Exception:
                    default_logger.warning("Failed to load cache, rebuilding.")
                    load_from_cache_fail = True

            if load_from_cache_fail:
                wlock = DICT_WRITING.get(abs_path, threading.RLock())
                DICT_WRITING[abs_path] = wlock
                with wlock:
                    self.FREQ, self.total = self.gen_pfdict(self.get_dict_file())
                    default_logger.debug("Writing model cache: %s", cache_file)
                    try:
                        # prevent moving across different filesystems
                        fd, fpath = tempfile.mkstemp(dir=tmpdir)
                        with os.fdopen(fd, "wb") as temp_cache_file:
                            marshal.dump((self.FREQ, self.total), temp_cache_file)
                        _replace_file(fpath, cache_file)
                    except Exception:
                        default_logger.exception("Failed persisting cache file")

                try:
                    del DICT_WRITING[abs_path]
                except KeyError:
                    pass

            self.initialized = True
            default_logger.info(
                "Loaded prefix dict in %.3fs (entries=%d)",
                time.time() - t1,
                len(self.FREQ),
            )
            default_logger.debug("Prefix dict built successfully")

    def check_initialized(self) -> None:
        if not self.initialized:
            self.initialize()

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
            tmplist = []
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

    def __cut_all(self, sentence: str) -> Iterator[str]:
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

    def __cut_DAG_NO_HMM(self, sentence: str) -> Iterator[str]:
        self.check_initialized()
        route = []
        jieba_next_functions._get_DAG_and_calc(
            self.FREQ, sentence, route, float(self.total)
        )
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
            buf = ""

    def __cut_DAG(self, sentence: str) -> Iterator[str]:
        self.check_initialized()
        route = []
        jieba_next_functions._get_DAG_and_calc(
            self.FREQ, sentence, route, float(self.total)
        )
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
                            recognized = finalseg.cut(buf)
                            for t in recognized:
                                yield t
                        else:
                            for elem in buf:
                                yield elem
                        buf = ""
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                recognized = finalseg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def cut(
        self, sentence: str, cut_all: bool = False, HMM: bool = True
    ) -> Iterator[str]:
        """
        The main function that segments an entire sentence that contains
        Chinese characters into seperated words.

        Parameter:
            - sentence: The str to be segmented.
            - cut_all: Model type. True for full pattern, False for accurate pattern.
            - HMM: Whether to use the Hidden Markov Model.
        """
        if cut_all:
            re_han = re_han_cut_all
            re_skip = re_skip_cut_all
        else:
            re_han = re_han_default
            re_skip = re_skip_default
        if cut_all:
            cut_block = self.__cut_all
        elif HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM
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
        """
        Finer segmentation for search engines.
        """
        words = self.cut(sentence, HMM=HMM)
        for w in words:
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

    _lcut = lcut
    _lcut_for_search = lcut_for_search

    def _lcut_no_hmm(self, sentence: str) -> list[str]:
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence: str) -> list[str]:
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence: str) -> list[str]:
        return self.lcut_for_search(sentence, False)

    def get_dict_file(self) -> TextIO:
        return open_dict_resource(self.dictionary)

    def load_userdict(self, f: str | Path | TextIO) -> None:
        """
        Load personalized dict to improve detect rate.

        Parameter:
            - f : A plain text file contains words and their ocurrences.
                  Can be a file-like object, or the path of the dictionary file,
                  whose encoding must be utf-8.

        Structure of dict file:
        word1 freq1 word_type1
        word2 freq2 word_type2
        ...
        Word type may be ignored
        """
        self.check_initialized()
        if isinstance(f, str):
            f = Path(f).open(encoding="utf-8")

        for ln in f:
            line = ln.strip()
            if not line:
                continue
            # match won't be None because there's at least one character
            word, freq, tag = re_userdict.match(line).groups()
            if freq is not None:
                freq = freq.strip()
            if tag is not None:
                tag = tag.strip()
            self.add_word(word, freq, tag)

    def add_word(
        self, word: str, freq: int | None = None, tag: str | None = None
    ) -> None:
        """
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
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

    def del_word(self, word: str) -> None:
        """
        Convenient function for deleting a word.
        """
        self.add_word(word, 0)

    def suggest_freq(self, segment: str | Sequence[str], tune: bool = False) -> int:
        """
        Suggest word frequency to force the characters in a word to be
        joined or splitted.

        Parameter:
            - segment : The segments that the word is expected to be cut into,
                        If the word should be treated as a whole, use a str.
            - tune : If True, tune the word frequency.

        Note that HMM may affect the final result. If the result doesn't change,
        set HMM=False.
        """
        self.check_initialized()
        ftotal = float(self.total)
        freq = 1
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
            add_word(word, freq)
        return freq

    def tokenize(
        self, unicode_sentence: str, mode: str = "default", HMM: bool = True
    ) -> Iterator[tuple[str, int, int]]:
        """
        Tokenize a sentence and yields tuples of (word, start, end)

        Parameter:
            - sentence: the str to be segmented.
            - mode: "default" or "search", "search" is for finer segmentation.
            - HMM: whether to use the Hidden Markov Model.
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
            if not Path(abs_path).is_file():
                raise DictionaryNotFoundError(
                    f"jieba_next: file does not exist: {abs_path}"
                )
            self.dictionary = abs_path
            self.initialized = False


# default Tokenizer instance

dt = Tokenizer()


# global functions
def get_freq(key: str, default: int | None = None) -> int | None:
    """Get word frequency from the in-memory dictionary.

    Preferred new name. Returns `default` when key not found.
    """
    return dt.FREQ.get(key, default)


def get_FREQ(k: str, d: int | None = None) -> int | None:
    """Deprecated alias of get_freq."""
    warnings.warn(
        "get_FREQ is deprecated, use get_freq instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return dt.FREQ.get(k, d)


add_word = dt.add_word
calc = dt.calc
cut = dt.cut
cut_for_search = dt.cut_for_search
del_word = dt.del_word
get_DAG = dt.get_DAG
get_dict_file = dt.get_dict_file
initialize = dt.initialize
lcut = dt.lcut
lcut_for_search = dt.lcut_for_search
load_userdict = dt.load_userdict
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


def _replace_file(src, dest):
    # rename can't be used across different file systems
    shutil.copy(src, dest)
    Path(src).unlink()


# Explicit public API
__all__ = [
    # Core types/instances
    "Tokenizer",
    "dt",
    # Logging
    "set_log_level",
    "setLogLevel",
    "configure_logging",
    "enable_default_logging",
    # Core operations
    "add_word",
    "calc",
    "cut",
    "cut_for_search",
    "del_word",
    "get_DAG",
    "get_dict_file",
    "get_freq",
    "get_FREQ",
    "initialize",
    "lcut",
    "lcut_for_search",
    "load_userdict",
    "set_dictionary",
    "suggest_freq",
    "tokenize",
    "user_word_tag_tab",
    # Cache helpers
    "set_cache_dir",
    "get_cache_dir",
    "open_dict_resource",
    # Exceptions
    "JiebaError",
    "DictionaryFormatError",
    "DictionaryNotFoundError",
    # Metadata
    "__version__",
    "__license__",
]
