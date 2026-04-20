from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, TextIO

import jieba_next

from .char_state_tab import P as char_state_tab_P
from .prob_emit import P as emit_P
from .prob_start import P as start_P
from .prob_trans import P as trans_P
from .viterbi import viterbi

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


re_han_detail = re.compile(r"([\u4E00-\u9FD5]+)")
re_skip_detail = re.compile(r"([\.0-9]+|[a-zA-Z0-9]+)")
re_han_internal = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
re_skip_internal = re.compile(r"(\r\n|\s)")

re_eng = re.compile(r"[a-zA-Z0-9]+")
re_num = re.compile(r"[\.0-9]+")

re_eng1 = re.compile(r"^[a-zA-Z0-9]$", re.UNICODE)


# Attributes explicitly delegated to the underlying Tokenizer via __getattr__.
# Narrowing this from "forward everything" makes the surface testable and
# predictable for static analysis / mocking.
_FORWARDED_TOKENIZER_ATTRS = frozenset({
    "FREQ",
    "add_word",
    "calc",
    "check_initialized",
    "del_word",
    "dictionary",
    "get_DAG",
    "load_userdict",
    "set_dictionary",
    "suggest_freq",
    "total",
    "user_word_tag_tab",
})


class Pair:
    """词与词性二元组。保持与原始 jieba 兼容的可迭代/比较行为。"""

    word: str
    flag: str

    def __init__(self, word: str, flag: str):
        self.word = word
        self.flag = flag

    def __unicode__(self) -> str:
        return f"{self.word}/{self.flag}"

    def __repr__(self) -> str:
        return f"pair({self.word!r}, {self.flag!r})"

    def __str__(self) -> str:
        return self.__unicode__()

    def __iter__(self):  # -> Iterator[str]
        return iter((self.word, self.flag))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Pair):
            return NotImplemented
        return self.word < other.word

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Pair)
            and self.word == other.word
            and self.flag == other.flag
        )

    def __hash__(self) -> int:
        return hash(self.word)

    def encode(self, arg: str) -> bytes:
        return self.__unicode__().encode(arg)


class POSTokenizer:
    tokenizer: jieba_next.Tokenizer
    word_tag_tab: dict[str, str]

    def __init__(self, tokenizer: jieba_next.Tokenizer | None = None):
        self.tokenizer = tokenizer or jieba_next.Tokenizer()
        with self.tokenizer.get_dict_file() as stream:
            self.load_word_tag(stream)

    def __repr__(self) -> str:
        return f"<POSTokenizer tokenizer={self.tokenizer!r}>"

    def __getattr__(self, name: str):
        # POS tokenization only supports the "detail" flow; explicitly block
        # these Tokenizer methods rather than silently producing ``str`` output
        # instead of ``Pair`` (matches historical behaviour).
        if name in ("cut_for_search", "lcut_for_search", "tokenize"):
            raise NotImplementedError(
                f"POSTokenizer does not support {name!r}; "
                "use jieba_next.tokenize/cut_for_search via the base Tokenizer."
            )
        # Forward only an explicit allowlist to the underlying Tokenizer. This
        # keeps the public surface discoverable while preserving compatibility
        # with ``posseg.dt.add_word(...)`` style calls used by existing users.
        if name in _FORWARDED_TOKENIZER_ATTRS:
            return getattr(self.tokenizer, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def initialize(self, dictionary: str | Path | None = None) -> None:
        self.tokenizer.initialize(dictionary)
        with self.tokenizer.get_dict_file() as stream:
            self.load_word_tag(stream)

    def load_word_tag(self, f: TextIO) -> None:
        """Populate ``word_tag_tab`` from a POS dictionary stream.

        Unlike the historical implementation this method no longer closes the
        stream; callers own the file's lifetime.
        """
        self.word_tag_tab = {}
        f_name = getattr(f, "name", "stream")
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip()
                if not line:
                    continue
                word, _, tag = line.split(" ")
                self.word_tag_tab[word] = tag
            except Exception as e:
                raise ValueError(
                    f"invalid POS dictionary entry in {f_name} at Line {lineno}: {line}"
                ) from e

    def makesure_userdict_loaded(self) -> None:
        if self.tokenizer.user_word_tag_tab:
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    # ------------------------------------------------------------------
    # Private cut helpers (single-underscore so tests can import them)
    # ------------------------------------------------------------------

    def _cut(self, sentence: str) -> Iterator[Pair]:
        prob, pos_list = viterbi(sentence, char_state_tab_P, start_P, trans_P, emit_P)
        begin, nexti = 0, 0

        for i, char in enumerate(sentence):
            pos = pos_list[i][0]
            if pos == "B":
                begin = i
            elif pos == "E":
                yield Pair(sentence[begin : i + 1], pos_list[i][1])
                nexti = i + 1
            elif pos == "S":
                yield Pair(char, pos_list[i][1])
                nexti = i + 1
        if nexti < len(sentence):
            yield Pair(sentence[nexti:], pos_list[nexti][1])

    def _cut_detail(self, sentence: str) -> Iterator[Pair]:
        blocks = re_han_detail.split(sentence)
        for blk in blocks:
            if re_han_detail.match(blk):
                yield from self._cut(blk)
            else:
                tmp = re_skip_detail.split(blk)
                for x in tmp:
                    if x:
                        if re_num.match(x):
                            yield Pair(x, "m")
                        elif re_eng.match(x):
                            yield Pair(x, "eng")
                        else:
                            yield Pair(x, "x")

    def _cut_dag_no_hmm(self, sentence: str) -> Iterator[Pair]:
        DAG = self.tokenizer.get_DAG(sentence)
        route: dict[int, tuple[float, int]] = {}
        self.tokenizer.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ""
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng1.match(l_word):
                buf += l_word
                x = y
            else:
                if buf:
                    yield Pair(buf, "eng")
                    buf = ""
                yield Pair(l_word, self.word_tag_tab.get(l_word, "x"))
                x = y
        if buf:
            yield Pair(buf, "eng")

    def _cut_dag(self, sentence: str) -> Iterator[Pair]:
        DAG = self.tokenizer.get_DAG(sentence)
        route: dict[int, tuple[float, int]] = {}

        self.tokenizer.calc(sentence, DAG, route)

        x = 0
        buf = ""
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield Pair(buf, self.word_tag_tab.get(buf, "x"))
                    elif not self.tokenizer.FREQ.get(buf):
                        yield from self._cut_detail(buf)
                    else:
                        for elem in buf:
                            yield Pair(elem, self.word_tag_tab.get(elem, "x"))
                    buf = ""
                yield Pair(l_word, self.word_tag_tab.get(l_word, "x"))
            x = y

        if buf:
            if len(buf) == 1:
                yield Pair(buf, self.word_tag_tab.get(buf, "x"))
            elif not self.tokenizer.FREQ.get(buf):
                yield from self._cut_detail(buf)
            else:
                for elem in buf:
                    yield Pair(elem, self.word_tag_tab.get(elem, "x"))

    def _cut_internal(self, sentence: str, HMM: bool = True) -> Iterator[Pair]:
        self.makesure_userdict_loaded()
        blocks = re_han_internal.split(sentence)
        cut_blk = self._cut_dag if HMM else self._cut_dag_no_hmm

        for blk in blocks:
            if re_han_internal.match(blk):
                yield from cut_blk(blk)
            else:
                tmp = re_skip_internal.split(blk)
                for x in tmp:
                    if re_skip_internal.match(x):
                        yield Pair(x, "x")
                    else:
                        for xx in x:
                            word_tag = self.word_tag_tab.get(xx)
                            if word_tag:
                                yield Pair(xx, word_tag)
                            else:
                                yield Pair(xx, "x")

    def _lcut_internal(self, sentence: str) -> list[Pair]:
        return list(self._cut_internal(sentence))

    def _lcut_internal_no_hmm(self, sentence: str) -> list[Pair]:
        return list(self._cut_internal(sentence, False))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cut(self, sentence: str, HMM: bool = True) -> Iterator[Pair]:
        self.makesure_userdict_loaded()

        if HMM:
            cuter = self._cut
            blocks = re_han_detail.split(sentence)
        else:
            cuter = self.tokenizer.cut
            blocks = jieba_next.re_han_default.split(sentence)

        for blk in blocks:
            if (HMM and re_han_detail.match(blk)) or (
                not HMM and jieba_next.re_han_default.match(blk)
            ):
                for word in cuter(blk):
                    if isinstance(word, Pair):
                        yield word
                    else:
                        for xx in word:
                            word_tag = self.word_tag_tab.get(xx)
                            if word_tag:
                                yield Pair(xx, word_tag)
                            else:
                                yield Pair(xx, "x")
            else:
                tmp = re_skip_detail.split(blk)
                for x in tmp:
                    if x:
                        if re_num.match(x):
                            yield Pair(x, "m")
                        elif re_eng.match(x):
                            yield Pair(x, "eng")
                        else:
                            yield Pair(x, "x")

    def lcut(self, *args, **kwargs) -> list[Pair]:
        return list(self.cut(*args, **kwargs))


# default Tokenizer instance


dt: POSTokenizer = POSTokenizer(jieba_next.dt)

# global functions
cut = dt.cut
lcut = dt.lcut
initialize = dt.initialize


def _lcut(*args, **kwargs):
    warnings.warn(
        "posseg._lcut is deprecated, use lcut instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return dt._lcut_internal(*args, **kwargs)


def _lcut_no_hmm(*args, **kwargs):
    warnings.warn(
        "posseg._lcut_no_hmm is deprecated, use lcut(HMM=False) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return dt._lcut_internal_no_hmm(*args, **kwargs)


# Explicit public API for posseg
__all__ = [
    "POSTokenizer",
    "Pair",
    "cut",
    "dt",
    "initialize",
    "lcut",
]
