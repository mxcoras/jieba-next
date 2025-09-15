from __future__ import annotations

import re
from typing import TYPE_CHECKING

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
        self.load_word_tag(self.tokenizer.get_dict_file())

    def __repr__(self) -> str:
        return f"<POSTokenizer tokenizer={self.tokenizer!r}>"

    def __getattr__(self, name: str):
        if name in ("cut_for_search", "lcut_for_search", "tokenize"):
            # may be possible?
            raise NotImplementedError
        return getattr(self.tokenizer, name)

    def initialize(self, dictionary: str | Path | None = None) -> None:
        self.tokenizer.initialize(dictionary)
        self.load_word_tag(self.tokenizer.get_dict_file())

    def load_word_tag(self, f) -> None:  # TextIO but keep loose for compatibility
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
        f.close()

    def makesure_userdict_loaded(self) -> None:
        if self.tokenizer.user_word_tag_tab:
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    def __cut(self, sentence: str) -> Iterator[Pair]:
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

    def __cut_detail(self, sentence: str) -> Iterator[Pair]:
        blocks = re_han_detail.split(sentence)
        for blk in blocks:
            if re_han_detail.match(blk):
                yield from self.__cut(blk)
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

    def __cut_DAG_NO_HMM(self, sentence: str) -> Iterator[Pair]:
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}
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
            buf = ""

    def __cut_DAG(self, sentence: str) -> Iterator[Pair]:
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}

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
                        recognized = self.__cut_detail(buf)
                        for t in recognized:
                            yield t
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
                recognized = self.__cut_detail(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield Pair(elem, self.word_tag_tab.get(elem, "x"))

    def __cut_internal(self, sentence: str, HMM: bool = True) -> Iterator[Pair]:
        self.makesure_userdict_loaded()
        blocks = re_han_internal.split(sentence)
        if HMM:
            cut_blk = self.__cut_DAG
        else:
            cut_blk = self.__cut_DAG_NO_HMM

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
        return list(self.__cut_internal(sentence))

    def _lcut_internal_no_hmm(self, sentence: str) -> list[Pair]:
        return list(self.__cut_internal(sentence, False))

    def cut(self, sentence: str, HMM: bool = True):  # -> Iterator[Pair]
        self.makesure_userdict_loaded()

        if HMM:
            cuter = self.__cut
        else:
            cuter = self.tokenizer.cut

        if HMM:
            blocks = re_han_detail.split(sentence)
        else:
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
_lcut = dt._lcut
_lcut_no_hmm = dt._lcut_no_hmm
initialize = dt.initialize


# Explicit public API for posseg
__all__ = [
    "Pair",
    "POSTokenizer",
    "dt",
    "cut",
    "lcut",
    "initialize",
]
