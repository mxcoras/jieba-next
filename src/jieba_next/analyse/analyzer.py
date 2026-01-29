from __future__ import annotations

import re

import jieba_next


def _identity_stem(word: str) -> str:
    return word


STOP_WORDS = frozenset((
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "have",
    "if",
    "in",
    "is",
    "it",
    "may",
    "not",
    "of",
    "on",
    "or",
    "tbd",
    "that",
    "the",
    "this",
    "to",
    "us",
    "we",
    "when",
    "will",
    "with",
    "yet",
    "you",
    "your",
    "的",
    "了",
    "和",
))

accepted_chars = re.compile(r"[\u4E00-\u9FD5]+")


class Token:  # noqa: B903
    __slots__ = ("text", "original", "pos", "startchar", "endchar")

    def __init__(self, text: str, start: int, end: int) -> None:
        self.text = text
        self.original = text
        self.pos = start
        self.startchar = start
        self.endchar = end


class Tokenizer:
    def __or__(self, other):
        return _AnalyzerPipeline(self, [other])


class _AnalyzerPipeline:
    def __init__(self, tokenizer: Tokenizer, filters: list):
        self._tokenizer = tokenizer
        self._filters = filters

    def __or__(self, other):
        return _AnalyzerPipeline(self._tokenizer, [*self._filters, other])

    def __call__(self, text: str, **kwargs):
        tokens = self._tokenizer(text, **kwargs)
        for flt in self._filters:
            tokens = flt(tokens)
        return tokens


class LowercaseFilter:
    def __call__(self, tokens):
        for token in tokens:
            token.text = token.text.lower()
            token.original = token.text
            yield token


class StopFilter:
    def __init__(self, stoplist=STOP_WORDS, minsize: int = 1):
        self.stoplist = frozenset(stoplist)
        self.minsize = minsize

    def __call__(self, tokens):
        for token in tokens:
            if len(token.text) < self.minsize or token.text in self.stoplist:
                continue
            yield token


class StemFilter:
    def __init__(self, stemfn=None, ignore=None, cachesize: int = 50000):
        self.stemfn = stemfn or _identity_stem
        self.ignore = ignore
        self._cache: dict[str, str] = {}
        self._cachesize = cachesize

    def __call__(self, tokens):
        cache = self._cache
        for token in tokens:
            text = token.text
            if text in cache:
                stemmed = cache[text]
            else:
                stemmed = self.stemfn(text)
                if self._cachesize and len(cache) >= self._cachesize:
                    cache.clear()
                cache[text] = stemmed
            token.text = stemmed
            token.original = stemmed
            yield token


class ChineseTokenizer(Tokenizer):
    def __call__(self, text, **kargs):
        words = jieba_next.tokenize(text, mode="search")
        for w, start_pos, stop_pos in words:
            if not accepted_chars.match(w) and len(w) <= 1:
                continue
            token = Token(w, start_pos, stop_pos)
            yield token


def ChineseAnalyzer(
    stoplist=STOP_WORDS,
    minsize=1,
    stemfn=None,
    cachesize=50000,
):
    return (
        ChineseTokenizer()
        | LowercaseFilter()
        | StopFilter(stoplist=stoplist, minsize=minsize)
        | StemFilter(stemfn=stemfn or _identity_stem, ignore=None, cachesize=cachesize)
    )
