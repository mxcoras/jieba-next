from __future__ import annotations

from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import jieba_next
import jieba_next.posseg

DEFAULT_IDF = Path(__file__).parent / "idf.txt"


class KeywordExtractor:
    STOP_WORDS = set(
        "the",
        "of",
        "is",
        "and",
        "to",
        "in",
        "that",
        "we",
        "for",
        "an",
        "are",
        "by",
        "be",
        "as",
        "on",
        "with",
        "can",
        "if",
        "from",
        "which",
        "you",
        "it",
        "this",
        "then",
        "at",
        "have",
        "all",
        "not",
        "one",
        "has",
        "or",
        "that",
    )

    def set_stop_words(self, stop_words_path: str | Path) -> None:
        abs_path = Path(stop_words_path).resolve()
        if not Path(abs_path).is_file():
            raise Exception("jieba_next: file does not exist: " + abs_path)
        with abs_path.open(encoding="utf-8") as f:
            for line in f:
                self.stop_words.add(line.strip())

    def extract_tags(self, *args, **kwargs):
        raise NotImplementedError


class IDFLoader:
    path: str
    idf_freq: dict[str, float]
    median_idf: float

    def __init__(self, idf_path: str | Path | None = None):
        self.path = ""
        self.idf_freq = {}
        self.median_idf = 0.0
        if idf_path:
            self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path: str | Path) -> None:
        if self.path != new_idf_path:
            self.path = new_idf_path
            with Path(new_idf_path).open(encoding="utf-8") as f:
                self.idf_freq = {}
                for line in f:
                    word, freq = line.strip().split(" ")
                    self.idf_freq[word] = float(freq)
            self.median_idf = sorted(self.idf_freq.values())[len(self.idf_freq) // 2]

    def get_idf(self) -> tuple[dict[str, float], float]:
        return self.idf_freq, self.median_idf


class TFIDF(KeywordExtractor):
    tokenizer: jieba_next.Tokenizer
    postokenizer: jieba_next.posseg.POSTokenizer
    stop_words: set[str]
    idf_loader: IDFLoader
    idf_freq: dict[str, float]
    median_idf: float

    def __init__(self, idf_path: str | Path | None = None):
        self.tokenizer = jieba_next.dt
        self.postokenizer = jieba_next.posseg.dt
        self.stop_words = self.STOP_WORDS.copy()
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def set_idf_path(self, idf_path: str | Path) -> None:
        new_abs_path = Path(idf_path).resolve()
        if not Path(new_abs_path).is_file():
            raise Exception("jieba_next: file does not exist: " + new_abs_path)
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def extract_tags(
        self,
        sentence: str,
        topK: int | None = 20,
        withWeight: bool = False,
        allowPOS: Sequence[str] | tuple[str, ...] = (),
        withFlag: bool = False,
    ) -> list | list[tuple[str, float]]:
        """
        Extract keywords from sentence using TF-IDF algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr'].
                        if the POS of w is not in this list,it will be filtered.
            - withFlag: only work with allowPOS is not empty.
                        if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        if allowPOS:
            allow_pos_frozen = frozenset(allowPOS)
            words_iter = self.postokenizer.cut(sentence)
        else:
            allow_pos_frozen = None
            words_iter = self.tokenizer.cut(sentence)
        freq: dict[object, float] = {}
        for w in words_iter:  # w may be Pair or str
            if allow_pos_frozen:
                if w.flag not in allow_pos_frozen:
                    continue
                if not withFlag:
                    w = w.word
            wc = w.word if allow_pos_frozen and withFlag else w
            if isinstance(wc, str):
                if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                    continue
            freq[w] = freq.get(w, 0.0) + 1.0
        total = sum(freq.values())
        for k in list(freq.keys()):
            if allowPOS and withFlag:
                kw = k.word
            elif allowPOS and not withFlag:  # Pair already converted to str
                kw = k
            else:
                kw = k if isinstance(k, str) else k  # may be str
            key_str = kw if isinstance(kw, str) else getattr(kw, "word", str(kw))
            freq[k] *= self.idf_freq.get(key_str, self.median_idf) / (total or 1.0)

        if withWeight:
            tags: list = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)
        if topK:
            return tags[:topK]
        else:
            return tags
