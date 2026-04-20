from __future__ import annotations

from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jieba_next
import jieba_next.posseg
from jieba_next.exceptions import DictionaryNotFoundError
from jieba_next.posseg import Pair

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

DEFAULT_IDF = Path(__file__).parent / "idf.txt"


class KeywordExtractor:
    STOP_WORDS = {
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
    }

    def set_stop_words(self, stop_words_path: str | Path) -> None:
        abs_path = Path(stop_words_path).resolve()
        if not abs_path.is_file():
            raise DictionaryNotFoundError(
                f"jieba_next: file does not exist: {abs_path}"
            )
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
        new_path_str = str(new_idf_path)
        if self.path != new_path_str:
            self.path = new_path_str
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
        if not new_abs_path.is_file():
            raise DictionaryNotFoundError(
                f"jieba_next: file does not exist: {new_abs_path}"
            )
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    # ------------------------------------------------------------------
    # Internal helpers (testable in isolation)
    # ------------------------------------------------------------------

    def _iter_candidate_words(
        self,
        sentence: str,
        allow_pos: frozenset[str] | None,
    ) -> Iterator[str | Pair]:
        """Yield ``Pair`` (if ``allow_pos`` is set) or ``str`` tokens."""
        if allow_pos is not None:
            for pair in self.postokenizer.cut(sentence):
                if pair.flag in allow_pos:
                    yield pair
        else:
            yield from self.tokenizer.cut(sentence)

    def _accumulate_freq(
        self,
        words: Iterable[str | Pair],
        *,
        use_pair_key: bool,
    ) -> dict[Any, float]:
        """Count occurrences with either ``Pair`` or ``str`` as dict key.

        ``use_pair_key`` is True only when caller asked for both allowPOS and
        withFlag; in that case the dict key is a ``Pair`` so the original
        POS flag survives to the final output.
        """
        freq: dict[Any, float] = {}
        for w in words:
            if isinstance(w, Pair):
                text = w.word
                key: Any = w if use_pair_key else w.word
            else:
                text = w
                key = w
            if len(text.strip()) < 2 or text.lower() in self.stop_words:
                continue
            freq[key] = freq.get(key, 0.0) + 1.0
        return freq

    def _apply_idf(self, freq: dict[Any, float]) -> None:
        total = sum(freq.values())
        scale = 1.0 / (total or 1.0)
        for k in freq:
            text = k.word if isinstance(k, Pair) else k
            freq[k] *= self.idf_freq.get(text, self.median_idf) * scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_tags(
        self,
        sentence: str,
        topK: int | None = 20,
        withWeight: bool = False,
        allowPOS: Sequence[str] | tuple[str, ...] = (),
        withFlag: bool = False,
    ) -> list | list[tuple[str, float]]:
        """Extract keywords from sentence using TF-IDF.

        Parameters:
            topK: return how many top keywords. ``None`` for all possible words.
            withWeight: if True, return a list of ``(word, weight)``; if False,
                return a list of words.
            allowPOS: the allowed POS list (e.g. ``['ns', 'n', 'vn', 'v','nr']``).
                If a word's POS is not in this list, it is filtered out.
            withFlag: only meaningful when ``allowPOS`` is non-empty. If True,
                return a list of ``Pair(word, weight)`` (like :func:`posseg.cut`);
                otherwise return a list of ``str``.
        """
        allow_pos = frozenset(allowPOS) if allowPOS else None
        use_pair_key = allow_pos is not None and withFlag

        words = self._iter_candidate_words(sentence, allow_pos)
        freq = self._accumulate_freq(words, use_pair_key=use_pair_key)
        self._apply_idf(freq)

        if withWeight:
            tags: list = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)
        if topK:
            return tags[:topK]
        return tags
