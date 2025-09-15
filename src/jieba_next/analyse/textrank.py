from __future__ import annotations

from collections import defaultdict
from operator import itemgetter

import jieba_next.posseg

from .tfidf import KeywordExtractor


class UndirectWeightedGraph:
    d: float = 0.85

    def __init__(self):
        self.graph: dict[object, list[tuple[object, object, int]]] = defaultdict(list)

    def addEdge(self, start: object, end: object, weight: int) -> None:
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self) -> dict[object, float]:
        ws: dict[object, float] = defaultdict(float)
        outSum: dict[object, float] = defaultdict(float)

        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)

        # this line for build stable iteration
        sorted_keys = sorted(self.graph.keys())
        for _ in range(10):  # 10 iters
            for n in sorted_keys:
                s = 0.0
                for e in self.graph[n]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s

        min_rank, max_rank = float("inf"), float("-inf")

        for w in ws.values():
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in ws.items():
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws


class TextRank(KeywordExtractor):
    tokenizer: jieba_next.posseg.POSTokenizer
    postokenizer: jieba_next.posseg.POSTokenizer
    stop_words: set[str]
    pos_filt: frozenset[str]
    span: int

    def __init__(self):
        self.tokenizer = self.postokenizer = jieba_next.posseg.dt
        self.stop_words = self.STOP_WORDS.copy()
        self.pos_filt = frozenset(("ns", "n", "vn", "v"))
        self.span = 5

    def pairfilter(self, wp) -> bool:  # Pair
        return (
            wp.flag in self.pos_filt
            and len(wp.word.strip()) >= 2
            and wp.word.lower() not in self.stop_words
        )

    def textrank(
        self,
        sentence: str,
        topK: int | None = 20,
        withWeight: bool = False,
        allowPOS: tuple[str, ...] = ("ns", "n", "vn", "v"),
        withFlag: bool = False,
    ) -> list:
        """
        Extract keywords from sentence using TextRank algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v'].
                        if the POS of w is not in this list, it will be filtered.
            - withFlag: if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        self.pos_filt = frozenset(allowPOS)
        g = UndirectWeightedGraph()
        cm: dict[tuple[object, object], int] = defaultdict(int)
        words = tuple(self.tokenizer.cut(sentence))
        for i, wp in enumerate(words):
            if self.pairfilter(wp):
                for j in range(i + 1, i + self.span):
                    if j >= len(words):
                        break
                    if not self.pairfilter(words[j]):
                        continue
                    if allowPOS and withFlag:
                        cm[(wp, words[j])] += 1
                    else:
                        cm[(wp.word, words[j].word)] += 1

        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)
        nodes_rank = g.rank()
        if withWeight:
            tags: list = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

        if topK:
            return tags[:topK]
        else:
            return tags

    extract_tags = textrank
