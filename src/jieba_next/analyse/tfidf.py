from operator import itemgetter
from pathlib import Path

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

    def set_stop_words(self, stop_words_path):
        abs_path = Path(stop_words_path).resolve()
        if not Path(abs_path).is_file():
            raise Exception("jieba_next: file does not exist: " + abs_path)
        with abs_path.open(encoding="utf-8") as f:
            for line in f:
                self.stop_words.add(line.strip())

    def extract_tags(self, *args, **kwargs):
        raise NotImplementedError


class IDFLoader:
    def __init__(self, idf_path=None):
        self.path = ""
        self.idf_freq = {}
        self.median_idf = 0.0
        if idf_path:
            self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path):
        if self.path != new_idf_path:
            self.path = new_idf_path
            with Path(new_idf_path).open(encoding="utf-8") as f:
                self.idf_freq = {}
                for line in f:
                    word, freq = line.strip().split(" ")
                    self.idf_freq[word] = float(freq)
            self.median_idf = sorted(self.idf_freq.values())[len(self.idf_freq) // 2]

    def get_idf(self):
        return self.idf_freq, self.median_idf


class TFIDF(KeywordExtractor):
    def __init__(self, idf_path=None):
        self.tokenizer = jieba_next.dt
        self.postokenizer = jieba_next.posseg.dt
        self.stop_words = self.STOP_WORDS.copy()
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def set_idf_path(self, idf_path):
        new_abs_path = Path(idf_path).resolve()
        if not Path(new_abs_path).is_file():
            raise Exception("jieba_next: file does not exist: " + new_abs_path)
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def extract_tags(
        self, sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False
    ):
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
            allowPOS = frozenset(allowPOS)
            words = self.postokenizer.cut(sentence)
        else:
            words = self.tokenizer.cut(sentence)
        freq = {}
        for w in words:
            if allowPOS:
                if w.flag not in allowPOS:
                    continue
                elif not withFlag:
                    w = w.word
            wc = w.word if allowPOS and withFlag else w
            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                continue
            freq[w] = freq.get(w, 0.0) + 1.0
        total = sum(freq.values())
        for k in freq:
            kw = k.word if allowPOS and withFlag else k
            freq[k] *= self.idf_freq.get(kw, self.median_idf) / total

        if withWeight:
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)
        if topK:
            return tags[:topK]
        else:
            return tags
