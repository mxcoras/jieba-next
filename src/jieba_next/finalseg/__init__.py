import re

from ..jieba_next_rust import _viterbi as viterbi_hmm
from .prob_emit import P as emit_P
from .prob_start import P as start_P
from .prob_trans import P as trans_P

PrevStatus = {"B": "ES", "M": "MB", "S": "SE", "E": "BM"}

Force_Split_Words = set()


def __cut(sentence):
    global emit_P
    prob, pos_list = viterbi_hmm(sentence, "BMES", start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == "B":
            begin = i
        elif pos == "E":
            yield sentence[begin : i + 1]
            nexti = i + 1
        elif pos == "S":
            yield char
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]


re_han = re.compile("([\u4e00-\u9fd5]+)")
re_skip = re.compile(r"([a-zA-Z0-9]+(?:\.\d+)?%?)")


def add_force_split(word):
    global Force_Split_Words
    Force_Split_Words.add(word)


def cut(sentence):
    blocks = re_han.split(sentence)
    for blk in blocks:
        if re_han.match(blk):
            for word in __cut(blk):
                if word not in Force_Split_Words:
                    yield word
                else:
                    yield from word
        else:
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    yield x
