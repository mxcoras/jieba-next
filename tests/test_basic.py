from __future__ import annotations

import os
import warnings

import pytest

import jieba_next as jieba


def test_basic_cut_roundtrip() -> None:
    sentence = "南京市长江大桥"
    tokens = list(jieba.cut(sentence))
    assert tokens
    assert "".join(tokens) == sentence


def test_cut_for_search_has_more_or_equal_tokens() -> None:
    sentence = "南京市长江大桥"
    base = jieba.lcut(sentence)
    search = jieba.lcut_for_search(sentence)
    assert len(search) >= len(base)
    assert all(isinstance(x, str) for x in search)


def test_tokenize_offsets_match_text() -> None:
    sentence = "小明硕士毕业于中国科学院计算所"
    for word, start, end in jieba.tokenize(sentence):
        assert sentence[start:end] == word


def test_add_word_affects_tokenizer() -> None:
    tokenizer = jieba.Tokenizer()
    tokenizer.add_word("云原生")
    assert "云原生" in tokenizer.lcut("云原生")


def test_deprecated_aliases_warn() -> None:
    with pytest.warns(
        DeprecationWarning, match="get_FREQ is deprecated, use get_freq instead"
    ):
        _ = jieba.get_FREQ("南京")

    with pytest.warns(
        DeprecationWarning, match="setLogLevel is deprecated, use set_log_level instead"
    ):
        jieba.setLogLevel(60)

    with warnings.catch_warnings():
        warnings.simplefilter("always", DeprecationWarning)
        with pytest.warns(
            DeprecationWarning,
            match="jieba_next.analyze is deprecated, use jieba_next.analyse instead",
        ):
            import jieba_next.analyze as _  # noqa: F401


def test_chinese_analyzer_smoke() -> None:
    from jieba_next.analyse import ChineseAnalyzer

    analyzer = ChineseAnalyzer()
    tokens = list(analyzer("南京市长江大桥"))
    assert tokens


def test_parallel_cut_toggle() -> None:
    if os.name == "nt":
        pytest.skip("parallel mode only supports posix system")
    sentence = "南京市长江大桥\n小明硕士毕业于中国科学院计算所"
    try:
        jieba.enable_parallel(2)
        tokens = list(jieba.cut(sentence))
        assert tokens
        assert "".join(tokens).replace("\n", "") in sentence.replace("\n", "")
    finally:
        jieba.disable_parallel()
