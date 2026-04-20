"""Unit tests for individual :class:`Tokenizer` helpers."""

from __future__ import annotations

import jieba_next


def test_isolated_tokenizer_uses_injected_cache_dir(isolated_tokenizer, tmp_path):
    # At least one cache file should have been written under tmp_path during
    # initialisation; the exact name depends on dict path but must start with
    # our prefix.
    entries = list(tmp_path.iterdir())
    assert entries, "expected at least one cache file"
    assert any(e.name.startswith("jieba-next") for e in entries)


def test_get_dag_single_char(isolated_tokenizer) -> None:
    dag = isolated_tokenizer.get_DAG("中")
    assert 0 in dag
    assert dag[0] == [0]


def test_cut_all_yields_overlapping_tokens(isolated_tokenizer) -> None:
    # _cut_all is the newly exposed private method (was __cut_all).
    tokens = list(isolated_tokenizer._cut_all("南京市长江大桥"))
    assert "南京" in tokens
    assert "长江" in tokens
    assert "长江大桥" in tokens


def test_cut_dag_and_no_hmm_preserve_input(isolated_tokenizer) -> None:
    sentence = "小明硕士毕业于中国科学院计算所"
    hmm = "".join(isolated_tokenizer._cut_dag(sentence))
    no_hmm = "".join(isolated_tokenizer._cut_dag_no_hmm(sentence))
    assert hmm == sentence
    assert no_hmm == sentence


def test_suggest_freq_returns_int(isolated_tokenizer) -> None:
    freq = isolated_tokenizer.suggest_freq("云计算", tune=False)
    assert isinstance(freq, int)
    assert freq > 0


def test_add_word_invalidates_rust_prefix(isolated_tokenizer) -> None:
    isolated_tokenizer._ensure_rust_prefix()
    assert isolated_tokenizer._rust_prefix is not None
    isolated_tokenizer.add_word("虚构词abc")
    assert isolated_tokenizer._rust_prefix is None


def test_default_dt_is_tokenizer_instance() -> None:
    assert isinstance(jieba_next.dt, jieba_next.Tokenizer)
