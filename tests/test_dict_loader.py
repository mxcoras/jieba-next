"""Unit tests for the prefix-dict parser extracted from :class:`Tokenizer`."""

from __future__ import annotations

import io

import pytest

from jieba_next._dict import parse_prefix_dict
from jieba_next.exceptions import DictionaryFormatError


def test_parse_prefix_dict_basic() -> None:
    freq, total = parse_prefix_dict(["hello 5 n\n", "world 3\n"])
    assert total == 8
    assert freq["hello"] == 5
    assert freq["world"] == 3
    # prefix entries should be present with 0 frequency where not a word
    assert freq["h"] == 0
    assert freq["hel"] == 0
    # full word entries preserved
    assert freq["world"] == 3


def test_parse_prefix_dict_ignores_trailing_columns() -> None:
    freq, total = parse_prefix_dict(["tok 7 noun extra-column\n"])
    assert freq["tok"] == 7
    assert total == 7


def test_parse_prefix_dict_rejects_missing_freq() -> None:
    with pytest.raises(DictionaryFormatError):
        parse_prefix_dict(["word-only\n"], source_name="inline")


def test_parse_prefix_dict_rejects_non_numeric_freq() -> None:
    with pytest.raises(DictionaryFormatError):
        parse_prefix_dict(["word notanumber\n"], source_name="inline")


def test_parse_prefix_dict_accepts_stream_like() -> None:
    stream = io.StringIO("abc 2\ndef 1\n")
    freq, total = parse_prefix_dict(stream)
    assert total == 3
    assert freq["abc"] == 2
    assert freq["def"] == 1
