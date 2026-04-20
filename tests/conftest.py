"""Shared pytest fixtures for jieba-next tests."""

from __future__ import annotations

import pytest

import jieba_next


@pytest.fixture
def isolated_tokenizer(tmp_path):
    """Return a freshly initialised :class:`Tokenizer` with a hermetic cache.

    The tokenizer writes its marshal cache under ``tmp_path`` so tests do not
    pollute (or depend on) the user-level cache directory.
    """
    tok = jieba_next.Tokenizer(cache_dir=tmp_path)
    tok.initialize()
    return tok
