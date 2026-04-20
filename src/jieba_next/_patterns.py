"""Regex patterns used by the top-level tokenizer."""

from __future__ import annotations

import re

re_userdict = re.compile("^(.+?)( [0-9]+)?( [a-z]+)?$", re.UNICODE)
re_eng = re.compile("[a-zA-Z0-9]", re.UNICODE)

# \u4E00-\u9FD5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# \r\n|\s : whitespace characters. Will not be handled.
re_han_default = re.compile("([\u4e00-\u9fd5a-zA-Z0-9+#&\\._%]+)", re.UNICODE)
re_skip_default = re.compile("(\r\n|\\s)", re.UNICODE)
re_han_cut_all = re.compile("([\u4e00-\u9fd5]+)", re.UNICODE)
re_skip_cut_all = re.compile("[^a-zA-Z0-9+#\n]", re.UNICODE)


__all__ = [
    "re_eng",
    "re_han_cut_all",
    "re_han_default",
    "re_skip_cut_all",
    "re_skip_default",
    "re_userdict",
]
