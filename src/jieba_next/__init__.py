"""jieba_next public package interface.

This module is intentionally a thin aggregation layer: all real logic lives in
private sub-modules (``tokenizer``, ``_cache``, ``_dict``, ...) to make the
code easier to unit-test. The public API below matches the historical
``jieba``/``jieba_fast``/``jieba_next`` surface exactly.
"""

from __future__ import annotations

import warnings
from importlib.metadata import version as _pkg_version

from . import finalseg, jieba_next_rust
from ._cache import (
    _CACHE_ENV_VAR,
    DEFAULT_DICT,
    _cache_file_name,
    _resolve_cache_file,
    cache_file_name,
    get_cache_dir,
    resolve_cache_file,
    set_cache_dir,
)
from ._compat import _replace_file, _user_cache_dir, replace_file, user_cache_dir
from ._dict import DEFAULT_DICT_NAME, open_dict_resource, parse_prefix_dict
from ._logging import (
    configure_logging,
    default_logger,
    enable_default_logging,
    set_log_level,
    setLogLevel,
)
from ._parallel import disable_parallel, enable_parallel
from ._patterns import (
    re_eng,
    re_han_cut_all,
    re_han_default,
    re_skip_cut_all,
    re_skip_default,
    re_userdict,
)
from .exceptions import (
    DictionaryFormatError,
    DictionaryNotFoundError,
    JiebaError,
)
from .tokenizer import DICT_WRITING, Tokenizer

__license__ = "MIT"

try:
    __version__ = _pkg_version("jieba-next")
except Exception:  # fallback when package metadata unavailable (editable install)
    __version__ = "0.0.0"


# Default ``Tokenizer`` instance backing the module-level convenience helpers.
dt = Tokenizer()


# ----------------------------------------------------------------------
# Module-level convenience helpers
# ----------------------------------------------------------------------


def get_freq(key: str, default: int | None = None) -> int | None:
    """Get word frequency from the in-memory dictionary.

    Preferred new name. Returns ``default`` when ``key`` is absent.
    """
    return dt.FREQ.get(key, default)


def get_FREQ(k: str, d: int | None = None) -> int | None:
    """Deprecated alias of :func:`get_freq`."""
    warnings.warn(
        "get_FREQ is deprecated, use get_freq instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return dt.FREQ.get(k, d)


add_word = dt.add_word
calc = dt.calc
cut = dt.cut
cut_for_search = dt.cut_for_search
del_word = dt.del_word
get_DAG = dt.get_DAG
get_dict_file = dt.get_dict_file
initialize = dt.initialize
lcut = dt.lcut
lcut_for_search = dt.lcut_for_search
load_userdict = dt.load_userdict
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


# Explicit public API
__all__ = [
    "DictionaryFormatError",
    "DictionaryNotFoundError",
    "JiebaError",
    "Tokenizer",
    "__license__",
    "__version__",
    "add_word",
    "calc",
    "configure_logging",
    "cut",
    "cut_for_search",
    "del_word",
    "disable_parallel",
    "dt",
    "enable_default_logging",
    "enable_parallel",
    "get_DAG",
    "get_FREQ",
    "get_cache_dir",
    "get_dict_file",
    "get_freq",
    "initialize",
    "lcut",
    "lcut_for_search",
    "load_userdict",
    "open_dict_resource",
    "setLogLevel",
    "set_cache_dir",
    "set_dictionary",
    "set_log_level",
    "suggest_freq",
    "tokenize",
    "user_word_tag_tab",
]
