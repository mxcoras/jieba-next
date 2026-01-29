from __future__ import annotations

import warnings

from .analyse import *  # noqa: F403

warnings.warn(
    "jieba_next.analyze is deprecated, use jieba_next.analyse instead",
    DeprecationWarning,
    stacklevel=2,
)
