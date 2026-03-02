from __future__ import annotations

import os
from typing import Optional

from langfuse import Langfuse

_LANGFUSE: Optional[Langfuse] = None


def get_langfuse() -> Optional[Langfuse]:
    """Return a singleton Langfuse client or None if ENV is not configured.

    ENV:
      - LANGFUSE_SECRET_KEY
      - LANGFUSE_PUBLIC_KEY
      - LANGFUSE_HOST (optional, default: https://cloud.langfuse.com)
    """
    global _LANGFUSE
    if _LANGFUSE is not None:
        return _LANGFUSE

    secret = os.getenv("LANGFUSE_SECRET_KEY")
    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not secret or not public:
        return None

    _LANGFUSE = Langfuse(
        secret_key=secret,
        public_key=public,
        host=host,
    )
    return _LANGFUSE