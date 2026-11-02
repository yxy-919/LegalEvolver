from __future__ import annotations

from typing import Optional

from Retrieval.common.province_utils import PROVINCE_ALIASES


def extract_province_from_query(query: str) -> Optional[str]:
    if not query:
        return None
    for alias in sorted(PROVINCE_ALIASES.keys(), key=len, reverse=True):
        if alias in query:
            return PROVINCE_ALIASES[alias]
    return None
