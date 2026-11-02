from __future__ import annotations

from typing import Any, Dict


def make_evidence_card(
    *,
    evidence_id: str,
    source_type: str,
    title: str,
    text: str,
    score: float,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "source_type": source_type,
        "title": (title or "").strip(),
        "text": (text or "").strip(),
        "score": float(score),
        "metadata": metadata,
    }
