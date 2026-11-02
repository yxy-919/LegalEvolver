from __future__ import annotations

from pathlib import Path
from typing import Tuple

from Retrieval._base_retriever import get_runtime_config, retrieve_dense
from Retrieval.config_utils import ensure_required_path, resolve_repo_path
from Retrieval.evidence_schema import make_evidence_card


def _get_index_paths(config_path: str | None = None) -> Tuple[Path, Path]:
    cfg = get_runtime_config(config_path)
    index_dir = resolve_repo_path(cfg["paths"]["index_dir"])
    return index_dir / "judicial_interpretations.faiss", index_dir / "judicial_interpretations_metadata.jsonl"


def judicial_interpretation_search(query: str, topk: int = 4) -> dict:
    if not query or not str(query).strip():
        return {"tool_name": "judicial_interpretation_search", "query": query, "results": [], "error": "missing query"}

    faiss_path, metadata_path = _get_index_paths()
    ensure_required_path(str(faiss_path), label="judicial_interpretation_index")
    ensure_required_path(str(metadata_path), label="judicial_interpretation_metadata")

    query = str(query).strip()
    raw_results = retrieve_dense(faiss_path, metadata_path, query, topk)

    results = []
    for score, row in raw_results[:topk]:
        title = str(row.get("title") or row.get("doc_title") or "").strip()
        text = "\n".join(part for part in [title, str(row.get("text", "")).strip()] if part)
        results.append(
            make_evidence_card(
                evidence_id=f"judicial_interpretation:{row['id']}",
                source_type="judicial_interpretation",
                title=title,
                text=text,
                score=score,
                metadata=dict(row),
            )
        )
    return {"tool_name": "judicial_interpretation_search", "query": query, "results": results}
