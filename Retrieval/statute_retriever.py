from __future__ import annotations

from pathlib import Path
from typing import Tuple

from Retrieval._base_retriever import get_runtime_config, retrieve_dense
from Retrieval.config_utils import ensure_required_path, resolve_repo_path
from Retrieval.evidence_schema import make_evidence_card


def _article_label(row: dict) -> str:
    return f"{row.get('article_no', '')}条{row.get('article_suffix', '') or ''}"


def _get_index_paths(config_path: str | None = None) -> Tuple[Path, Path]:
    cfg = get_runtime_config(config_path)
    index_dir = resolve_repo_path(cfg["paths"]["index_dir"])
    return index_dir / "statutes.faiss", index_dir / "statutes_metadata.jsonl"


def statute_search(query: str, topk: int = 4) -> dict:
    if not query or not str(query).strip():
        return {"tool_name": "statute_search", "query": query, "results": [], "error": "missing query"}
    faiss_path, metadata_path = _get_index_paths()[:2]
    ensure_required_path(str(faiss_path), label="statute_index")
    ensure_required_path(str(metadata_path), label="statute_metadata")
    query = str(query).strip()
    raw_results = retrieve_dense(faiss_path, metadata_path, query, topk)

    results = []
    for score, row in raw_results[:topk]:
        article_label = _article_label(row)
        title = f"中华人民共和国刑法 第{article_label} {row.get('article_title', '')}".strip()
        text = f"第{article_label} {row.get('text', '')}".strip()
        results.append(
            make_evidence_card(
                evidence_id=f"statute:{row['article_version_id']}",
                source_type="statute",
                title=title,
                text=text,
                score=score,
                metadata=dict(row),
            )
        )
    return {"tool_name": "statute_search", "query": query, "results": results}
    # results = [{},{},……]
