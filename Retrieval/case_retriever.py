from __future__ import annotations

from pathlib import Path
from typing import Tuple

from Retrieval._base_retriever import get_runtime_config, retrieve_dense
from Retrieval.config_utils import ensure_required_path, resolve_repo_path
from Retrieval.evidence_schema import make_evidence_card


def _get_index_paths(config_path: str | None = None) -> Tuple[Path, Path]:
    cfg = get_runtime_config(config_path)
    index_dir = resolve_repo_path(cfg["paths"]["index_dir"])
    return index_dir / "cases.faiss", index_dir / "cases_metadata.jsonl"


def case_search(query: str, topk: int = 4) -> dict:
    if not query or not str(query).strip():
        return {"tool_name": "case_search", "query": query, "results": [], "error": "missing query"}
    faiss_path, metadata_path = _get_index_paths()[:2]
    ensure_required_path(str(faiss_path), label="case_index")
    ensure_required_path(str(metadata_path), label="case_metadata")

    query = str(query).strip()
    raw_results = retrieve_dense(faiss_path, metadata_path, query, topk)

    results = []
    for score, row in raw_results[:topk]:
        accusation = row.get("accusation", [])
        articles = row.get("relevant_articles", [])
        title = f"类案 | {'/'.join(accusation) or '未知罪名'} | 第{','.join(str(x) for x in articles) or '?'}条 | {row.get('imprisonment', '?')}个月"
        results.append(
            make_evidence_card(
                evidence_id=f"case:{row['case_id']}",
                source_type="precedent_case",
                title=title,
                text=row.get("fact", ""),
                score=score,
                metadata=dict(row),
            )
        )
    return {"tool_name": "case_search", "query": query, "results": results}
