from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from Retrieval._base_retriever import get_runtime_config, load_context_rows, retrieve_dense
from Retrieval.config_utils import ensure_required_path, resolve_repo_path
from Retrieval.evidence_schema import make_evidence_card
from Retrieval.province_extractor import extract_province_from_query


def _get_paths(config_path: str | None = None) -> Tuple[Dict, Path]:
    cfg = get_runtime_config(config_path)
    index_dir = resolve_repo_path(cfg["paths"]["index_dir"])
    return cfg, index_dir


def _load_context_card(cfg: Dict) -> Dict:
    context_path = ensure_required_path(cfg["paths"]["sentencing_general_context"], label="sentencing_general_context")
    context_version = str(cfg.get("retrieval", {}).get("context_version", "2017"))
    rows = load_context_rows(str(context_path))
    row = next((item for item in rows if str(item.get("version_group")) == context_version), None)
    if row is None:
        raise ValueError(f"Missing sentencing_general_context for version_group={context_version}")
    metadata = dict(row)
    return make_evidence_card(
        evidence_id=f"context:{row['context_id']}",
        source_type="sentencing_general_context",
        title=f"量刑一般规则 | {context_version}",
        text=row.get("text", ""),
        score=1.0,
        metadata=metadata,
    )


def _is_default_national(row: Dict, default_version: str) -> bool:
    if row.get("is_default_for_cail2018") is True:
        return True
    return str(row.get("version_group")) == default_version and row.get("legal_snapshot") != "modern_latest"


def _unique_crime_terms(rows: Iterable[Dict]) -> List[str]:
    terms: Set[str] = set()
    for row in rows:
        for crime in row.get("crime_tags", []) or []:
            if isinstance(crime, str) and crime:
                terms.add(crime)
    return sorted(terms, key=lambda item: (-len(item), item))

# crime_terms是全国指导意见中涉及到的全部罪名list，match_query_crimes函数是判断query中是否包含crime_terms中的罪名
def _match_query_crimes(query: str, crime_terms: Iterable[str]) -> List[str]:
    matched: List[str] = []
    for term in crime_terms:
        short = term[:-1] if term.endswith("罪") else term
        if term in query or (short and short in query):
            matched.append(term)
    deduped: List[str] = []
    seen: Set[str] = set()
    for item in matched:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped

# 情况 1：query 没识别出罪名，只返回 2017 默认指导意见。
# 情况 2：query 里的罪名 2017 都覆盖，只返回 2017 默认指导意见。
# 情况 3：query 里有 2017 没覆盖的罪名，允许返回 modern_latest 中匹配该缺失罪名的 section。

def _select_national_hits(hits: List[Tuple[float, Dict]], query: str, topk: int, default_version: str) -> Tuple[List[Tuple[float, Dict, bool]], List[str], List[str]]:
    rows = [row for _, row in hits]
    # query中涉及到的罪名
    query_crimes = _match_query_crimes(query, _unique_crime_terms(rows))
    default_crimes: Set[str] = set()
    # 2017 默认指导意见中涉及到的罪名
    for row in rows:
        if _is_default_national(row, default_version):
            default_crimes.update(row.get("crime_tags", []) or [])

    missing_crimes = [crime for crime in query_crimes if crime not in default_crimes]
    missing_set = set(missing_crimes)
    covered_set = set(query_crimes) - missing_set

    selected: List[Tuple[float, Dict, bool]] = []
    for score, row in hits:
        row_crimes = set(row.get("crime_tags", []) or [])
        is_default = _is_default_national(row, default_version)

        if not query_crimes:
            if is_default:
                selected.append((score, row, False))
            continue

        if not missing_set:
            if is_default:
                selected.append((score, row, False))
            continue

        if is_default and (row.get("section_type") == "common_factor" or row_crimes & covered_set):
            selected.append((score, row, False))
        elif (not is_default) and row_crimes & missing_set:
            selected.append((score, row, True))

    if not selected:
        selected = [(score, row, False) for score, row in hits if _is_default_national(row, default_version)]
    return selected[:topk], query_crimes, missing_crimes


def _make_national_card(score: float, row: Dict, *, used_as_fallback: bool) -> Dict:
    metadata = dict(row)
    metadata["used_as_fallback"] = used_as_fallback
    return make_evidence_card(
        evidence_id=f"guideline_national:{row['section_id']}",
        source_type="guideline_national",
        title=f"全国指导意见 | {row.get('section_heading', '')}",
        text=row.get("section_text", ""),
        score=score,
        metadata=metadata,
    )


def guideline_search(query: str, topk: int = 4) -> dict:
    if not query or not str(query).strip():
        return {
            "tool_name": "guideline_search",
            "query": query,
            "province_extracted": None,
            "sentencing_general_context": None,
            "national_guideline_evidence": [],
            "local_guideline_evidence": [],
            "error": "missing query",
        }

    cfg, index_dir = _get_paths()
    query = str(query).strip()
    province = extract_province_from_query(query)

    national_faiss = ensure_required_path(str(index_dir / "guidelines_national.faiss"), label="guidelines_national_index")
    national_meta = ensure_required_path(str(index_dir / "guidelines_national_metadata.jsonl"), label="guidelines_national_metadata")

    default_version = str(cfg.get("retrieval", {}).get("guideline_version_group", "2017"))
    # National sections are small; fetch all dense-ranked candidates, then apply
    # 2017-first fallback filtering for crimes absent from the 2017 guideline.
    national_hits = retrieve_dense(national_faiss, national_meta, query, topk, search_topn=10000)
    selected_national_hits, query_crimes, missing_crimes = _select_national_hits(
        national_hits,
        query,
        topk,
        default_version,
    )
    national_cards = [
        _make_national_card(score, row, used_as_fallback=used_as_fallback)
        for score, row, used_as_fallback in selected_national_hits
    ]

    local_cards: List[Dict] = []
    if province is not None:
        local_faiss = index_dir / "guidelines_local" / f"{province}.faiss"
        local_meta = index_dir / "guidelines_local" / f"{province}_metadata.jsonl"
        if local_faiss.exists() and local_meta.exists():
            local_hits = retrieve_dense(local_faiss, local_meta, query, topk)
            local_cards = [
                make_evidence_card(
                    evidence_id=f"guideline_local:{row['section_id']}",
                    source_type="guideline_local",
                    title=f"{province}实施细则 | {row.get('section_heading', '')}",
                    text=row.get("section_text", ""),
                    score=score,
                    metadata=dict(row),
                )
                for score, row in local_hits[:topk]
            ]

    return {
        "tool_name": "guideline_search",
        "query": query,
        "province_extracted": province,
        "sentencing_general_context": _load_context_card(cfg),
        "national_guideline_evidence": national_cards,
        "local_guideline_evidence": local_cards,
        "query_crime_tags": query_crimes,
        "missing_in_default_guideline": missing_crimes,
        "national_fallback_used": any(card["metadata"].get("used_as_fallback") for card in national_cards),
        "note": "地方细则是否被检索取决于 query 是否包含省份。",
    }
# national_fallback_used是指和CAIL2018对应年份的指导意见中没有找到query中罪名时，是否使用了最新指导意见中的section。
