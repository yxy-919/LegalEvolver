from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Retrieval.common.io import read_jsonl
from Retrieval.config_utils import ensure_required_path, load_react_config, resolve_repo_path
from Retrieval.faiss_utils import load_index, load_metadata, search_index
from Retrieval.sailer_encoder import SailerEncoder


@lru_cache(maxsize=4)
def get_runtime_config(config_path: str | None = None) -> Dict:
    return load_react_config(config_path)


@lru_cache(maxsize=2)
def get_shared_encoder(config_path: str | None = None) -> SailerEncoder:
    cfg = get_runtime_config(config_path)
    runtime_cfg = cfg.get("runtime", {})
    if runtime_cfg.get("cuda_visible_devices") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime_cfg["cuda_visible_devices"])
    retrieval_cfg = cfg.get("retrieval", {})
    return SailerEncoder(
        str(resolve_repo_path(cfg["paths"]["sailer_model"])),
        batch_size=int(retrieval_cfg.get("batch_size", 64)),
        max_length=int(retrieval_cfg.get("max_length", 512)),
        device=runtime_cfg.get("retrieval_device", "cpu"),
    )


@lru_cache(maxsize=16)
def load_index_bundle(index_path: str, metadata_path: str) -> Tuple[object, List[Dict]]:
    return load_index(index_path), load_metadata(metadata_path)


@lru_cache(maxsize=16)
def load_context_rows(path: str) -> List[Dict]:
    return read_jsonl(path)


def retrieve_dense(index_path: Path, metadata_path: Path, query: str, topk: int, *, config_path: str | None = None, search_topn: int | None = None) -> List[Tuple[float, Dict]]:
    encoder = get_shared_encoder(config_path)
    index, metadata = load_index_bundle(str(index_path), str(metadata_path))
    query_vector = encoder.encode([query])
    request_topn = min(len(metadata), max(topk, search_topn or topk))
    scores, indices = search_index(index, query_vector, request_topn)
    results: List[Tuple[float, Dict]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        results.append((float(score), metadata[idx]))
    return results
