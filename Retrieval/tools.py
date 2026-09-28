from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .encoder import SailerEncoder
from .index import DenseIndex


@dataclass(frozen=True)
class ToolSpec:
    name: str
    source_type: str
    index_path: Path
    metadata_path: Path


class LegalRetrievalEnvironment:
    """Expose the four retrieval tools used by LegalEvolver.

    The release intentionally consumes prebuilt indexes. Corpus processing and
    index construction are outside the scope of this repository.
    """

    TOOL_NAMES = (
        "statute_search",
        "case_search",
        "guideline_search",
        "judicial_interpretation_search",
    )

    def __init__(
        self,
        config_path: str | Path,
        reranker: Callable[[str, list[tuple[float, dict[str, Any]]]], list[tuple[float, dict[str, Any]]]] | None = None,
    ) -> None:
        config_file = Path(config_path)
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        expand = lambda value: os.path.expandvars(str(value))
        index_root = Path(expand(config["index_root"]))
        if not index_root.is_absolute():
            index_root = (config_file.parent / index_root).resolve()
        self.encoder = SailerEncoder(expand(config["encoder_path"]))
        self.max_tool_calls = int(config.get("max_tool_calls", 10))
        self.top_k = int(config.get("top_k", 4))
        self.candidate_count = int(config.get("candidate_count", 24))
        self.reranker = reranker or self._dense_reranker
        self.call_count = 0
        self.tools: dict[str, tuple[ToolSpec, DenseIndex]] = {}

        for name in self.TOOL_NAMES:
            item = config["tools"][name]
            spec = ToolSpec(
                name=name,
                source_type=item["source_type"],
                index_path=index_root / item["index"],
                metadata_path=index_root / item["metadata"],
            )
            self.tools[name] = (spec, DenseIndex(spec.index_path, spec.metadata_path))

    @staticmethod
    def _dense_reranker(
        query: str,
        candidates: list[tuple[float, dict[str, Any]]],
    ) -> list[tuple[float, dict[str, Any]]]:
        del query
        return sorted(candidates, key=lambda item: item[0], reverse=True)

    def reset(self) -> None:
        self.call_count = 0

    def call(self, tool_name: str, query: str, top_k: int | None = None) -> dict[str, Any]:
        if tool_name not in self.tools:
            raise KeyError(f"unknown retrieval tool: {tool_name}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if self.call_count >= self.max_tool_calls:
            raise RuntimeError("maximum number of tool calls reached")

        self.call_count += 1
        spec, index = self.tools[tool_name]
        vector = self.encoder.encode([query.strip()])
        candidates = index.search(vector, self.candidate_count)
        ranked = self.reranker(query, candidates)[: int(top_k or self.top_k)]
        return {
            "tool_name": tool_name,
            "query": query,
            "results": [
                {
                    "source_type": spec.source_type,
                    "score": score,
                    **document,
                }
                for score, document in ranked
            ],
        }

    def statute_search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        return self.call("statute_search", query, top_k)

    def case_search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        return self.call("case_search", query, top_k)

    def guideline_search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        return self.call("guideline_search", query, top_k)

    def judicial_interpretation_search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        return self.call("judicial_interpretation_search", query, top_k)
