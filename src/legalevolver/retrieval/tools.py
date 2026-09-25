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
    source: str
    index: Path
    metadata: Path


class LegalRetrievalEnvironment:
    """Four legal retrieval tools sharing one encoder and a global call budget."""

    def __init__(self, config_path: str | Path, reranker: Callable | None = None) -> None:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        expanded = lambda value: os.path.expandvars(str(value))
        root = Path(expanded(config["index_root"]))
        self.encoder = SailerEncoder(expanded(config["encoder_path"]))
        self.max_calls = int(config.get("max_tool_calls", 10))
        self.top_k = int(config.get("top_k", 4))
        self.candidate_count = int(config.get("rerank_candidates", 24))
        self.reranker = reranker or self._default_reranker
        self.calls = 0
        self.tools: dict[str, tuple[ToolSpec, DenseIndex]] = {}
        for name, item in config["tools"].items():
            spec = ToolSpec(
                name=name,
                source=item["source"],
                index=root / item["index"],
                metadata=root / item["metadata"],
            )
            self.tools[name] = (spec, DenseIndex(spec.index, spec.metadata))

    @staticmethod
    def _default_reranker(query: str, candidates: list[tuple[float, dict[str, Any]]]) -> list:
        """Dense-score fallback; source-specific legal rerankers can be injected."""
        del query
        return sorted(candidates, key=lambda item: item[0], reverse=True)

    def reset(self) -> None:
        self.calls = 0

    def search(self, tool_name: str, query: str) -> list[dict[str, Any]]:
        if self.calls >= self.max_calls:
            raise RuntimeError("maximum number of tool calls reached")
        if tool_name not in self.tools:
            raise KeyError(f"unknown retrieval tool: {tool_name}")
        self.calls += 1
        _, index = self.tools[tool_name]
        vector = self.encoder.encode([query])
        candidates = index.search(vector, self.candidate_count)
        ranked = self.reranker(query, candidates)[: self.top_k]
        return [{"score": score, **document} for score, document in ranked]

