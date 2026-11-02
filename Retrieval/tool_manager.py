from __future__ import annotations

from typing import Dict

from Retrieval.case_retriever import case_search
from Retrieval.guideline_retriever import guideline_search
from Retrieval.judicial_interpretation_retriever import judicial_interpretation_search
from Retrieval.statute_retriever import statute_search

_ALLOWED_KEYS = {"tool_name", "query", "topk"}
_TOOLS = {
    "statute_search": statute_search,
    "case_search": case_search,
    "guideline_search": guideline_search,
    "judicial_interpretation_search": judicial_interpretation_search,
}


class ToolManager:
    def __init__(self, *, topk_default: int = 4, topk_max: int = 4):
        self.topk_default = topk_default
        self.topk_max = topk_max

    def sanitize_input(self, action_input: Dict | None) -> Dict:
        action_input = dict(action_input or {})
        cleaned: Dict = {}
        if "query" in action_input:
            cleaned["query"] = str(action_input["query"])
        topk = action_input.get("topk", self.topk_default)
        try:
            topk = int(topk)
        except Exception:
            topk = self.topk_default
        cleaned["topk"] = max(1, min(self.topk_max, topk))
        return cleaned

    def call_tool(self, tool_name: str, action_input: Dict | None) -> Dict:
        if tool_name not in _TOOLS:
            return {"error": f"unknown tool: {tool_name}"}
        raw_input = dict(action_input or {})
        extra_fields = sorted(key for key in raw_input.keys() if key not in _ALLOWED_KEYS)
        cleaned_input = self.sanitize_input(raw_input)
        warnings = []
        if extra_fields:
            warnings.append(f"dropped extra fields: {extra_fields}")
        if not cleaned_input.get("query"):
            return {
                "error": "missing query",
                "tool_name": tool_name,
                "raw_input": raw_input,
                "cleaned_input": cleaned_input,
                "warnings": warnings,
            }
        result = _TOOLS[tool_name](cleaned_input["query"], cleaned_input["topk"])
        result["raw_input"] = raw_input
        result["cleaned_input"] = cleaned_input
        result.setdefault("warnings", []).extend(warnings)
        return result
