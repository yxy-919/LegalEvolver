from __future__ import annotations

import json
import os
from typing import Any


class JudgeClient:
    """Model-agnostic client for any strong OpenAI-compatible judge."""

    def __init__(self) -> None:
        from openai import OpenAI

        self.model = os.getenv("JUDGE_MODEL", "judge-model")
        self.client = OpenAI(
            api_key=os.getenv("JUDGE_API_KEY", "EMPTY"),
            base_url=os.environ["JUDGE_BASE_URL"],
        )

    def complete_json(self, prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.choices[0].message.content or "{}")
