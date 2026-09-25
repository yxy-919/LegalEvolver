from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


class JudgeClient:
    """Model-agnostic client for an OpenAI-compatible judge endpoint."""

    def __init__(self) -> None:
        self.model = os.getenv("JUDGE_MODEL", "judge-model")
        self.client = OpenAI(
            api_key=os.environ["JUDGE_API_KEY"],
            base_url=os.environ["JUDGE_BASE_URL"],
        )

    def score(self, prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.choices[0].message.content or "{}")

