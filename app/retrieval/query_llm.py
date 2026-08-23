"""Small synchronous, provider-neutral query planning client.

The response is deliberately returned as text: processors own the schema
validation, so malformed model output is an explicit error at the boundary.
"""

from __future__ import annotations

import json
from typing import Protocol

from openai import OpenAI


class QueryPlannerLLM(Protocol):
    def plan(self, question: str, *, strategy: str) -> str:
        """Return a JSON object suitable for the requested strategy."""


class OpenAICompatibleQueryPlanner:
    """Synchronous client for OpenRouter and OpenAI-compatible chat APIs."""

    def __init__(
        self, api_key: str, model: str, *, base_url: str, timeout_seconds: float = 30.0
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not model:
            raise ValueError("model must not be empty")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
        )

    @property
    def component_id(self) -> str:
        return f"openai-compatible:{self.model}"

    def plan(self, question: str, *, strategy: str) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")
        instructions = {
            "rewrite": (
                "Rewrite the question as one concise, self-contained search query without "
                'answering it. Return {"query":"..."}.'
            ),
            "hyde": (
                "Write one short hypothetical manual passage that would answer the question; "
                'the passage will be embedded for retrieval. Return {"query":"..."}.'
            ),
            "multi": (
                "Generate distinct search queries expressing different useful formulations of "
                'the same information need. Return {"queries":["...", "..."]}.'
            ),
            "decomposition": (
                "Split the question into independently searchable subquestions needed to answer "
                'the whole question. Return {"queries":["...", "..."]}.'
            ),
        }
        try:
            instruction = instructions[strategy]
        except KeyError as exc:
            raise ValueError(f"unsupported query planning strategy: {strategy}") from exc
        prompt = (
            "Return ONLY one valid JSON object with no markdown or extra keys. "
            f"{instruction} Question: {question}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("provider returned empty content")
            # Validate JSON here while retaining the original structured text.
            json.loads(content)
            return content
        except Exception as exc:
            raise RuntimeError(f"query planner request failed: {exc}") from exc


class OpenRouterQueryPlanner(OpenAICompatibleQueryPlanner):
    def __init__(self, api_key: str, model: str, *, timeout_seconds: float = 30.0) -> None:
        super().__init__(
            api_key, model, base_url="https://openrouter.ai/api/v1", timeout_seconds=timeout_seconds
        )


class OpenAIQueryPlanner(OpenAICompatibleQueryPlanner):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(api_key, model, base_url=base_url, timeout_seconds=timeout_seconds)
