"""Small controller contracts used by the optional RAG strategies.

Controllers are deliberately separate from retrieval and generation.  They may be
backed by a real model, or replaced by a deterministic fake in tests.  No
controller is trusted to produce answer text or citations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Literal, Protocol

from openai import AsyncOpenAI

from app.models import RetrievalOutcome, SearchResult
from app.settings import GenerationSettings
from app.tracing import emit_trace


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reason: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class SupportAssessment:
    supported: bool
    reason: str = ""
    confidence: float | None = None


ActionName = Literal["retrieve", "answer", "stop"]


@dataclass(frozen=True)
class AgentAction:
    action: ActionName
    query: str | None = None
    reason: str = ""


class RagController(Protocol):
    async def assess_evidence(
        self, question: str, outcome: RetrievalOutcome
    ) -> EvidenceAssessment: ...

    async def rewrite_query(self, question: str, outcome: RetrievalOutcome) -> str: ...

    async def assess_support(
        self, question: str, draft: str, evidence: list[SearchResult]
    ) -> SupportAssessment: ...

    async def next_action(
        self, question: str, candidates: list[SearchResult], steps: int
    ) -> AgentAction: ...


async def invoke_controller(controller: Any, method: str, *args: Any) -> Any:
    """Call sync or async fakes equally; useful for tiny unit-test controllers."""
    value = getattr(controller, method)(*args)
    return await value if isawaitable(value) else value


def as_evidence_assessment(value: Any, *, default: bool = False) -> EvidenceAssessment:
    if isinstance(value, EvidenceAssessment):
        return value
    if isinstance(value, bool):
        return EvidenceAssessment(value)
    if isinstance(value, dict):
        sufficient = value.get("sufficient", value.get("enough", value.get("supported", default)))
        return EvidenceAssessment(
            bool(sufficient), str(value.get("reason", "")), value.get("confidence")
        )
    return EvidenceAssessment(default)


def as_support_assessment(value: Any, *, default: bool = False) -> SupportAssessment:
    if isinstance(value, SupportAssessment):
        return value
    if isinstance(value, bool):
        return SupportAssessment(value)
    if isinstance(value, dict):
        supported = value.get("supported", value.get("sufficient", value.get("enough", default)))
        return SupportAssessment(
            bool(supported), str(value.get("reason", "")), value.get("confidence")
        )
    return SupportAssessment(default)


def as_agent_action(value: Any) -> AgentAction:
    if isinstance(value, AgentAction):
        return value
    if isinstance(value, str):
        action = value.strip().lower()
        return AgentAction(action if action in {"retrieve", "answer", "stop"} else "stop")  # type: ignore[arg-type]
    if isinstance(value, dict):
        action = str(value.get("action", value.get("type", "stop"))).lower()
        if action not in {"retrieve", "answer", "stop"}:
            action = "stop"
        return AgentAction(action, value.get("query"), str(value.get("reason", "")))  # type: ignore[arg-type]
    return AgentAction("stop", reason="invalid_controller_action")


class HeuristicRagController:
    """Safe no-network defaults: retrieve once, then answer."""

    async def assess_evidence(self, question: str, outcome: RetrievalOutcome) -> EvidenceAssessment:
        del question
        return EvidenceAssessment(bool(outcome.results), reason="candidate_presence")

    async def rewrite_query(self, question: str, outcome: RetrievalOutcome) -> str:
        del outcome
        return question

    async def assess_support(
        self, question: str, draft: str, evidence: list[SearchResult]
    ) -> SupportAssessment:
        del question, draft
        return SupportAssessment(bool(evidence), reason="candidate_presence")

    async def next_action(
        self, question: str, candidates: list[SearchResult], steps: int
    ) -> AgentAction:
        del question
        return AgentAction("answer" if candidates else "retrieve", reason=f"heuristic_step_{steps}")


class OpenRouterJSONController(HeuristicRagController):
    """OpenRouter-backed JSON controller.

    This class is optional and never used by tests unless explicitly injected.
    The final answer still comes from the normal Generator/Vanilla path.
    """

    def __init__(self, settings: GenerationSettings, api_key: str) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=settings.timeout_seconds,
        )

    @property
    def component_id(self) -> str:
        return f"openrouter-controller:{self.settings.model}"

    @staticmethod
    def _candidates(items: list[SearchResult], *, limit: int = 6) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": item.chunk.chunk_id,
                "title": item.chunk.title,
                "section_path": item.chunk.section_path,
                "score": item.score,
                "excerpt": item.chunk.text[:400],
            }
            for item in items[:limit]
        ]

    async def _json(self, operation: str, payload: dict[str, Any]) -> Any:
        started = time.perf_counter()
        trace_payload = {
            "operation": operation,
            "model": self.settings.model,
        }
        emit_trace(
            "controller", "controller.input", status="started", payload={"operation": operation}
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是RAG流程控制器，只输出合法JSON，不输出Markdown。"
                            "assess_evidence输出{sufficient:boolean,reason:string};"
                            "rewrite_query输出{query:string};"
                            "assess_support输出{supported:boolean,reason:string};"
                            "next_action输出{action:retrieve|answer|stop,query?:string,reason:string}。"
                            "只能根据输入中的问题、候选证据和草稿判断，不补充外部事实。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"operation": operation, **payload}, ensure_ascii=False
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            emit_trace(
                "controller",
                "controller.output",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                payload={**trace_payload, "result": result},
            )
            return result
        except Exception as exc:
            emit_trace(
                "controller",
                "controller.output",
                status="failed",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                payload={**trace_payload, "error": type(exc).__name__},
            )
            raise

    async def assess_evidence(self, question: str, outcome: RetrievalOutcome) -> EvidenceAssessment:
        result = await self._json(
            "assess_evidence",
            {
                "question": question,
                "candidates": self._candidates(outcome.results),
            },
        )
        return as_evidence_assessment(result)

    async def rewrite_query(self, question: str, outcome: RetrievalOutcome) -> str:
        result = await self._json(
            "rewrite_query",
            {
                "question": question,
                "candidates": self._candidates(outcome.results),
            },
        )
        return str(result.get("query", question)) if isinstance(result, dict) else question

    async def assess_support(
        self, question: str, draft: str, evidence: list[SearchResult]
    ) -> SupportAssessment:
        result = await self._json(
            "assess_support",
            {
                "question": question,
                "draft": draft,
                "evidence": self._candidates(evidence),
            },
        )
        return as_support_assessment(result)

    async def next_action(
        self, question: str, candidates: list[SearchResult], steps: int
    ) -> AgentAction:
        result = await self._json(
            "next_action",
            {
                "question": question,
                "candidates": self._candidates(candidates),
                "steps": steps,
            },
        )
        return as_agent_action(result)
