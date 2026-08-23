from __future__ import annotations

from typing import Protocol

from app.models import Answer, RetrievalFilters, RetrievalOutcome

REFUSAL_TEXT = "知识库中未找到足够依据。"


class RagStrategy(Protocol):
    @property
    def component_id(self) -> str: ...

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalOutcome: ...

    async def answer(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
        *,
        started: float | None = None,
    ) -> Answer: ...

    async def answer_from_outcome(
        self,
        question: str,
        outcome: RetrievalOutcome,
        filters: RetrievalFilters | None = None,
        *,
        started: float | None = None,
    ) -> Answer: ...
