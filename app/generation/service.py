from __future__ import annotations

import time

from app.generation.base import Generator
from app.models import Answer, RetrievalFilters, RetrievalOutcome
from app.rag.base import REFUSAL_TEXT as REFUSAL_TEXT
from app.rag.base import RagStrategy
from app.rag.vanilla import VanillaRagStrategy
from app.retrieval.base import Retriever
from app.retrieval.evidence import DiversifiedEvidenceSelector, EvidenceSelector
from app.settings import GenerationSettings, RetrievalSettings
from app.tracing import emit_trace


class ChatService:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        retrieval_settings: RetrievalSettings,
        evidence_selector: EvidenceSelector | None = None,
        generation_settings: GenerationSettings | None = None,
        strategy: RagStrategy | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.settings = retrieval_settings
        self.evidence_selector = evidence_selector or DiversifiedEvidenceSelector(
            top_k=retrieval_settings.evidence_top_k,
            min_score=retrieval_settings.min_score,
            per_topic_limit=retrieval_settings.per_topic_limit,
        )
        self.generation_settings = generation_settings or GenerationSettings()
        self.strategy = strategy or VanillaRagStrategy(
            retriever=retriever,
            generator=generator,
            retrieval_settings=retrieval_settings,
            evidence_selector=self.evidence_selector,
            generation_settings=self.generation_settings,
        )

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalOutcome:
        return await self.strategy.retrieve(question, top_k, filters)

    async def answer(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
    ) -> Answer:
        started = time.perf_counter()
        emit_trace(
            "request",
            "request.received",
            status="started",
            payload={
                "question": question,
                "filters": (filters or RetrievalFilters()).model_dump(mode="json"),
                "candidate_top_k": self.settings.candidate_top_k,
                "evidence_top_k": self.settings.evidence_top_k,
            },
        )
        return await self.strategy.answer(
            question,
            filters,
            started=started,
        )

    async def answer_from_outcome(
        self,
        question: str,
        outcome: RetrievalOutcome,
        filters: RetrievalFilters | None = None,
        *,
        started: float | None = None,
    ) -> Answer:
        return await self.strategy.answer_from_outcome(
            question=question,
            outcome=outcome,
            filters=filters,
            started=started,
        )
