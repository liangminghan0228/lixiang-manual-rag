from __future__ import annotations

import asyncio
import re
import time

from app.generation.base import Generator
from app.models import Answer, Citation, RetrievalFilters, RetrievalOutcome, RetrievalQuery
from app.rag.base import REFUSAL_TEXT
from app.retrieval.base import Retriever
from app.retrieval.evidence import DiversifiedEvidenceSelector, EvidenceSelector
from app.settings import GenerationSettings, RetrievalSettings
from app.tracing import emit_trace
from app.tracing.serializers import serialize_candidate


class VanillaRagStrategy:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        retrieval_settings: RetrievalSettings,
        evidence_selector: EvidenceSelector | None = None,
        generation_settings: GenerationSettings | None = None,
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

    @property
    def component_id(self) -> str:
        return "vanilla-rag-v1"

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalOutcome:
        selected_top_k = top_k or self.settings.top_k
        return await asyncio.to_thread(
            self.retriever.retrieve,
            question,
            selected_top_k,
            filters,
        )

    @staticmethod
    def _citation_indexes(text: str, evidence_count: int) -> tuple[set[int], bool]:
        indexes = {int(value) for value in re.findall(r"\[(\d+)]", text)}
        return indexes, bool(indexes) and all(1 <= value <= evidence_count for value in indexes)

    async def answer(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
        *,
        started: float | None = None,
    ) -> Answer:
        started = started or time.perf_counter()
        outcome = await self.retrieve(question, self.settings.candidate_top_k, filters)
        return await self.answer_from_outcome(
            question,
            outcome,
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
        started = started or time.perf_counter()
        query = outcome.query or RetrievalQuery(
            text=question, filters=filters or RetrievalFilters()
        )
        evidence_bundle = self.evidence_selector.select(query, outcome.results)
        evidence = evidence_bundle.items

        if not evidence:
            total_ms = (time.perf_counter() - started) * 1000
            timings = dict(outcome.timings_ms)
            timings["llm"] = 0.0
            timings["total"] = round(total_ms, 3)
            answer = Answer(
                text=REFUSAL_TEXT,
                citations=[],
                evidence=[],
                timings_ms=timings,
                refused=True,
                citation_validated=True,
            )
            emit_trace(
                "response",
                "response.completed",
                elapsed_ms=timings["total"],
                payload=answer.model_dump(mode="json"),
            )
            return answer

        llm_started = time.perf_counter()
        text = await self.generator.generate(question, evidence)
        citation_indexes, citation_valid = self._citation_indexes(text, len(evidence))
        emit_trace(
            "citation",
            "citation.validated",
            payload={
                "operation": "generate",
                "parsed_indexes": sorted(citation_indexes),
                "evidence_count": len(evidence),
                "valid": citation_valid,
            },
        )
        repair_attempts = 0
        while (
            self.generation_settings.require_inline_citations
            and not citation_valid
            and repair_attempts < self.generation_settings.citation_repair_attempts
        ):
            emit_trace(
                "citation",
                "citation.repair.started",
                status="started",
                payload={"attempt": repair_attempts + 1, "draft": text},
            )
            text = await self.generator.repair_citations(question, text, evidence)
            repair_attempts += 1
            citation_indexes, citation_valid = self._citation_indexes(text, len(evidence))
            emit_trace(
                "citation",
                "citation.repair.completed",
                payload={
                    "attempt": repair_attempts,
                    "text": text,
                    "parsed_indexes": sorted(citation_indexes),
                    "valid": citation_valid,
                },
            )

        llm_ms = (time.perf_counter() - llm_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        timings = dict(outcome.timings_ms)
        timings["llm"] = round(llm_ms, 3)
        timings["total"] = round(total_ms, 3)

        if self.generation_settings.require_inline_citations and not citation_valid:
            answer = Answer(
                text=REFUSAL_TEXT,
                citations=[],
                evidence=evidence,
                timings_ms=timings,
                refused=True,
                citation_validated=False,
            )
            emit_trace(
                "response",
                "response.completed",
                elapsed_ms=timings["total"],
                payload=answer.model_dump(mode="json"),
            )
            return answer

        used_indexes = citation_indexes or set(range(1, len(evidence) + 1))
        citations = [
            Citation(
                index=index,
                chunk_id=result.chunk.chunk_id,
                title=result.chunk.title,
                source_url=result.chunk.source_url,
                section_path=result.chunk.section_path,
                excerpt=result.chunk.text.split("正文：", 1)[-1][:240],
            )
            for index, result in enumerate(evidence, start=1)
            if index in used_indexes
        ]

        answer = Answer(
            text=text,
            citations=citations,
            evidence=evidence,
            timings_ms=timings,
            refused=False,
            citation_validated=citation_valid,
        )
        emit_trace(
            "response",
            "response.completed",
            elapsed_ms=timings["total"],
            payload={
                **answer.model_dump(mode="json"),
                "repair_attempts": repair_attempts,
                "selected_evidence": [serialize_candidate(item) for item in evidence],
            },
        )
        return answer
