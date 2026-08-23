from __future__ import annotations

import time

from app.models import Answer, RetrievalFilters, RetrievalOutcome
from app.rag.common import StrategyBase
from app.rag.controller import as_evidence_assessment, as_support_assessment
from app.rag.vanilla import VanillaRagStrategy
from app.tracing import emit_trace


class SelfRagStrategy(StrategyBase):
    """Self-RAG baseline: one evidence check, at most one rewrite, one support check."""

    strategy_name = "self_rag"

    def __init__(
        self,
        retriever,
        generator,
        retrieval_settings,
        *,
        controller=None,
        evidence_selector=None,
        generation_settings=None,
    ):
        super().__init__(
            VanillaRagStrategy(
                retriever, generator, retrieval_settings, evidence_selector, generation_settings
            ),
            controller,
        )

    async def retrieve(
        self, question: str, top_k: int | None = None, filters: RetrievalFilters | None = None
    ) -> RetrievalOutcome:
        outcome = await self.vanilla.retrieve(question, top_k, filters)
        initial_total = outcome.timings_ms.get("total", sum(outcome.timings_ms.values()))
        outcome.timings_ms["self_rag_retrieval_initial"] = round(initial_total, 3)
        self._trace(
            "strategy.retrieve", phase="initial", query=question, candidates=len(outcome.results)
        )
        controller_started = time.perf_counter()
        raw = await self._controller(("assess_evidence", "evaluate_evidence"), question, outcome)
        outcome.timings_ms["self_rag_controller_evidence"] = round(
            (time.perf_counter() - controller_started) * 1000, 3
        )
        outcome.timings_ms["self_rag_controller"] = outcome.timings_ms[
            "self_rag_controller_evidence"
        ]
        outcome.timings_ms["total"] = round(
            initial_total + outcome.timings_ms["self_rag_controller"], 3
        )
        assessment = as_evidence_assessment(raw, default=bool(outcome.results))
        self._trace(
            "strategy.evidence_assessed", sufficient=assessment.sufficient, reason=assessment.reason
        )
        if assessment.sufficient:
            return outcome
        controller_started = time.perf_counter()
        rewritten = await self._controller(("rewrite_query", "rewrite"), question, outcome)
        outcome.timings_ms["self_rag_controller_rewrite"] = round(
            (time.perf_counter() - controller_started) * 1000, 3
        )
        outcome.timings_ms["self_rag_controller"] = round(
            outcome.timings_ms["self_rag_controller"]
            + outcome.timings_ms["self_rag_controller_rewrite"],
            3,
        )
        outcome.timings_ms["total"] = round(
            initial_total + outcome.timings_ms["self_rag_controller"], 3
        )
        if (
            not isinstance(rewritten, str)
            or not rewritten.strip()
            or rewritten.strip() == question.strip()
        ):
            self._trace("strategy.stop", reason="evidence_insufficient_no_rewrite")
            return outcome
        rewrite_ms = outcome.timings_ms["self_rag_controller_rewrite"]
        outcome.timings_ms["self_rag_rewrite"] = rewrite_ms
        self._trace("strategy.rewrite", query=rewritten, elapsed_ms=rewrite_ms)
        revised = await self.vanilla.retrieve(rewritten, top_k, filters)
        revised_total = revised.timings_ms.get("total", sum(revised.timings_ms.values()))
        # Keep both retrieval rounds visible; the revised outcome must not erase
        # the initial round's provider-specific timings.
        merged = dict(outcome.timings_ms)
        for key, value in revised.timings_ms.items():
            merged[f"self_rag_rewritten_{key}"] = value
        merged["self_rag_retrieval_rewritten"] = round(revised_total, 3)
        merged["self_rag_retrieval"] = round(initial_total + revised_total, 3)
        merged["total"] = round(initial_total + revised_total + merged["self_rag_controller"], 3)
        revised.timings_ms = merged
        self._trace(
            "strategy.retrieve", phase="rewritten", query=rewritten, candidates=len(revised.results)
        )
        return revised

    async def answer(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
        *,
        started: float | None = None,
    ) -> Answer:
        started = started or time.perf_counter()
        outcome = await self.retrieve(question, self.vanilla.settings.candidate_top_k, filters)
        return await self.answer_from_outcome(question, outcome, filters, started=started)

    async def answer_from_outcome(
        self,
        question: str,
        outcome: RetrievalOutcome,
        filters=None,
        *,
        started: float | None = None,
    ) -> Answer:
        started = started or time.perf_counter()
        answer = await self.vanilla.answer_from_outcome(question, outcome, filters, started=started)
        if answer.refused:
            self._trace("strategy.stop", reason="vanilla_refusal")
            return answer
        controller_started = time.perf_counter()
        raw = await self._controller(
            ("assess_support", "evaluate_support"), question, answer.text, answer.evidence
        )
        answer.timings_ms["self_rag_controller_support"] = round(
            (time.perf_counter() - controller_started) * 1000, 3
        )
        answer.timings_ms["self_rag_controller"] = round(
            answer.timings_ms.get("self_rag_controller", 0.0)
            + answer.timings_ms["self_rag_controller_support"],
            3,
        )
        answer.timings_ms["total"] = round(
            answer.timings_ms.get("total", 0.0) + answer.timings_ms["self_rag_controller_support"],
            3,
        )
        support = as_support_assessment(raw, default=True)
        self._trace("strategy.support_assessed", supported=support.supported, reason=support.reason)
        if support.supported:
            self._trace("strategy.stop", reason="answer_supported")
            return answer
        refused = answer.model_copy(
            update={
                "text": "知识库中未找到足够依据。",
                "citations": [],
                "refused": True,
                "citation_validated": True,
            }
        )
        emit_trace(
            "response",
            "response.completed",
            elapsed_ms=refused.timings_ms.get("total"),
            payload={**refused.model_dump(mode="json"), "strategy": self.strategy_name},
        )
        self._trace("strategy.stop", reason="unsupported_answer_refused")
        return refused
