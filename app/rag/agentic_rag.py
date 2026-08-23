from __future__ import annotations

import time

from app.models import Answer, RetrievalFilters, RetrievalOutcome, RetrievalQuery, SearchResult
from app.rag.common import StrategyBase
from app.rag.controller import as_agent_action
from app.rag.vanilla import VanillaRagStrategy


class AgenticRagStrategy(StrategyBase):
    """Bounded retrieve/answer/stop controller loop with deterministic final generation."""

    strategy_name = "agentic_rag"

    def __init__(
        self,
        retriever,
        generator,
        retrieval_settings,
        *,
        controller=None,
        max_steps: int = 4,
        evidence_selector=None,
        generation_settings=None,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        super().__init__(
            VanillaRagStrategy(
                retriever, generator, retrieval_settings, evidence_selector, generation_settings
            ),
            controller,
        )
        self.max_steps = max_steps

    @staticmethod
    def _merge(candidates: list[SearchResult], incoming: list[SearchResult]) -> list[SearchResult]:
        merged = {item.chunk.chunk_id: item for item in candidates}
        for item in incoming:
            old = merged.get(item.chunk.chunk_id)
            if old is None or item.score > old.score:
                merged[item.chunk.chunk_id] = item
        ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return [item.model_copy(update={"rank": rank}) for rank, item in enumerate(ordered, 1)]

    async def retrieve(
        self, question: str, top_k: int | None = None, filters: RetrievalFilters | None = None
    ) -> RetrievalOutcome:
        selected_top_k = top_k or self.vanilla.settings.candidate_top_k
        started = time.perf_counter()
        candidates: list[SearchResult] = []
        retrieval_ms = 0.0
        reason = "action_stop"
        for step in range(self.max_steps):
            raw = await self._controller(
                ("next_action", "choose_action", "plan"), question, candidates, step
            )
            action = as_agent_action(raw)
            self._trace(
                "strategy.step",
                step=step + 1,
                action=action.action,
                query=action.query,
                candidates=len(candidates),
            )
            if action.action == "retrieve":
                query = action.query or question
                outcome = await self.vanilla.retrieve(query, selected_top_k, filters)
                retrieval_ms += outcome.timings_ms.get("total", 0.0)
                candidates = self._merge(candidates, outcome.results)
                reason = "retrieved"
                continue
            if action.action == "answer":
                reason = "controller_answer"
                break
            reason = action.reason or "controller_stop"
            break
        else:
            reason = "max_steps_reached"
        self._trace(
            "strategy.stop",
            reason=reason,
            steps=min(self.max_steps, step + 1),
            candidates=len(candidates),
        )
        total_ms = round((time.perf_counter() - started) * 1000, 3)
        return RetrievalOutcome(
            results=candidates,
            timings_ms={
                "agentic_controller": round(max(0.0, total_ms - retrieval_ms), 3),
                "agentic_retrieval": round(retrieval_ms, 3),
                "agentic_steps": float(min(self.max_steps, step + 1)),
                "total": total_ms,
            },
            query=RetrievalQuery(text=question, filters=filters or RetrievalFilters()),
        )

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
        return await self.vanilla.answer_from_outcome(question, outcome, filters, started=started)
