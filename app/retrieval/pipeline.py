from __future__ import annotations

import time

from app.models import RetrievalFilters, RetrievalOutcome, RetrievalQuery
from app.retrieval.base import QueryProcessor, Reranker, Retriever
from app.settings import RetrievalSettings
from app.tracing import emit_trace
from app.tracing.serializers import serialize_candidate


class RetrievalPipeline:
    def __init__(
        self,
        query_processor: QueryProcessor,
        retriever: Retriever,
        reranker: Reranker,
        settings: RetrievalSettings,
    ) -> None:
        self.query_processor = query_processor
        self.candidate_retriever = retriever
        self.reranker = reranker
        self.settings = settings

    @property
    def component_id(self) -> str:
        return (
            f"pipeline:{self.query_processor.component_id}:"
            f"{self.candidate_retriever.component_id}:{self.reranker.component_id}"
        )

    def retrieve(
        self,
        question: str | RetrievalQuery,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalOutcome:
        query = (
            question
            if isinstance(question, RetrievalQuery)
            else self.query_processor.process(question, filters)
        )
        candidate_k = max(top_k, self.settings.candidate_top_k)
        outcome = self.candidate_retriever.retrieve(query, candidate_k)
        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(query, outcome.results)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        timings = dict(outcome.timings_ms)
        timings["rerank"] = round(rerank_ms, 3)
        timings["total"] = round(sum(value for key, value in timings.items() if key != "total"), 3)
        emit_trace(
            "retrieval",
            "retrieval.pipeline.completed",
            elapsed_ms=timings["total"],
            payload={
                "component_id": self.component_id,
                "candidate_top_k": candidate_k,
                "returned_top_k": top_k,
                "candidates": [serialize_candidate(result) for result in reranked[:top_k]],
            },
        )
        return RetrievalOutcome(results=reranked[:top_k], timings_ms=timings, query=query)
