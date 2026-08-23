from __future__ import annotations

import time

from app.models import QueryPlan, RetrievalFilters, RetrievalOutcome, RetrievalQuery, SearchResult
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
        query_plan = self._normalize_query(question, filters)
        candidate_k = max(top_k, self.settings.candidate_top_k)
        valid_queries = [value.strip() for value in query_plan.queries if value.strip()]
        if query_plan.fusion_strategy == "single_query":
            if len(valid_queries) > 1:
                raise ValueError("single_query strategy accepts at most one non-empty query")
            query = valid_queries[0] if valid_queries else query_plan.text.strip()
            if not query:
                raise ValueError("retrieval query must not be empty")
            candidate_outcome = self.candidate_retriever.retrieve(
                query, candidate_k, query_plan.filters
            )
            candidates = candidate_outcome.results
            timings = dict(candidate_outcome.timings_ms)
        elif query_plan.fusion_strategy == "rrf":
            if not valid_queries and query_plan.text.strip():
                valid_queries = [query_plan.text.strip()]
            if not valid_queries:
                raise ValueError("rrf strategy requires at least one non-empty query")
            candidates, timings = self._retrieve_rrf(query_plan, valid_queries, candidate_k)
        elif query_plan.fusion_strategy == "multi_query":
            # Keep the historical error type for callers that used the old
            # experimental strategy name; it remains an explicit failure.
            raise NotImplementedError("Multi-query fusion strategy is unsupported; use rrf instead")
        else:
            raise ValueError(f"unsupported fusion strategy: {query_plan.fusion_strategy!r}")

        planner_ms = query_plan.metadata.get("planner_elapsed_ms")
        if isinstance(planner_ms, (int, float)):
            timings["query_planner"] = round(float(planner_ms), 3)

        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(query_plan, candidates)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
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
        return RetrievalOutcome(results=reranked[:top_k], timings_ms=timings, query=query_plan)

    def _retrieve_rrf(
        self, query_plan: QueryPlan, queries: list[str], candidate_k: int
    ) -> tuple[list[SearchResult], dict[str, float]]:
        started = time.perf_counter()
        fused: dict[str, tuple[float, SearchResult, int]] = {}
        timings: dict[str, float] = {}
        for query_index, query in enumerate(queries):
            outcome = self.candidate_retriever.retrieve(query, candidate_k, query_plan.filters)
            timings[f"candidate_{query_index}"] = outcome.timings_ms.get(
                "total", sum(outcome.timings_ms.values())
            )
            for result in outcome.results:
                chunk_id = result.chunk.chunk_id
                score, first, first_seen = fused.get(chunk_id, (0.0, result, len(fused)))
                contribution = 1.0 / (self.settings.rrf_k + result.rank)
                fused[chunk_id] = (score + contribution, first, first_seen)

        ranked = sorted(fused.values(), key=lambda item: (-item[0], item[2]))
        results = [
            SearchResult(
                chunk=first.chunk,
                score=score,
                recall_score=score,
                rank=rank,
                retriever_id=first.retriever_id,
            )
            for rank, (score, first, _) in enumerate(ranked, start=1)
        ]
        timings["fusion"] = round(
            max(0.0, (time.perf_counter() - started) * 1000 - sum(timings.values())),
            3,
        )
        timings["total"] = round((time.perf_counter() - started) * 1000, 3)
        emit_trace(
            "retrieval",
            "retrieval.pipeline.fused",
            elapsed_ms=timings["total"],
            payload={
                "fusion_strategy": "rrf",
                "rrf_k": self.settings.rrf_k,
                "queries": queries,
                "candidates": [serialize_candidate(result) for result in results],
            },
        )
        return results, timings

    def _normalize_query(
        self,
        question: str | RetrievalQuery,
        filters: RetrievalFilters | None,
    ) -> QueryPlan:
        if isinstance(question, QueryPlan):
            return question
        if isinstance(question, RetrievalQuery):
            return QueryPlan(
                original_query=question.text,
                text=question.text,
                queries=[question.text],
                filters=question.filters,
                fusion_strategy="single_query",
                metadata={},
            )
        return self.query_processor.process(question, filters)
