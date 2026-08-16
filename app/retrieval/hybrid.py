from __future__ import annotations

import time

from app.models import RetrievalFilters, RetrievalOutcome, RetrievalQuery, SearchResult
from app.retrieval.base import Retriever
from app.tracing import emit_trace
from app.tracing.serializers import serialize_candidate


class HybridRetriever:
    def __init__(self, dense: Retriever, sparse: Retriever, *, rrf_k: int = 60) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    @property
    def component_id(self) -> str:
        return f"hybrid-rrf-{self.rrf_k}-v1"

    def retrieve(
        self,
        question: str | RetrievalQuery,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalOutcome:
        started = time.perf_counter()
        dense_outcome = self.dense.retrieve(question, top_k, filters)
        query = dense_outcome.query or (
            question if isinstance(question, RetrievalQuery) else RetrievalQuery(text=question)
        )
        sparse_outcome = self.sparse.retrieve(query, top_k)
        fused: dict[str, tuple[float, SearchResult]] = {}
        contributions: dict[str, dict[str, float | int]] = {}
        for source, outcome in (("dense", dense_outcome), ("bm25", sparse_outcome)):
            for result in outcome.results:
                score, selected = fused.get(result.chunk.chunk_id, (0.0, result))
                contribution = 1.0 / (self.rrf_k + result.rank)
                fused[result.chunk.chunk_id] = (
                    score + contribution,
                    selected,
                )
                contributions.setdefault(result.chunk.chunk_id, {}).update(
                    {f"{source}_rank": result.rank, f"{source}_contribution": contribution}
                )
        ranked = sorted(fused.values(), key=lambda item: item[0], reverse=True)[:top_k]
        results = [
            SearchResult(
                chunk=result.chunk,
                score=score,
                recall_score=score,
                rank=rank,
                retriever_id=self.component_id,
            )
            for rank, (score, result) in enumerate(ranked, start=1)
        ]
        total_ms = (time.perf_counter() - started) * 1000
        fused_items = []
        for result in results:
            item = serialize_candidate(result)
            item.update(contributions.get(result.chunk.chunk_id, {}))
            fused_items.append(item)
        emit_trace(
            "retrieval",
            "retrieval.fused",
            elapsed_ms=round(total_ms, 3),
            payload={"rrf_k": self.rrf_k, "candidates": fused_items},
        )
        return RetrievalOutcome(
            results=results,
            query=query,
            timings_ms={
                **{f"dense_{key}": value for key, value in dense_outcome.timings_ms.items()},
                **{f"bm25_{key}": value for key, value in sparse_outcome.timings_ms.items()},
                "total": round(total_ms, 3),
            },
        )
