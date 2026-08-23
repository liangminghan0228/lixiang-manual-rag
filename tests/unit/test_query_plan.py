from __future__ import annotations

import pytest

from app.models import QueryPlan, RetrievalFilters, RetrievalOutcome, RetrievalQuery, SearchResult
from app.retrieval.base import QueryProcessor
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query import IdentityQueryProcessor
from app.settings import RetrievalSettings


class CapturingRetriever:
    component_id = "capturing-retriever-v1"

    def __init__(self, chunk) -> None:
        self.chunk = chunk
        self.calls: list[tuple[str | QueryPlan | RetrievalQuery, int, RetrievalFilters | None]] = []

    def retrieve(
        self, question, top_k: int, filters: RetrievalFilters | None = None
    ) -> RetrievalOutcome:
        self.calls.append((question, top_k, filters))
        result = SearchResult(chunk=self.chunk, score=1.0, rank=1)
        return RetrievalOutcome(
            results=[result],
            timings_ms={"candidate": 1.0, "total": 1.0},
            query=question
            if isinstance(question, QueryPlan)
            else RetrievalQuery(text=str(question)),
        )


class CapturingReranker:
    component_id = "capturing-reranker-v1"
    is_loaded = True

    def __init__(self) -> None:
        self.calls: list[tuple[QueryPlan, list[SearchResult]]] = []

    def rerank(self, query: QueryPlan, candidates: list[SearchResult]) -> list[SearchResult]:
        self.calls.append((query, candidates))
        return candidates


def test_identity_query_processor_outputs_query_plan_filters_and_queries(sample_chunk) -> None:
    processor = IdentityQueryProcessor()
    filters = RetrievalFilters(topic_ids=["topic-safe"])

    plan = processor.process("  驾驶前    检查  车灯  ", filters)

    assert isinstance(plan, QueryPlan)
    assert plan.original_query == "驾驶前 检查 车灯"
    assert plan.text == "驾驶前 检查 车灯"
    assert plan.queries == ["驾驶前 检查 车灯"]
    assert plan.filters == filters
    assert plan.fusion_strategy == "single_query"
    assert plan.metadata["processor"] == "identity-v1"


def test_retrieval_pipeline_runs_single_query_baseline(sample_chunk) -> None:
    settings = RetrievalSettings(top_k=3, candidate_top_k=4)
    processor: QueryProcessor = IdentityQueryProcessor()
    retriever = CapturingRetriever(sample_chunk)
    reranker = CapturingReranker()
    pipeline = RetrievalPipeline(processor, retriever, reranker, settings)

    outcome = pipeline.retrieve("   驾驶前    检查车灯   ", top_k=2)

    assert outcome.query is not None
    assert isinstance(outcome.query, QueryPlan)
    assert outcome.query.text == "驾驶前 检查车灯"
    assert outcome.query.queries == ["驾驶前 检查车灯"]
    assert outcome.query.filters == RetrievalFilters()
    assert outcome.results[0].chunk.chunk_id == sample_chunk.chunk_id
    assert len(reranker.calls) == 1
    assert retriever.calls[0][0] == "驾驶前 检查车灯"
    assert retriever.calls[0][1] == settings.candidate_top_k


def test_retrieval_pipeline_keeps_backward_compatible_retrieval_query(sample_chunk) -> None:
    settings = RetrievalSettings(top_k=2, candidate_top_k=2, evidence_top_k=2)
    processor: QueryProcessor = IdentityQueryProcessor()
    retriever = CapturingRetriever(sample_chunk)
    reranker = CapturingReranker()
    plan_query = RetrievalQuery(
        text="旧接口查询",
        filters=RetrievalFilters(topic_ids=["topic-safe"]),
    )
    pipeline = RetrievalPipeline(processor, retriever, reranker, settings)

    outcome = pipeline.retrieve(plan_query, top_k=2)

    assert isinstance(outcome.query, QueryPlan)
    assert outcome.query.text == "旧接口查询"
    assert outcome.query.queries == ["旧接口查询"]
    assert outcome.query.filters == plan_query.filters
    assert retriever.calls[0][0] == "旧接口查询"


def test_retrieval_pipeline_marks_multi_query_strategy_not_implemented(sample_chunk) -> None:
    settings = RetrievalSettings(top_k=3, candidate_top_k=4)
    processor = IdentityQueryProcessor()
    retriever = CapturingRetriever(sample_chunk)
    reranker = CapturingReranker()
    pipeline = RetrievalPipeline(processor, retriever, reranker, settings)

    plan = QueryPlan(
        original_query="驾驶前 检查车灯",
        text="驾驶前 检查车灯",
        queries=["驾驶前 检查车灯", "车辆 周边 检查"],
        filters=RetrievalFilters(),
        fusion_strategy="multi_query",
        metadata={},
    )

    with pytest.raises(NotImplementedError, match="Multi-query fusion"):
        pipeline.retrieve(plan, top_k=2)
