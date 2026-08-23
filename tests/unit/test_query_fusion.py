from __future__ import annotations

import pytest

from app.models import QueryPlan, RetrievalFilters, RetrievalOutcome, SearchResult
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query import IdentityQueryProcessor
from app.settings import RetrievalSettings


class QueryRetriever:
    component_id = "fake-retriever-v1"

    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def retrieve(self, question, top_k, filters=None):
        self.calls.append((question, top_k, filters))
        results = self.results_by_query[question]
        return RetrievalOutcome(results=results, timings_ms={"total": 1.0})


class IdentityReranker:
    component_id = "identity-reranker-v1"
    is_loaded = True

    def rerank(self, query, candidates):
        return candidates


def _result(chunk, rank, source="source-a"):
    return SearchResult(chunk=chunk, score=1.0 / rank, rank=rank, retriever_id=source)


def _pipeline(retriever):
    settings = RetrievalSettings(top_k=3, candidate_top_k=3, rrf_k=60)
    return RetrievalPipeline(IdentityQueryProcessor(), retriever, IdentityReranker(), settings)


def test_rrf_fuses_duplicate_chunks_and_keeps_stable_rank(sample_chunk):
    second = sample_chunk.model_copy(update={"chunk_id": "second"})
    third = sample_chunk.model_copy(update={"chunk_id": "third"})
    retriever = QueryRetriever(
        {
            "q1": [_result(sample_chunk, 1), _result(second, 2)],
            "q2": [_result(second, 1), _result(third, 2)],
        }
    )
    plan = QueryPlan(
        original_query="q1 q2",
        text="q1",
        queries=["q1", "q2"],
        fusion_strategy="rrf",
        metadata={"planner_elapsed_ms": 50.0},
    )

    outcome = _pipeline(retriever).retrieve(plan, top_k=3)

    assert [item.chunk.chunk_id for item in outcome.results] == [
        "second",
        sample_chunk.chunk_id,
        "third",
    ]
    assert outcome.results[0].recall_score == outcome.results[0].score
    assert [item.rank for item in outcome.results] == [1, 2, 3]
    assert [call[0] for call in retriever.calls] == ["q1", "q2"]
    assert outcome.timings_ms["query_planner"] == 50.0
    assert outcome.timings_ms["total"] >= 52.0


def test_rrf_passes_filters_to_each_query(sample_chunk):
    retriever = QueryRetriever({"q1": [_result(sample_chunk, 1)], "q2": [_result(sample_chunk, 1)]})
    filters = RetrievalFilters(topic_ids=["topic-safe"])
    plan = QueryPlan(
        original_query="q1 q2",
        text="q1",
        queries=["q1", "q2"],
        filters=filters,
        fusion_strategy="rrf",
    )

    _pipeline(retriever).retrieve(plan, top_k=1)

    assert all(call[2] == filters for call in retriever.calls)


def test_single_query_rejects_multiple_queries_instead_of_ignoring_them(sample_chunk):
    retriever = QueryRetriever({"q1": [_result(sample_chunk, 1)]})
    plan = QueryPlan(
        original_query="q1 q2", text="q1", queries=["q1", "q2"], fusion_strategy="single_query"
    )

    with pytest.raises(ValueError, match="at most one"):
        _pipeline(retriever).retrieve(plan, top_k=1)
    assert retriever.calls == []


def test_invalid_strategy_and_empty_query_fail_explicitly(sample_chunk):
    retriever = QueryRetriever({})
    pipeline = _pipeline(retriever)
    invalid = QueryPlan(original_query="q", text="q", queries=["q"], fusion_strategy="unknown")
    empty = QueryPlan(original_query="", text="", queries=["  "], fusion_strategy="rrf")

    with pytest.raises(ValueError, match="unsupported fusion strategy"):
        pipeline.retrieve(invalid, top_k=1)
    with pytest.raises(ValueError, match="at least one"):
        pipeline.retrieve(empty, top_k=1)
