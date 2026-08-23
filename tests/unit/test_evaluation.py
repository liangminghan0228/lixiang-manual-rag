from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.evaluation import (
    EvaluationCase,
    RagasJudge,
    aggregate_generation_metrics,
    evaluate_retrieval_case,
)
from app.models import Answer, SearchResult
from scripts.build_eval_dataset import QUESTION_COUNTS


class FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[dict[str, Any]] = []

    async def ascore(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


def test_retrieval_f1_and_mrr_use_gold_chunk_ids(sample_chunk) -> None:
    second = sample_chunk.model_copy(
        update={"chunk_id": "gold-2", "topic_id": "topic-2", "content_hash": "hash-2"}
    )
    irrelevant = sample_chunk.model_copy(
        update={"chunk_id": "other", "topic_id": "topic-3", "content_hash": "hash-3"}
    )
    case = EvaluationCase(
        id="multi",
        user_input="两方面分别是什么？",
        question_type="multi_topic",
        reference="参考答案",
        reference_contexts=[sample_chunk.text, second.text],
        gold_chunk_ids=[sample_chunk.chunk_id, second.chunk_id],
    )
    results = [
        SearchResult(chunk=sample_chunk, score=1, rank=1),
        SearchResult(chunk=irrelevant, score=0.9, rank=2),
        SearchResult(chunk=second, score=0.8, rank=3),
    ]

    metrics = evaluate_retrieval_case(case, results, f1_k=3, mrr_k=10)

    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.recall_at_k == 1
    assert metrics.f1_at_k == pytest.approx(0.8)
    assert metrics.reciprocal_rank == 1


@pytest.mark.asyncio
async def test_ragas_judge_maps_generation_metrics(sample_chunk) -> None:
    faithfulness = FakeMetric(0.9)
    relevancy = FakeMetric(0.8)
    completeness = FakeMetric(0.7)
    judge = RagasJudge(faithfulness, relevancy, completeness)
    result = SearchResult(chunk=sample_chunk, score=1, rank=1)
    case = EvaluationCase(
        id="single",
        user_input="如何操作？",
        question_type="single_chunk",
        reference="参考答案",
        reference_contexts=[sample_chunk.text],
        gold_chunk_ids=[sample_chunk.chunk_id],
    )
    answer = Answer(
        text="按说明操作。[1]",
        citations=[],
        evidence=[result],
        timings_ms={},
        citation_validated=True,
    )

    sample, metrics = await judge.score(case, answer)

    assert sample.user_input == case.user_input
    assert sample.retrieved_context_ids == [sample_chunk.chunk_id]
    assert metrics.faithfulness == 0.9
    assert metrics.answer_relevancy == 0.8
    assert metrics.completeness == 0.7
    assert len(faithfulness.calls) == len(relevancy.calls) == len(completeness.calls) == 1


@pytest.mark.asyncio
async def test_unanswerable_case_only_scores_refusal() -> None:
    metrics = [FakeMetric(1), FakeMetric(1), FakeMetric(1)]
    judge = RagasJudge(*metrics)
    case = EvaluationCase(
        id="no-answer",
        user_input="未来价格是多少？",
        question_type="unanswerable",
        answerable=False,
        reference="知识库中未找到足够依据。",
    )
    answer = Answer(
        text="知识库中未找到足够依据。",
        citations=[],
        evidence=[],
        timings_ms={},
        refused=True,
        citation_validated=True,
    )

    _, result = await judge.score(case, answer)

    assert result.refusal_correct is True
    assert not any(metric.calls for metric in metrics)
    assert aggregate_generation_metrics([result])["refusal_accuracy"] == 1


@pytest.mark.asyncio
async def test_answerable_refusal_is_scored_as_generation_failure() -> None:
    metrics = [FakeMetric(1), FakeMetric(1), FakeMetric(1)]
    judge = RagasJudge(*metrics)
    case = EvaluationCase(
        id="missed-answer",
        user_input="如何操作？",
        question_type="single_chunk",
        reference="参考答案",
        reference_contexts=["手册中的操作说明"],
        gold_chunk_ids=["gold-1"],
    )
    answer = Answer(
        text="知识库中未找到足够依据。",
        citations=[],
        evidence=[],
        timings_ms={},
        refused=True,
        citation_validated=True,
    )

    _, result = await judge.score(case, answer)

    assert result.faithfulness == 0
    assert result.answer_relevancy == 0
    assert result.completeness == 0
    assert not any(metric.calls for metric in metrics)


def test_single_dataset_contract_contains_fifty_cases() -> None:
    assert QUESTION_COUNTS == {
        "single_chunk": 18,
        "multi_chunk_same_topic": 8,
        "multi_topic": 10,
        "cross_manual": 6,
        "unanswerable": 8,
    }
    assert sum(QUESTION_COUNTS.values()) == 50
