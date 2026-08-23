from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, model_validator
from ragas.cache import DiskCacheBackend
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, FactualCorrectness, Faithfulness

from app.models import Answer, RetrievalFilters, SearchResult
from app.retrieval.embedder import Embedder

RETRIEVAL_F1_K = 5
RETRIEVAL_MRR_K = 10


class EvaluationCase(BaseModel):
    id: str
    user_input: str = Field(min_length=1)
    question_type: str
    answerable: bool = True
    reference: str = Field(min_length=1)
    reference_contexts: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    tags: list[str] = Field(default_factory=list)
    label_status: str = "reference_review_required"

    @model_validator(mode="after")
    def validate_gold_evidence(self) -> EvaluationCase:
        if self.answerable and (not self.gold_chunk_ids or not self.reference_contexts):
            raise ValueError("answerable cases require gold_chunk_ids and reference_contexts")
        if not self.answerable and (self.gold_chunk_ids or self.reference_contexts):
            raise ValueError("unanswerable cases must not contain gold evidence")
        return self


class RetrievalCaseMetrics(BaseModel):
    case_id: str
    question_type: str
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    reciprocal_rank: float
    retrieved_chunk_ids: list[str]


class GenerationCaseMetrics(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    completeness: float | None = None
    refusal_correct: bool | None = None


def evaluate_retrieval_case(
    case: EvaluationCase,
    results: list[SearchResult],
    *,
    f1_k: int = RETRIEVAL_F1_K,
    mrr_k: int = RETRIEVAL_MRR_K,
) -> RetrievalCaseMetrics:
    if not case.answerable:
        raise ValueError("retrieval F1 and MRR are only defined for answerable cases")
    gold = set(case.gold_chunk_ids)
    selected = results[:f1_k]
    retrieved = [result.chunk.chunk_id for result in selected]
    hit_count = len(set(retrieved) & gold)
    precision = hit_count / len(retrieved) if retrieved else 0.0
    recall = hit_count / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    first_rank = next(
        (
            rank
            for rank, result in enumerate(results[:mrr_k], start=1)
            if result.chunk.chunk_id in gold
        ),
        None,
    )
    return RetrievalCaseMetrics(
        case_id=case.id,
        question_type=case.question_type,
        precision_at_k=precision,
        recall_at_k=recall,
        f1_at_k=f1,
        reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
        retrieved_chunk_ids=[result.chunk.chunk_id for result in results[:mrr_k]],
    )


def aggregate_retrieval_metrics(
    metrics: list[RetrievalCaseMetrics],
) -> dict[str, object]:
    if not metrics:
        return {"overall": {"count": 0}, "by_question_type": {}}

    def summarize(items: list[RetrievalCaseMetrics]) -> dict[str, float | int]:
        count = len(items)
        return {
            "count": count,
            "precision_at_5": sum(item.precision_at_k for item in items) / count,
            "recall_at_5": sum(item.recall_at_k for item in items) / count,
            "f1_at_5": sum(item.f1_at_k for item in items) / count,
            "mrr_at_10": sum(item.reciprocal_rank for item in items) / count,
        }

    grouped: dict[str, list[RetrievalCaseMetrics]] = {}
    for item in metrics:
        grouped.setdefault(item.question_type, []).append(item)
    return {
        "overall": summarize(metrics),
        "by_question_type": {
            question_type: summarize(items) for question_type, items in sorted(grouped.items())
        },
    }


def aggregate_generation_metrics(
    metrics: Iterable[GenerationCaseMetrics],
) -> dict[str, float | int | None]:
    items = list(metrics)

    def average(name: str) -> float | None:
        values = [getattr(item, name) for item in items if getattr(item, name) is not None]
        return sum(values) / len(values) if values else None

    refusal_values = [item.refusal_correct for item in items if item.refusal_correct is not None]
    return {
        "answerable_count": sum(item.faithfulness is not None for item in items),
        "unanswerable_count": len(refusal_values),
        "faithfulness": average("faithfulness"),
        "answer_relevancy": average("answer_relevancy"),
        "completeness": average("completeness"),
        "refusal_accuracy": (
            sum(bool(value) for value in refusal_values) / len(refusal_values)
            if refusal_values
            else None
        ),
    }


class LocalRagasEmbedding(BaseRagasEmbedding):
    """Expose the project's Embedder through Ragas' embedding interface."""

    def __init__(self, embedder: Embedder) -> None:
        super().__init__()
        self.embedder = embedder

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        del kwargs
        return self.embedder.embed_query(text)

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        del kwargs
        return await asyncio.to_thread(self.embedder.embed_query, text)

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        del kwargs
        return self.embedder.embed_documents(texts)

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        del kwargs
        return await asyncio.to_thread(self.embedder.embed_documents, texts)


class AsyncMetric(Protocol):
    async def ascore(self, **kwargs: Any) -> Any: ...


class RagasJudge:
    def __init__(
        self,
        faithfulness: AsyncMetric,
        answer_relevancy: AsyncMetric,
        completeness: AsyncMetric,
    ) -> None:
        self.faithfulness = faithfulness
        self.answer_relevancy = answer_relevancy
        self.completeness = completeness

    @staticmethod
    def _score_value(result: Any) -> float | None:
        value = float(result.value)
        return value if math.isfinite(value) else None

    async def score(
        self,
        case: EvaluationCase,
        answer: Answer,
    ) -> tuple[SingleTurnSample, GenerationCaseMetrics]:
        sample = SingleTurnSample(
            user_input=case.user_input,
            response=answer.text,
            retrieved_contexts=[item.chunk.text for item in answer.evidence],
            retrieved_context_ids=[item.chunk.chunk_id for item in answer.evidence],
            reference=case.reference,
            reference_contexts=case.reference_contexts,
            reference_context_ids=case.gold_chunk_ids,
        )
        if not case.answerable:
            return sample, GenerationCaseMetrics(refusal_correct=answer.refused)
        if answer.refused or not answer.evidence:
            return sample, GenerationCaseMetrics(
                faithfulness=0.0,
                answer_relevancy=0.0,
                completeness=0.0,
            )
        faithfulness = await self.faithfulness.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )
        relevancy = await self.answer_relevancy.ascore(
            user_input=sample.user_input,
            response=sample.response,
        )
        completeness = await self.completeness.ascore(
            response=sample.response,
            reference=sample.reference,
        )
        return sample, GenerationCaseMetrics(
            faithfulness=self._score_value(faithfulness),
            answer_relevancy=self._score_value(relevancy),
            completeness=self._score_value(completeness),
        )


def build_ragas_judge(
    *,
    model: str,
    api_key: str,
    base_url: str,
    embedder: Embedder,
    cache_dir: Path,
    relevancy_strictness: int = 1,
) -> RagasJudge:
    cache = DiskCacheBackend(str(cache_dir))
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    evaluator_llm = llm_factory(
        model,
        provider="openai",
        client=client,
        adapter="instructor",
        cache=cache,
        temperature=0,
    )
    embeddings = LocalRagasEmbedding(embedder)
    return RagasJudge(
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=embeddings,
            strictness=relevancy_strictness,
        ),
        FactualCorrectness(llm=evaluator_llm, mode="recall"),
    )
