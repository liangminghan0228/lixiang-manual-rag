from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.models import Answer, SearchResult


class EvidenceGroup(BaseModel):
    id: str
    acceptable_chunk_ids: list[str] = Field(default_factory=list)
    acceptable_topic_ids: list[str] = Field(default_factory=list)


class AnswerPoint(BaseModel):
    id: str
    text: str
    keywords: list[str] = Field(default_factory=list)
    required_evidence_groups: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    id: str
    question: str
    question_type: str = "single_chunk"
    answerable: bool = True
    gold_topic_ids: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    required_evidence_groups: list[EvidenceGroup] = Field(default_factory=list)
    answer_points: list[AnswerPoint] = Field(default_factory=list)
    label_status: str = "reviewed"

    def evidence_groups(self) -> list[EvidenceGroup]:
        if self.required_evidence_groups:
            return self.required_evidence_groups
        if self.gold_chunk_ids or self.gold_topic_ids:
            return [
                EvidenceGroup(
                    id="legacy-gold",
                    acceptable_chunk_ids=self.gold_chunk_ids,
                    acceptable_topic_ids=self.gold_topic_ids,
                )
            ]
        return []


class RetrievalCaseMetrics(BaseModel):
    case_id: str
    question_type: str
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    group_coverage_at_k: float
    all_groups_hit_at_k: float
    retrieved_chunk_ids: list[str]
    retrieved_topic_ids: list[str]


def _matches_group(result: SearchResult, group: EvidenceGroup) -> bool:
    return result.chunk.chunk_id in group.acceptable_chunk_ids or (
        result.chunk.topic_id in group.acceptable_topic_ids
    )


def evaluate_retrieval_case(
    case: EvaluationCase,
    results: list[SearchResult],
    k: int,
) -> RetrievalCaseMetrics:
    selected = results[:k]
    groups = case.evidence_groups()
    relevant = [any(_matches_group(result, group) for group in groups) for result in selected]
    hit_groups = {
        group.id for group in groups if any(_matches_group(result, group) for result in selected)
    }
    group_count = len(groups)
    recall = len(hit_groups) / group_count if group_count else 0.0
    first_rank = next((rank for rank, hit in enumerate(relevant, start=1) if hit), None)
    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
    # A group contributes gain only once. Otherwise several chunks from the same
    # chapter could make nDCG larger than 1 without improving evidence coverage.
    seen_groups: set[str] = set()
    dcg = 0.0
    for rank, result in enumerate(selected, start=1):
        new_groups = {
            group.id
            for group in groups
            if group.id not in seen_groups and _matches_group(result, group)
        }
        if new_groups:
            dcg += 1.0 / math.log2(rank + 1)
            seen_groups.update(new_groups)
    ideal_hits = min(group_count, k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return RetrievalCaseMetrics(
        case_id=case.id,
        question_type=case.question_type,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=dcg / ideal_dcg if ideal_dcg else 0.0,
        group_coverage_at_k=recall,
        all_groups_hit_at_k=float(bool(groups) and len(hit_groups) == group_count),
        retrieved_chunk_ids=[result.chunk.chunk_id for result in selected],
        retrieved_topic_ids=[result.chunk.topic_id for result in selected],
    )


def aggregate_retrieval_metrics(
    metrics: list[RetrievalCaseMetrics],
) -> dict[str, object]:
    if not metrics:
        return {"count": 0}

    def summarize(items: list[RetrievalCaseMetrics]) -> dict[str, float | int]:
        count = len(items)
        return {
            "count": count,
            "recall": sum(item.recall_at_k for item in items) / count,
            "mrr": sum(item.reciprocal_rank for item in items) / count,
            "ndcg": sum(item.ndcg_at_k for item in items) / count,
            "group_coverage": sum(item.group_coverage_at_k for item in items) / count,
            "all_groups_hit_rate": sum(item.all_groups_hit_at_k for item in items) / count,
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


def evaluate_answer_case(case: EvaluationCase, answer: Answer) -> dict[str, float | bool]:
    if not case.answerable:
        return {
            "refusal_correct": answer.refused,
            "answer_point_coverage": 1.0 if answer.refused else 0.0,
            "citation_correctness": 1.0 if answer.refused else 0.0,
        }
    normalized = answer.text.lower()
    covered = [
        point
        for point in case.answer_points
        if not point.keywords or any(keyword.lower() in normalized for keyword in point.keywords)
    ]
    valid_chunk_ids = {
        chunk_id for group in case.evidence_groups() for chunk_id in group.acceptable_chunk_ids
    }
    valid_topic_ids = {
        topic_id for group in case.evidence_groups() for topic_id in group.acceptable_topic_ids
    }
    evidence_by_chunk = {item.chunk.chunk_id: item.chunk for item in answer.evidence}
    supported_citations = 0
    for citation in answer.citations:
        evidence = evidence_by_chunk.get(citation.chunk_id)
        if citation.chunk_id in valid_chunk_ids or (
            evidence is not None and evidence.topic_id in valid_topic_ids
        ):
            supported_citations += 1
    return {
        "refusal_correct": not answer.refused,
        "answer_point_coverage": (
            len(covered) / len(case.answer_points)
            if case.answer_points
            else float(not answer.refused)
        ),
        "citation_correctness": (
            supported_citations / len(answer.citations) if answer.citations else 0.0
        ),
        "citation_format_valid": answer.citation_validated,
    }


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    true_positive_rate: float
    true_negative_rate: float
    balanced_accuracy: float


def suggest_min_score(
    labeled_scores: Iterable[tuple[bool, float | None]],
) -> ThresholdMetrics | None:
    """Select a refusal threshold using balanced answerability accuracy.

    This deliberately calibrates only the evidence/no-evidence gate. Retrieval
    Recall@K remains a separate metric because a high score is not proof that
    the expected topic was recalled.
    """

    samples = [(label, score) for label, score in labeled_scores if score is not None]
    positives = sum(label for label, _ in samples)
    negatives = len(samples) - positives
    if not samples or not positives or not negatives:
        return None

    scores = sorted({score for _, score in samples})
    epsilon = 1e-9
    candidates = [scores[0] - epsilon]
    candidates.extend((left + right) / 2 for left, right in zip(scores, scores[1:], strict=False))
    candidates.append(scores[-1] + epsilon)

    best: ThresholdMetrics | None = None
    for threshold in candidates:
        true_positives = sum(label and score >= threshold for label, score in samples)
        true_negatives = sum(not label and score < threshold for label, score in samples)
        true_positive_rate = true_positives / positives
        true_negative_rate = true_negatives / negatives
        balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
        metrics = ThresholdMetrics(
            threshold=threshold,
            true_positive_rate=true_positive_rate,
            true_negative_rate=true_negative_rate,
            balanced_accuracy=balanced_accuracy,
        )
        if best is None or (metrics.balanced_accuracy, metrics.threshold) > (
            best.balanced_accuracy,
            best.threshold,
        ):
            best = metrics
    return best
