from __future__ import annotations

from app.evaluation import EvaluationCase, EvidenceGroup, evaluate_retrieval_case
from app.models import SearchResult


def test_multi_evidence_metrics_count_each_group_once(sample_chunk) -> None:
    second = sample_chunk.model_copy(
        update={
            "chunk_id": "6b6d802a-771c-44ba-a827-aaa0d841cb54",
            "topic_id": "topic-other",
            "content_hash": "other-hash",
        }
    )
    duplicate = sample_chunk.model_copy(
        update={
            "chunk_id": "e510f979-d8ad-44c9-aef2-70b0a5c3b395",
            "content_hash": "duplicate-hash",
        }
    )
    case = EvaluationCase(
        id="multi",
        question="两方面分别是什么？",
        question_type="multi_topic",
        required_evidence_groups=[
            EvidenceGroup(id="g1", acceptable_topic_ids=[sample_chunk.topic_id]),
            EvidenceGroup(id="g2", acceptable_chunk_ids=[second.chunk_id]),
        ],
    )
    results = [
        SearchResult(chunk=sample_chunk, score=1, rank=1),
        SearchResult(chunk=duplicate, score=0.9, rank=2),
        SearchResult(chunk=second, score=0.8, rank=3),
    ]

    metrics = evaluate_retrieval_case(case, results, k=3)

    assert metrics.recall_at_k == 1
    assert metrics.all_groups_hit_at_k == 1
    assert 0 < metrics.ndcg_at_k <= 1
    assert metrics.reciprocal_rank == 1


def test_full_dataset_has_expected_strata() -> None:
    path = "data/eval/full_v1.jsonl"
    cases = [
        EvaluationCase.model_validate_json(line)
        for line in open(path, encoding="utf-8")  # noqa: PTH123, SIM115
        if line.strip()
    ]
    counts = {
        kind: sum(case.question_type == kind for case in cases)
        for kind in {case.question_type for case in cases}
    }
    assert len(cases) == 100
    assert counts == {
        "single_chunk": 55,
        "multi_chunk_same_topic": 20,
        "multi_topic": 10,
        "unanswerable": 15,
    }
    assert all(case.evidence_groups() for case in cases if case.answerable)
