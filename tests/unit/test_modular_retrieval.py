from __future__ import annotations

from app.models import RetrievalFilters, RetrievalQuery, SearchResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.evidence import DiversifiedEvidenceSelector
from app.retrieval.vector_store import InMemoryVectorStore


def test_bm25_and_payload_filters(sample_chunk) -> None:
    other = sample_chunk.model_copy(
        update={
            "chunk_id": "7017144b-dad1-4b42-bb92-cf55beb2d6b9",
            "topic_id": "topic-other",
            "snapshot_id": "other-snapshot",
            "text": "车辆保险费用和成交价格",
            "content_hash": "other",
        }
    )
    store = InMemoryVectorStore()
    store.upsert([sample_chunk, other], [[1.0, 0.0], [0.0, 1.0]])

    outcome = BM25Retriever(store).retrieve(
        "驾驶前检查车灯",
        5,
        RetrievalFilters(snapshot_id=sample_chunk.snapshot_id),
    )

    assert [item.chunk.chunk_id for item in outcome.results] == [sample_chunk.chunk_id]
    assert outcome.results[0].retriever_id.startswith("bm25-")
    store.delete_chunks([other.chunk_id])
    assert store.count() == 1


def test_evidence_selector_deduplicates_and_diversifies(sample_chunk) -> None:
    same_text = sample_chunk.model_copy(update={"chunk_id": "a1a99e40-0e41-424e-8e3b-14e1d4056d22"})
    other_topic = sample_chunk.model_copy(
        update={
            "chunk_id": "9ff18eea-e2bb-4a3d-bf87-1abcb0683655",
            "topic_id": "topic-other",
            "content_hash": "other",
        }
    )
    candidates = [
        SearchResult(chunk=sample_chunk, score=0.9, rank=1),
        SearchResult(chunk=same_text, score=0.8, rank=2),
        SearchResult(chunk=other_topic, score=0.7, rank=3),
    ]
    selector = DiversifiedEvidenceSelector(top_k=3, min_score=0.5, per_topic_limit=1)

    bundle = selector.select(RetrievalQuery(text="检查"), candidates)

    assert [item.chunk.chunk_id for item in bundle.items] == [
        sample_chunk.chunk_id,
        other_topic.chunk_id,
    ]
