from __future__ import annotations

import pytest

from app.generation.mock import MockGenerator
from app.generation.service import REFUSAL_TEXT, ChatService
from app.models import RetrievalOutcome, SearchResult
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import DeterministicHashEmbedder
from app.retrieval.vector_store import InMemoryVectorStore
from app.settings import RetrievalSettings


class RepairingGenerator:
    component_id = "repairing-test"

    async def generate(self, question: str, evidence: list[SearchResult]) -> str:
        del question, evidence
        return "驾驶前应检查车辆。"

    async def repair_citations(
        self,
        question: str,
        answer: str,
        evidence: list[SearchResult],
    ) -> str:
        del question, answer, evidence
        return "驾驶前应检查车灯和车辆周边。[1]"


def test_inmemory_upsert_is_idempotent_and_retrievable(sample_chunk) -> None:
    embedder = DeterministicHashEmbedder(64)
    store = InMemoryVectorStore()
    vector = embedder.embed_documents([sample_chunk.text])[0]

    store.upsert([sample_chunk], [vector])
    store.upsert([sample_chunk], [vector])
    outcome = DenseRetriever(embedder, store).retrieve("驾驶前检查车灯", top_k=5)

    assert store.count() == 1
    assert outcome.results[0].chunk.chunk_id == sample_chunk.chunk_id
    assert outcome.results[0].rank == 1
    assert "embedding" in outcome.timings_ms


@pytest.mark.asyncio
async def test_chat_service_returns_citations(sample_chunk) -> None:
    embedder = DeterministicHashEmbedder(64)
    store = InMemoryVectorStore()
    store.upsert([sample_chunk], embedder.embed_documents([sample_chunk.text]))
    service = ChatService(
        DenseRetriever(embedder, store),
        MockGenerator(),
        RetrievalSettings(top_k=5, evidence_top_k=1),
    )

    answer = await service.answer("驾驶前需要检查什么？")

    assert not answer.refused
    assert answer.citations[0].chunk_id == sample_chunk.chunk_id
    assert "[1]" in answer.text
    assert answer.evidence


@pytest.mark.asyncio
async def test_chat_service_refuses_without_evidence() -> None:
    class EmptyRetriever:
        component_id = "empty"

        def retrieve(self, question, top_k: int, filters=None) -> RetrievalOutcome:
            del question, top_k, filters
            return RetrievalOutcome(results=[], timings_ms={"total": 0.0})

    service = ChatService(
        EmptyRetriever(),
        MockGenerator(),
        RetrievalSettings(top_k=5, evidence_top_k=3),
    )
    answer = await service.answer("不存在的问题")

    assert answer.refused
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []


@pytest.mark.asyncio
async def test_chat_service_repairs_missing_inline_citation(sample_chunk) -> None:
    embedder = DeterministicHashEmbedder(64)
    store = InMemoryVectorStore()
    store.upsert([sample_chunk], embedder.embed_documents([sample_chunk.text]))
    service = ChatService(
        DenseRetriever(embedder, store),
        RepairingGenerator(),
        RetrievalSettings(top_k=5, evidence_top_k=1),
    )

    answer = await service.answer("驾驶前需要检查什么？")

    assert answer.citation_validated
    assert answer.citations[0].index == 1
    assert answer.citations[0].excerpt
