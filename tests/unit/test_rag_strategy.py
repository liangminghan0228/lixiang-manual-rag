from __future__ import annotations

import pytest

from app.generation.mock import MockGenerator
from app.models import RetrievalOutcome, SearchResult
from app.rag.vanilla import VanillaRagStrategy
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


@pytest.mark.asyncio
async def test_vanilla_strategy_retrieve_defaults_to_settings_top_k() -> None:
    class CaptureRetriever:
        component_id = "capture"

        def __init__(self) -> None:
            self.last_top_k: int | None = None

        def retrieve(self, question, top_k: int, filters=None) -> RetrievalOutcome:
            del question, filters
            self.last_top_k = top_k
            return RetrievalOutcome(results=[], timings_ms={"total": 0.0})

    retriever = CaptureRetriever()
    strategy = VanillaRagStrategy(
        retriever,
        MockGenerator(),
        RetrievalSettings(top_k=5, evidence_top_k=1),
    )

    await strategy.retrieve("q")
    assert retriever.last_top_k == 5


@pytest.mark.asyncio
async def test_vanilla_strategy_returns_citations(sample_chunk) -> None:
    embedder = DeterministicHashEmbedder(64)
    store = InMemoryVectorStore()
    store.upsert([sample_chunk], embedder.embed_documents([sample_chunk.text]))
    strategy = VanillaRagStrategy(
        DenseRetriever(embedder, store),
        MockGenerator(),
        RetrievalSettings(top_k=5, evidence_top_k=1),
    )

    answer = await strategy.answer("驾驶前需要检查什么？")

    assert not answer.refused
    assert answer.citations[0].chunk_id == sample_chunk.chunk_id
    assert "[1]" in answer.text
    assert answer.evidence


@pytest.mark.asyncio
async def test_vanilla_strategy_refuses_without_evidence() -> None:
    class EmptyRetriever:
        component_id = "empty"

        def retrieve(self, question, top_k: int, filters=None) -> RetrievalOutcome:
            del question, top_k, filters
            return RetrievalOutcome(results=[], timings_ms={"total": 0.0})

    strategy = VanillaRagStrategy(
        EmptyRetriever(),
        MockGenerator(),
        RetrievalSettings(top_k=5, evidence_top_k=3),
    )
    answer = await strategy.answer("不存在的问题")

    assert answer.refused
    assert answer.text == "知识库中未找到足够依据。"
    assert answer.citations == []


@pytest.mark.asyncio
async def test_vanilla_strategy_repairs_missing_inline_citation(sample_chunk) -> None:
    embedder = DeterministicHashEmbedder(64)
    store = InMemoryVectorStore()
    store.upsert([sample_chunk], embedder.embed_documents([sample_chunk.text]))
    strategy = VanillaRagStrategy(
        DenseRetriever(embedder, store),
        RepairingGenerator(),
        RetrievalSettings(top_k=5, evidence_top_k=1),
    )

    answer = await strategy.answer("驾驶前需要检查什么？")

    assert answer.citation_validated
    assert answer.citations[0].index == 1
    assert answer.citations[0].excerpt
