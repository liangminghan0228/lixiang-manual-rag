from __future__ import annotations

from app.generation.mock import MockGenerator
from app.rag.agentic_rag import AgenticRagStrategy
from app.rag.controller import AgentAction
from app.rag.graph_rag import GraphRagStrategy
from app.rag.self_rag import SelfRagStrategy
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import DeterministicHashEmbedder
from app.retrieval.vector_store import InMemoryVectorStore
from app.settings import RetrievalSettings


class SelfController:
    def __init__(self, support: bool = True) -> None:
        self.retrieval_checks = 0
        self.rewrites = 0
        self.support = support

    async def assess_evidence(self, question, outcome):
        self.retrieval_checks += 1
        return {"sufficient": self.retrieval_checks > 1}

    async def rewrite_query(self, question, outcome):
        self.rewrites += 1
        return "车灯检查"

    async def assess_support(self, question, draft, evidence):
        return {"supported": self.support}


def settings() -> RetrievalSettings:
    return RetrievalSettings(top_k=3, candidate_top_k=3, evidence_top_k=2)


def dense(sample_chunk):
    embedder = DeterministicHashEmbedder(32)
    store = InMemoryVectorStore()
    store.upsert([sample_chunk], embedder.embed_documents([sample_chunk.text]))
    return DenseRetriever(embedder, store), store


async def test_self_rag_rewrites_at_most_once_and_checks_support(sample_chunk):
    retriever, _ = dense(sample_chunk)
    controller = SelfController()
    strategy = SelfRagStrategy(retriever, MockGenerator(), settings(), controller=controller)

    answer = await strategy.answer("如何准备驾驶？")

    assert not answer.refused
    assert controller.rewrites == 1
    assert controller.retrieval_checks == 1


async def test_self_rag_refuses_when_support_check_fails(sample_chunk):
    retriever, _ = dense(sample_chunk)
    strategy = SelfRagStrategy(
        retriever,
        MockGenerator(),
        settings(),
        controller=SelfController(support=False),
    )

    answer = await strategy.answer("如何准备驾驶？")

    assert answer.refused
    assert answer.citations == []


async def test_self_rag_keeps_both_retrieval_rounds_and_controller_timings(sample_chunk):
    retriever, _ = dense(sample_chunk)
    strategy = SelfRagStrategy(
        retriever,
        MockGenerator(),
        settings(),
        controller=SelfController(),
    )

    outcome = await strategy.retrieve("如何准备驾驶？")

    assert "self_rag_retrieval_initial" in outcome.timings_ms
    assert "self_rag_retrieval_rewritten" in outcome.timings_ms
    assert "self_rag_controller" in outcome.timings_ms
    assert outcome.timings_ms["total"] >= outcome.timings_ms["self_rag_retrieval"]


class AgentController:
    def __init__(self):
        self.actions = iter(
            [
                AgentAction("retrieve", query="车灯"),
                AgentAction("retrieve", query="车辆周边"),
                AgentAction("answer"),
            ]
        )

    async def next_action(self, question, candidates, steps):
        return next(self.actions)


async def test_agentic_rag_merges_retrieval_and_uses_vanilla_answer(sample_chunk):
    retriever, _ = dense(sample_chunk)
    strategy = AgenticRagStrategy(
        retriever,
        MockGenerator(),
        settings(),
        controller=AgentController(),
        max_steps=3,
    )

    outcome = await strategy.retrieve("驾驶准备")
    answer = await strategy.answer_from_outcome("驾驶准备", outcome)

    assert outcome.results
    assert not answer.refused
    assert answer.citation_validated


class LoopController:
    async def next_action(self, question, candidates, steps):
        return {"action": "retrieve", "query": question}


async def test_agentic_rag_has_hard_step_bound(sample_chunk):
    retriever, _ = dense(sample_chunk)
    strategy = AgenticRagStrategy(
        retriever,
        MockGenerator(),
        settings(),
        controller=LoopController(),
        max_steps=2,
    )

    outcome = await strategy.retrieve("驾驶准备")

    assert outcome.timings_ms["agentic_steps"] == 2


async def test_graph_rag_expands_same_topic_one_hop(sample_chunk):
    neighbor = sample_chunk.model_copy(
        update={
            "chunk_id": "neighbor",
            "title": "行车注意事项",
            "section_path": ["安全驾驶", "行车注意事项"],
            "text": "正文：请注意道路情况。",
            "content_hash": "neighbor-hash",
        }
    )
    _, store = dense(sample_chunk)
    embedder = DeterministicHashEmbedder(32)
    store.upsert([neighbor], embedder.embed_documents([neighbor.text]))
    strategy = GraphRagStrategy(
        BM25Retriever(store),
        MockGenerator(),
        settings(),
        vector_store=store,
    )

    outcome = await strategy.retrieve("车灯")

    assert outcome.query.text == "车灯"
    assert {item.chunk.chunk_id for item in outcome.results} == {sample_chunk.chunk_id, "neighbor"}
    assert all(item.retriever_id == "graph-document-structure-v1" for item in outcome.results)


def test_graph_rag_does_not_connect_chunks_across_manual_versions(sample_chunk):
    other_version = sample_chunk.model_copy(
        update={
            "chunk_id": "other-version",
            "snapshot_id": "new-snapshot",
            "content_hash": "other-version-hash",
        }
    )

    graph = GraphRagStrategy._build_graph([sample_chunk, other_version])

    assert graph[sample_chunk.chunk_id] == set()
    assert graph[other_version.chunk_id] == set()


def test_graph_rag_builds_bounded_adjacent_graph(sample_chunk):
    chunks = [
        sample_chunk.model_copy(
            update={
                "chunk_id": f"chunk-{index}",
                "document_id": f"TEST:test-snapshot:topic-safe:{index}",
                "content_hash": f"hash-{index}",
            }
        )
        for index in range(100)
    ]

    graph = GraphRagStrategy._build_graph(chunks)

    assert sum(len(neighbors) for neighbors in graph.values()) == 2 * (len(chunks) - 1)
    assert max(map(len, graph.values())) <= 2
    assert graph["chunk-0"] == {"chunk-1"}
    assert graph["chunk-99"] == {"chunk-98"}
