from __future__ import annotations

from dataclasses import dataclass

from app.generation.base import Generator
from app.ingestion.chunker import Chunker
from app.rag.base import RagStrategy
from app.registry import (
    CHUNKERS,
    EMBEDDERS,
    EVIDENCE_SELECTORS,
    GENERATORS,
    QUERY_PROCESSORS,
    RAG_STRATEGIES,
    RERANKERS,
    RETRIEVERS,
    VECTOR_STORES,
    require_factory,
)
from app.retrieval.base import QueryProcessor, Reranker, Retriever
from app.retrieval.embedder import Embedder
from app.retrieval.evidence import EvidenceSelector
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.vector_store import VectorStore
from app.settings import AppSettings, RuntimeEnvironment


@dataclass(frozen=True)
class RagBuildContext:
    settings: AppSettings
    runtime: RuntimeEnvironment
    retriever: Retriever
    generator: Generator
    evidence_selector: EvidenceSelector
    vector_store: VectorStore


def build_chunker(settings: AppSettings) -> Chunker:
    chunker_factory = require_factory(CHUNKERS, settings.data.chunker, "chunker")
    return chunker_factory(settings.data.target_chars, settings.data.overlap_chars)


def build_embedder(settings: AppSettings) -> Embedder:
    embedder_factory = require_factory(EMBEDDERS, settings.embedding.provider, "embedder")
    return embedder_factory(settings.embedding)


def build_vector_store(settings: AppSettings) -> VectorStore:
    vector_store_factory = require_factory(
        VECTOR_STORES,
        settings.vector_store.provider,
        "vector store",
    )
    return vector_store_factory(settings.vector_store)


def build_candidate_retriever(
    settings: AppSettings, embedder: Embedder, vector_store: VectorStore
) -> Retriever:
    candidate_retriever_factory = require_factory(
        RETRIEVERS,
        settings.retrieval.provider,
        "retriever",
    )
    return candidate_retriever_factory(settings.retrieval, embedder, vector_store)


def build_query_processor(
    settings: AppSettings,
    runtime: RuntimeEnvironment,
) -> QueryProcessor:
    processor_settings = settings.retrieval.query_processor
    query_processor_factory = require_factory(
        QUERY_PROCESSORS,
        processor_settings.provider,
        "query processor",
    )
    return query_processor_factory(processor_settings, runtime)


def build_reranker(settings: AppSettings) -> Reranker:
    reranker_factory = require_factory(
        RERANKERS,
        settings.retrieval.reranker.provider,
        "reranker",
    )
    return reranker_factory(settings.retrieval.reranker)


def build_retriever(
    settings: AppSettings,
    query_processor: QueryProcessor,
    candidate_retriever: Retriever,
    reranker: Reranker,
) -> Retriever:
    return RetrievalPipeline(
        query_processor,
        candidate_retriever,
        reranker,
        settings.retrieval,
    )


def build_evidence_selector(settings: AppSettings) -> EvidenceSelector:
    evidence_factory = require_factory(
        EVIDENCE_SELECTORS,
        settings.retrieval.evidence_selector,
        "evidence selector",
    )
    return evidence_factory(
        top_k=settings.retrieval.evidence_top_k,
        min_score=settings.retrieval.min_score,
        per_topic_limit=settings.retrieval.per_topic_limit,
    )


def build_generator(settings: AppSettings, runtime: RuntimeEnvironment) -> Generator:
    generator_factory = require_factory(GENERATORS, settings.generation.provider, "generator")
    return generator_factory(settings.generation, runtime)


def build_rag_strategy(
    settings: AppSettings,
    runtime: RuntimeEnvironment,
    retriever: Retriever,
    generator: Generator,
    evidence_selector: EvidenceSelector,
    vector_store: VectorStore,
) -> RagStrategy:
    strategy_factory = require_factory(
        RAG_STRATEGIES,
        settings.rag.strategy,
        "RAG strategy",
    )
    return strategy_factory(
        RagBuildContext(
            settings=settings,
            runtime=runtime,
            retriever=retriever,
            generator=generator,
            evidence_selector=evidence_selector,
            vector_store=vector_store,
        )
    )
