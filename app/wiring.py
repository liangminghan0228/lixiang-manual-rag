from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.generation.base import Generator
from app.generation.mock import MockGenerator
from app.generation.openrouter import OpenRouterGenerator
from app.generation.service import ChatService
from app.ingestion.chunker import Chunker
from app.ingestion.crawler import LiXiangManualCrawler
from app.ingestion.parser import LiXiangHtmlParser
from app.registry import (
    CHUNKERS,
    EMBEDDERS,
    EVIDENCE_SELECTORS,
    QUERY_PROCESSORS,
    RERANKERS,
    VECTOR_STORES,
    require_factory,
)
from app.retrieval.base import Reranker, Retriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import Embedder
from app.retrieval.evidence import EvidenceSelector
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.vector_store import VectorStore
from app.settings import AppSettings, RuntimeEnvironment, load_settings


@dataclass
class Container:
    settings: AppSettings
    runtime: RuntimeEnvironment
    crawler: LiXiangManualCrawler
    parser: LiXiangHtmlParser
    chunker: Chunker
    embedder: Embedder
    vector_store: VectorStore
    candidate_retriever: Retriever
    retriever: Retriever
    reranker: Reranker
    evidence_selector: EvidenceSelector
    generator: Generator
    chat_service: ChatService


def build_container(
    config_path: str | Path | None = None,
    *,
    apply_runtime_overrides: bool = True,
) -> Container:
    settings, runtime = load_settings(
        config_path,
        apply_runtime_overrides=apply_runtime_overrides,
    )

    chunker_factory = require_factory(CHUNKERS, settings.data.chunker, "chunker")
    chunker = chunker_factory(settings.data.target_chars, settings.data.overlap_chars)

    if settings.embedding.provider == "bge_m3_local":
        embedder_factory = require_factory(EMBEDDERS, settings.embedding.provider, "embedder")
        embedder: Embedder = embedder_factory(settings.embedding)
    elif settings.embedding.provider == "hash_mock":
        embedder_factory = require_factory(EMBEDDERS, settings.embedding.provider, "embedder")
        embedder = embedder_factory(settings.embedding.mock_dimension)
    else:
        raise ValueError(f"unsupported embedder: {settings.embedding.provider}")

    if settings.vector_store.provider == "qdrant":
        store_factory = require_factory(
            VECTOR_STORES, settings.vector_store.provider, "vector store"
        )
        vector_store: VectorStore = store_factory(settings.vector_store)
    elif settings.vector_store.provider == "in_memory":
        store_factory = require_factory(
            VECTOR_STORES, settings.vector_store.provider, "vector store"
        )
        vector_store = store_factory()
    else:
        raise ValueError(f"unsupported vector store: {settings.vector_store.provider}")

    dense: Retriever = DenseRetriever(embedder, vector_store)
    bm25: Retriever = BM25Retriever(
        vector_store,
        k1=settings.retrieval.bm25_k1,
        b=settings.retrieval.bm25_b,
    )
    if settings.retrieval.provider == "dense":
        candidate_retriever = dense
    elif settings.retrieval.provider == "bm25":
        candidate_retriever = bm25
    elif settings.retrieval.provider == "hybrid":
        candidate_retriever = HybridRetriever(
            dense,
            bm25,
            rrf_k=settings.retrieval.rrf_k,
        )
    else:
        raise ValueError(f"unsupported retriever: {settings.retrieval.provider}")

    query_processor_factory = require_factory(
        QUERY_PROCESSORS,
        settings.retrieval.query_processor,
        "query processor",
    )
    query_processor = query_processor_factory()
    reranker_factory = require_factory(
        RERANKERS,
        settings.retrieval.reranker.provider,
        "reranker",
    )
    reranker = (
        reranker_factory(settings.retrieval.reranker)
        if settings.retrieval.reranker.provider == "bge_local"
        else reranker_factory()
    )
    retriever: Retriever = RetrievalPipeline(
        query_processor,
        candidate_retriever,
        reranker,
        settings.retrieval,
    )
    evidence_factory = require_factory(
        EVIDENCE_SELECTORS,
        settings.retrieval.evidence_selector,
        "evidence selector",
    )
    evidence_selector: EvidenceSelector = evidence_factory(
        top_k=settings.retrieval.evidence_top_k,
        min_score=settings.retrieval.min_score,
        per_topic_limit=settings.retrieval.per_topic_limit,
    )

    if settings.generation.provider == "mock":
        generator: Generator = MockGenerator()
    elif settings.generation.provider == "openrouter":
        if runtime.openrouter_api_key:
            generator = OpenRouterGenerator(settings.generation, runtime.openrouter_api_key)
        elif settings.generation.mock_when_key_missing:
            generator = MockGenerator()
        else:
            raise ValueError("OPENROUTER_API_KEY is required")
    else:
        raise ValueError(f"unsupported generator: {settings.generation.provider}")

    return Container(
        settings=settings,
        runtime=runtime,
        crawler=LiXiangManualCrawler(settings.data),
        parser=LiXiangHtmlParser(),
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        candidate_retriever=candidate_retriever,
        retriever=retriever,
        reranker=reranker,
        evidence_selector=evidence_selector,
        generator=generator,
        chat_service=ChatService(
            retriever,
            generator,
            settings.retrieval,
            evidence_selector,
            settings.generation,
        ),
    )
