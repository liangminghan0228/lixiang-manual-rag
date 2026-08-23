from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.factories import (
    build_candidate_retriever,
    build_chunker,
    build_embedder,
    build_evidence_selector,
    build_generator,
    build_query_processor,
    build_rag_strategy,
    build_reranker,
    build_retriever,
    build_vector_store,
)
from app.generation.base import Generator
from app.generation.service import ChatService
from app.ingestion.chunker import Chunker
from app.ingestion.crawler import LiXiangManualCrawler
from app.ingestion.parser import LiXiangHtmlParser
from app.rag.base import RagStrategy
from app.retrieval.base import QueryProcessor, Reranker, Retriever
from app.retrieval.embedder import Embedder
from app.retrieval.evidence import EvidenceSelector
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
    query_processor: QueryProcessor
    candidate_retriever: Retriever
    retriever: Retriever
    reranker: Reranker
    evidence_selector: EvidenceSelector
    generator: Generator
    rag_strategy: RagStrategy
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

    chunker: Chunker = build_chunker(settings)
    embedder: Embedder = build_embedder(settings)
    vector_store: VectorStore = build_vector_store(settings)
    candidate_retriever: Retriever = build_candidate_retriever(
        settings,
        embedder,
        vector_store,
    )
    query_processor: QueryProcessor = build_query_processor(settings, runtime)
    reranker: Reranker = build_reranker(settings)
    retriever: Retriever = build_retriever(
        settings,
        query_processor,
        candidate_retriever,
        reranker,
    )
    evidence_selector: EvidenceSelector = build_evidence_selector(settings)
    generator: Generator = build_generator(settings, runtime)
    rag_strategy: RagStrategy = build_rag_strategy(
        settings,
        runtime,
        retriever,
        generator,
        evidence_selector,
        vector_store,
    )

    return Container(
        settings=settings,
        runtime=runtime,
        crawler=LiXiangManualCrawler(settings.data),
        parser=LiXiangHtmlParser(),
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        query_processor=query_processor,
        candidate_retriever=candidate_retriever,
        retriever=retriever,
        reranker=reranker,
        evidence_selector=evidence_selector,
        generator=generator,
        rag_strategy=rag_strategy,
        chat_service=ChatService(
            retriever,
            generator,
            settings.retrieval,
            evidence_selector,
            settings.generation,
            strategy=rag_strategy,
        ),
    )
