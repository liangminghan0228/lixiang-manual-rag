from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.generation.mock import MockGenerator
from app.generation.openrouter import OpenRouterGenerator
from app.ingestion.chunker import HeadingChunker
from app.rag import AgenticRagStrategy, GraphRagStrategy, SelfRagStrategy, VanillaRagStrategy
from app.rag.controller import OpenRouterJSONController
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import BgeM3Embedder, DeterministicHashEmbedder
from app.retrieval.evidence import DiversifiedEvidenceSelector
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query import (
    DecompositionQueryProcessor,
    ExpansionQueryProcessor,
    HyDEQueryProcessor,
    IdentityQueryProcessor,
    LLMRewriteQueryProcessor,
    MultiQueryProcessor,
    NormalizingQueryProcessor,
)
from app.retrieval.query_llm import OpenAICompatibleQueryPlanner
from app.retrieval.reranker import BgeReranker, NoOpReranker
from app.retrieval.vector_store import InMemoryVectorStore, QdrantVectorStore
from app.settings import RuntimeEnvironment


def _build_openrouter_generator(settings: Any, runtime: RuntimeEnvironment) -> Any:
    if runtime.openrouter_api_key:
        return OpenRouterGenerator(settings, runtime.openrouter_api_key)
    if settings.mock_when_key_missing:
        return MockGenerator()
    raise ValueError("OPENROUTER_API_KEY is required")


def _fixed_query_planner(settings: Any, runtime: RuntimeEnvironment) -> Any:
    api_key = runtime.query_optimizer_api_key or runtime.openrouter_api_key
    model = settings.model or runtime.query_optimizer_model or runtime.openrouter_model
    if not api_key:
        raise ValueError("QUERY_OPTIMIZER_API_KEY or OPENROUTER_API_KEY is required")
    if not model or model == "openrouter/free":
        raise ValueError("a fixed query optimizer model is required")
    return OpenAICompatibleQueryPlanner(
        api_key,
        model,
        base_url=settings.base_url,
        timeout_seconds=settings.timeout_seconds,
    )


def _build_controller(rag_settings: Any, generation_settings: Any, runtime: Any) -> Any:
    if not runtime.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required for this RAG strategy")
    model = (
        rag_settings.controller_model
        or runtime.rag_controller_model
        or runtime.openrouter_model
        or generation_settings.model
    )
    if not model or model == "openrouter/free":
        raise ValueError("a fixed RAG controller model is required")
    controller_settings = generation_settings.model_copy(
        update={"model": model, "temperature": 0.0}
    )
    return OpenRouterJSONController(controller_settings, runtime.openrouter_api_key)


def _build_vanilla_strategy(context: Any) -> Any:
    return VanillaRagStrategy(
        context.retriever,
        context.generator,
        context.settings.retrieval,
        context.evidence_selector,
        context.settings.generation,
    )


def _build_self_rag_strategy(context: Any) -> Any:
    controller = _build_controller(
        context.settings.rag,
        context.settings.generation,
        context.runtime,
    )
    return SelfRagStrategy(
        context.retriever,
        context.generator,
        context.settings.retrieval,
        controller=controller,
        evidence_selector=context.evidence_selector,
        generation_settings=context.settings.generation,
    )


def _build_agentic_rag_strategy(context: Any) -> Any:
    controller = _build_controller(
        context.settings.rag,
        context.settings.generation,
        context.runtime,
    )
    return AgenticRagStrategy(
        context.retriever,
        context.generator,
        context.settings.retrieval,
        controller=controller,
        max_steps=context.settings.rag.max_steps,
        evidence_selector=context.evidence_selector,
        generation_settings=context.settings.generation,
    )


def _build_graph_rag_strategy(context: Any) -> Any:
    return GraphRagStrategy(
        context.retriever,
        context.generator,
        context.settings.retrieval,
        vector_store=context.vector_store,
        evidence_selector=context.evidence_selector,
        generation_settings=context.settings.generation,
    )


Factory = Callable[..., Any]

CHUNKERS: dict[str, Factory] = {"heading": HeadingChunker}
EMBEDDERS: dict[str, Factory] = {
    "bge_m3_local": lambda settings: BgeM3Embedder(settings),
    "hash_mock": lambda settings: DeterministicHashEmbedder(settings.mock_dimension),
}
VECTOR_STORES: dict[str, Factory] = {
    "qdrant": lambda settings: QdrantVectorStore(settings),
    "in_memory": lambda _: InMemoryVectorStore(),
}
QUERY_PROCESSORS: dict[str, Factory] = {
    "identity": lambda _, __: IdentityQueryProcessor(),
    "normalize": lambda settings, _: NormalizingQueryProcessor(alias=settings.aliases),
    "expansion": lambda settings, _: ExpansionQueryProcessor(settings.expansions),
    "rewrite": lambda settings, runtime: LLMRewriteQueryProcessor(
        _fixed_query_planner(settings, runtime),
        max_queries=settings.max_queries,
    ),
    "multi_query": lambda settings, runtime: MultiQueryProcessor(
        _fixed_query_planner(settings, runtime),
        max_queries=settings.max_queries,
    ),
    "hyde": lambda settings, runtime: HyDEQueryProcessor(
        _fixed_query_planner(settings, runtime),
        max_queries=settings.max_queries,
    ),
    "decomposition": lambda settings, runtime: DecompositionQueryProcessor(
        _fixed_query_planner(settings, runtime),
        max_queries=settings.max_queries,
    ),
}
RERANKERS: dict[str, Factory] = {
    "noop": lambda _: NoOpReranker(),
    "bge_local": lambda settings: BgeReranker(settings),
}
RETRIEVERS: dict[str, Factory] = {
    "dense": lambda settings, embedder, vector_store: DenseRetriever(embedder, vector_store),
    "bm25": lambda settings, _, vector_store: BM25Retriever(
        vector_store,
        k1=settings.bm25_k1,
        b=settings.bm25_b,
    ),
    "hybrid": lambda settings, embedder, vector_store: HybridRetriever(
        DenseRetriever(embedder, vector_store),
        BM25Retriever(vector_store, k1=settings.bm25_k1, b=settings.bm25_b),
        rrf_k=settings.rrf_k,
    ),
}
EVIDENCE_SELECTORS: dict[str, Factory] = {
    "diversified": DiversifiedEvidenceSelector,
}
GENERATORS: dict[str, Factory] = {
    "mock": lambda _, __: MockGenerator(),
    "openrouter": _build_openrouter_generator,
}
RAG_STRATEGIES: dict[str, Factory] = {
    "vanilla": _build_vanilla_strategy,
    "self_rag": _build_self_rag_strategy,
    "agentic_rag": _build_agentic_rag_strategy,
    "graph_rag": _build_graph_rag_strategy,
}


def require_factory(registry: dict[str, Factory], name: str, component: str) -> Factory:
    try:
        return registry[name]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"unsupported {component}: {name}; choose one of: {choices}") from exc
