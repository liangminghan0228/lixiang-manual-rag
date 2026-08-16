from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.generation.mock import MockGenerator
from app.ingestion.chunker import HeadingChunker
from app.retrieval.embedder import BgeM3Embedder, DeterministicHashEmbedder
from app.retrieval.evidence import DiversifiedEvidenceSelector
from app.retrieval.query import IdentityQueryProcessor
from app.retrieval.reranker import BgeReranker, NoOpReranker
from app.retrieval.vector_store import InMemoryVectorStore, QdrantVectorStore

Factory = Callable[..., Any]

CHUNKERS: dict[str, Factory] = {"heading": HeadingChunker}
EMBEDDERS: dict[str, Factory] = {
    "bge_m3_local": BgeM3Embedder,
    "hash_mock": DeterministicHashEmbedder,
}
VECTOR_STORES: dict[str, Factory] = {
    "qdrant": QdrantVectorStore,
    "in_memory": InMemoryVectorStore,
}
QUERY_PROCESSORS: dict[str, Factory] = {"identity": IdentityQueryProcessor}
RERANKERS: dict[str, Factory] = {
    "noop": NoOpReranker,
    "bge_local": BgeReranker,
}
EVIDENCE_SELECTORS: dict[str, Factory] = {
    "diversified": DiversifiedEvidenceSelector,
}
GENERATORS: dict[str, Factory] = {"mock": MockGenerator}


def require_factory(registry: dict[str, Factory], name: str, component: str) -> Factory:
    try:
        return registry[name]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"unsupported {component}: {name}; choose one of: {choices}") from exc
