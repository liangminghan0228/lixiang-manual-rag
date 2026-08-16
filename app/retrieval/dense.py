from __future__ import annotations

import time

from app.models import RetrievalFilters, RetrievalOutcome, RetrievalQuery
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.tracing import emit_trace
from app.tracing.serializers import serialize_candidate, vector_summary


class DenseRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    @property
    def component_id(self) -> str:
        return f"dense:{self.embedder.component_id}:{self.vector_store.component_id}"

    def retrieve(
        self,
        question: str | RetrievalQuery,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalOutcome:
        query = (
            question
            if isinstance(question, RetrievalQuery)
            else RetrievalQuery(
                text=question,
                filters=filters or RetrievalFilters(),
            )
        )
        if not query.text.strip():
            raise ValueError("question must not be empty")
        started = time.perf_counter()
        embedding_started = time.perf_counter()
        embedding_settings = getattr(self.embedder, "settings", None)
        encoding_config = (
            {
                "batch_size": embedding_settings.batch_size,
                "max_length": embedding_settings.max_length,
                "use_fp16": embedding_settings.use_fp16,
                "return_dense": True,
                "return_sparse": False,
                "return_colbert_vecs": False,
            }
            if embedding_settings is not None
            else {}
        )
        emit_trace(
            "embedding",
            "embedding.started",
            status="started",
            payload={
                "input": query.text,
                "input_chars": len(query.text),
                "model": self.embedder.component_id,
                "config": encoding_config,
            },
        )
        vector = self.embedder.embed_query(query.text)
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        emit_trace(
            "embedding",
            "embedding.completed",
            elapsed_ms=round(embedding_ms, 3),
            payload={
                "input": query.text,
                "input_chars": len(query.text),
                "model": self.embedder.component_id,
                "config": encoding_config,
                **vector_summary(vector),
            },
        )

        qdrant_started = time.perf_counter()
        results = self.vector_store.search(vector, top_k, query.filters)
        qdrant_ms = (time.perf_counter() - qdrant_started) * 1000
        emit_trace(
            "retrieval",
            "retrieval.dense.completed",
            elapsed_ms=round(qdrant_ms, 3),
            payload={
                "top_k": top_k,
                "filters": query.filters.model_dump(mode="json"),
                "candidates": [serialize_candidate(result) for result in results],
            },
        )
        total_ms = (time.perf_counter() - started) * 1000
        return RetrievalOutcome(
            results=results,
            query=query,
            timings_ms={
                "embedding": round(embedding_ms, 3),
                "qdrant": round(qdrant_ms, 3),
                "total": round(total_ms, 3),
            },
        )
