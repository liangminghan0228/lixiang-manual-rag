from __future__ import annotations

import math
import re
import time
from collections import Counter

from app.models import RetrievalFilters, RetrievalOutcome, RetrievalQuery, SearchResult
from app.retrieval.vector_store import VectorStore
from app.tracing import emit_trace
from app.tracing.serializers import serialize_candidate


def tokenize(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    bigrams = [chinese[index : index + 2] for index in range(max(len(chinese) - 1, 0))]
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [*list(chinese), *bigrams, *words]


class BM25Retriever:
    def __init__(self, vector_store: VectorStore, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.vector_store = vector_store
        self.k1 = k1
        self.b = b

    @property
    def component_id(self) -> str:
        return f"bm25-k1-{self.k1}-b-{self.b}-v1"

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
        started = time.perf_counter()
        chunks = self.vector_store.all_chunks(query.filters)
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        document_count = len(tokenized)
        if not document_count:
            return RetrievalOutcome(results=[], timings_ms={"bm25": 0.0, "total": 0.0}, query=query)
        average_length = sum(len(tokens) for tokens in tokenized) / document_count or 1.0
        document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
        query_tokens = tokenize(query.text)
        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(tokenized):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                df = document_frequency[token]
                idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * len(tokens) / average_length
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((score, index))
        scored.sort(reverse=True)
        results = [
            SearchResult(
                chunk=chunks[index],
                score=score,
                recall_score=score,
                rank=rank,
                retriever_id=self.component_id,
            )
            for rank, (score, index) in enumerate(scored[:top_k], start=1)
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000
        emit_trace(
            "retrieval",
            "retrieval.bm25.completed",
            elapsed_ms=round(elapsed_ms, 3),
            payload={
                "top_k": top_k,
                "document_count": document_count,
                "query_tokens": query_tokens,
                "candidates": [serialize_candidate(result) for result in results],
            },
        )
        return RetrievalOutcome(
            results=results,
            timings_ms={"bm25": round(elapsed_ms, 3), "total": round(elapsed_ms, 3)},
            query=query,
        )
