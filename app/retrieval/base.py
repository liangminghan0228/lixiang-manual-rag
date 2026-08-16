from __future__ import annotations

from typing import Protocol

from app.models import RetrievalFilters, RetrievalOutcome, RetrievalQuery, SearchResult


class QueryProcessor(Protocol):
    @property
    def component_id(self) -> str: ...

    def process(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalQuery: ...


class Retriever(Protocol):
    @property
    def component_id(self) -> str: ...

    def retrieve(
        self,
        question: str | RetrievalQuery,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalOutcome: ...


class Reranker(Protocol):
    @property
    def component_id(self) -> str: ...

    @property
    def is_loaded(self) -> bool: ...

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: list[SearchResult],
    ) -> list[SearchResult]: ...
