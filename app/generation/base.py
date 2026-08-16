from __future__ import annotations

from typing import Protocol

from app.models import SearchResult


class Generator(Protocol):
    @property
    def component_id(self) -> str: ...

    async def generate(self, question: str, contexts: list[SearchResult]) -> str: ...

    async def repair_citations(
        self,
        question: str,
        draft: str,
        contexts: list[SearchResult],
    ) -> str: ...
