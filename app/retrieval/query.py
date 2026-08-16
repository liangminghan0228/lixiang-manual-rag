from __future__ import annotations

import re

from app.models import RetrievalFilters, RetrievalQuery
from app.tracing import emit_trace


class IdentityQueryProcessor:
    @property
    def component_id(self) -> str:
        return "identity-v1"

    def process(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalQuery:
        normalized = re.sub(r"\s+", " ", question).strip()
        if not normalized:
            raise ValueError("question must not be empty")
        query = RetrievalQuery(text=normalized, filters=filters or RetrievalFilters())
        emit_trace(
            "request",
            "request.normalized",
            payload={"query": query.model_dump(mode="json")},
        )
        return query
