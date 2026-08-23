from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from app.models import QueryPlan, RetrievalFilters
from app.tracing import emit_trace


class IdentityQueryProcessor:
    @property
    def component_id(self) -> str:
        return "identity-v1"

    def process(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
    ) -> QueryPlan:
        normalized = re.sub(r"\s+", " ", question).strip()
        if not normalized:
            raise ValueError("question must not be empty")
        query = QueryPlan(
            original_query=normalized,
            text=normalized,
            queries=[normalized],
            filters=filters or RetrievalFilters(),
            fusion_strategy="single_query",
            metadata={"processor": self.component_id},
        )
        emit_trace(
            "request",
            "request.normalized",
            payload={"query": query.model_dump(mode="json")},
        )
        return query


def _clean(question: str) -> str:
    value = re.sub(r"\s+", " ", str(question)).strip()
    if not value:
        raise ValueError("question must not be empty")
    return value


def _normalize_text(question: str) -> str:
    value = unicodedata.normalize("NFKC", _clean(question)).lower()
    value = re.sub(r"[，。！？；：、,.!?;:]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


class _RuleQueryProcessor:
    fusion_strategy = "single_query"
    mode = "rule"

    def __init__(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.metadata = dict(metadata or {})

    def _plan(
        self,
        original: str,
        queries: Sequence[str],
        filters: RetrievalFilters | None,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> QueryPlan:
        values = list(dict.fromkeys(_clean(q) for q in queries))
        if not values:
            raise ValueError("query processor produced no queries")
        metadata = {"processor": self.component_id, **self.metadata, **dict(extra_metadata or {})}
        plan = QueryPlan(
            original_query=original,
            text=values[0],
            queries=values,
            filters=filters or RetrievalFilters(),
            fusion_strategy=self.fusion_strategy,
            metadata=metadata,
        )
        emit_trace(
            "request",
            "request.normalized",
            payload={"query": plan.model_dump(mode="json")},
        )
        return plan


class NormalizingQueryProcessor(_RuleQueryProcessor):
    """Whitespace and punctuation-light normalization without changing meaning."""

    def __init__(
        self,
        *,
        alias: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(metadata=metadata)
        self.alias = dict(alias or {})

    @property
    def component_id(self) -> str:
        return "normalizing-v1"

    def process(self, question: str, filters: RetrievalFilters | None = None) -> QueryPlan:
        original = _clean(question)
        normalized = _normalize_text(original)
        for source, target in self.alias.items():
            normalized = re.sub(re.escape(source), target, normalized, flags=re.IGNORECASE)
        return self._plan(original, [normalized], filters)


class ExpansionQueryProcessor(_RuleQueryProcessor):
    def __init__(
        self,
        expansions: Mapping[str, Sequence[str] | str] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(metadata=metadata)
        self.expansions = dict(expansions or {})

    @property
    def component_id(self) -> str:
        return "expansion-v1"

    def process(self, question: str, filters: RetrievalFilters | None = None) -> QueryPlan:
        original = _clean(question)
        additions: list[str] = []
        for term, values in self.expansions.items():
            if re.search(re.escape(term), original, flags=re.IGNORECASE):
                choices = [values] if isinstance(values, str) else values
                additions.extend(str(item).strip() for item in choices if str(item).strip())
        expanded = " ".join(dict.fromkeys([original, *additions]))
        return self._plan(original, [expanded], filters)


class LLMQueryProcessor(_RuleQueryProcessor):
    """Common strict JSON implementation for model-backed query processors."""

    def __init__(
        self, planner: Any, *, max_queries: int = 4, metadata: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(metadata=metadata)
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        self.planner, self.max_queries = planner, max_queries

    @property
    def planner_id(self) -> str:
        return str(getattr(self.planner, "component_id", type(self.planner).__name__))

    def _plan(
        self,
        original: str,
        queries: Sequence[str],
        filters: RetrievalFilters | None,
        *,
        planner_elapsed_ms: float,
    ) -> QueryPlan:
        return super()._plan(
            original,
            queries,
            filters,
            extra_metadata={
                "planner": self.planner_id,
                "planner_elapsed_ms": planner_elapsed_ms,
            },
        )

    def _ask(self, original: str) -> tuple[Mapping[str, Any], float]:
        started = time.perf_counter()
        model = getattr(self.planner, "model", None)
        payload = {"strategy": self.mode, "model": model or self.planner_id}
        try:
            if hasattr(self.planner, "plan"):
                raw = self.planner.plan(original, strategy=self.mode)
            elif hasattr(self.planner, "complete"):
                raw = self.planner.complete(original)
            elif callable(self.planner):
                raw = self.planner(original)
            else:
                raise TypeError("planner must provide plan() or complete()")
            import json

            data = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(data, Mapping):
                raise ValueError("planner output must be a JSON object")
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            emit_trace(
                "query",
                "query.planner",
                elapsed_ms=elapsed_ms,
                payload=payload,
            )
            return data, elapsed_ms
        except Exception as exc:
            emit_trace(
                "query",
                "query.planner",
                status="failed",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                payload={**payload, "error": type(exc).__name__},
            )
            raise ValueError(f"query planner failed ({self.mode}): {exc}") from exc


class LLMRewriteQueryProcessor(LLMQueryProcessor):
    mode = "rewrite"

    @property
    def component_id(self) -> str:
        return f"llm-rewrite:{self.planner_id}:v1"

    def process(self, question: str, filters: RetrievalFilters | None = None) -> QueryPlan:
        original = _clean(question)
        data, elapsed_ms = self._ask(original)
        query = data.get("query", data.get("rewrite", data.get("rewritten_query")))
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query planner output missing non-empty 'query'")
        return self._plan(original, [query], filters, planner_elapsed_ms=elapsed_ms)


class HyDEQueryProcessor(LLMQueryProcessor):
    mode = "hyde"

    @property
    def component_id(self) -> str:
        return f"hyde:{self.planner_id}:v1"

    def process(self, question: str, filters: RetrievalFilters | None = None) -> QueryPlan:
        original = _clean(question)
        data, elapsed_ms = self._ask(original)
        query = data.get("query", data.get("hypothetical_document", data.get("hyde")))
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query planner output missing non-empty 'query'")
        return self._plan(original, [query], filters, planner_elapsed_ms=elapsed_ms)


class MultiQueryProcessor(LLMQueryProcessor):
    mode = "multi"
    fusion_strategy = "rrf"

    @property
    def component_id(self) -> str:
        return f"multi-query:{self.planner_id}:v1"

    def process(self, question: str, filters: RetrievalFilters | None = None) -> QueryPlan:
        original = _clean(question)
        data, elapsed_ms = self._ask(original)
        values = data.get("queries")
        if not isinstance(values, list) or not values:
            raise ValueError("query planner output missing non-empty 'queries' list")
        return self._plan(
            original,
            [str(q) for q in values[: self.max_queries]],
            filters,
            planner_elapsed_ms=elapsed_ms,
        )


class DecompositionQueryProcessor(MultiQueryProcessor):
    mode = "decomposition"

    @property
    def component_id(self) -> str:
        return f"decomposition:{self.planner_id}:v1"


# Short names are convenient for applications while the explicit names above
# remain the public, self-documenting API.
RewriteQueryProcessor = LLMRewriteQueryProcessor
