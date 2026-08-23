from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from app.models import Chunk, RetrievalFilters, RetrievalOutcome, RetrievalQuery, SearchResult
from app.rag.common import StrategyBase
from app.rag.vanilla import VanillaRagStrategy
from app.retrieval.bm25 import tokenize
from app.tracing import emit_trace


def _tokens(text: str) -> set[str]:
    return set(tokenize(text))


class GraphRagStrategy(StrategyBase):
    """Document-structure graph baseline, not entity-community GraphRAG."""

    strategy_name = "graph_rag_document_structure"

    def __init__(
        self,
        retriever,
        generator,
        retrieval_settings,
        *,
        vector_store=None,
        evidence_selector=None,
        generation_settings=None,
    ):
        super().__init__(
            VanillaRagStrategy(
                retriever, generator, retrieval_settings, evidence_selector, generation_settings
            ),
            None,
        )
        self.vector_store = vector_store or getattr(retriever, "vector_store", None)
        if self.vector_store is None or not hasattr(self.vector_store, "all_chunks"):
            raise ValueError("GraphRagStrategy requires a vector_store with all_chunks()")
        self.graph: dict[str, set[str]] = {}

    @property
    def component_id(self) -> str:
        return "graph-rag:document-structure:v1"

    @staticmethod
    def _build_graph(chunks: list[Chunk]) -> dict[str, set[str]]:
        graph = {chunk.chunk_id: set() for chunk in chunks}
        # A graph edge must never cross manual versions.  ``snapshot_id`` is
        # the ingestion/version boundary while manual_id/manual_key identify
        # which manual the snapshot belongs to.
        by_topic: dict[tuple[str, str | None, str, str], list[Chunk]] = defaultdict(list)
        by_section: dict[tuple[str, str | None, str, tuple[str, ...]], list[Chunk]] = defaultdict(
            list
        )
        for chunk in chunks:
            boundary = (chunk.manual_id, chunk.manual_key, chunk.snapshot_id)
            by_topic[(*boundary, chunk.topic_id)].append(chunk)
            by_section[(*boundary, tuple(chunk.section_path))].append(chunk)

        # Connect only adjacent chunks in each structural group.  Besides
        # avoiding quadratic all-to-all edges, sorting makes the graph stable
        # across vector-store iteration order and bounds each node's degree.
        for group in (*by_topic.values(), *by_section.values()):
            ordered = sorted(group, key=lambda chunk: (chunk.document_id, chunk.chunk_id))
            for index in range(len(ordered) - 1):
                left, right = ordered[index], ordered[index + 1]
                graph[left.chunk_id].add(right.chunk_id)
                graph[right.chunk_id].add(left.chunk_id)
        return graph

    async def retrieve(
        self, question: str, top_k: int | None = None, filters: RetrievalFilters | None = None
    ) -> RetrievalOutcome:
        started = time.perf_counter()
        selected_top_k = top_k or self.vanilla.settings.top_k

        def search_graph() -> tuple[
            list[Chunk], dict[str, set[str]], list[str], list[SearchResult]
        ]:
            chunks = list(self.vector_store.all_chunks(filters))
            graph = self._build_graph(chunks)
            qtokens = _tokens(question)
            scored: dict[str, float] = {}
            for chunk in chunks:
                overlap = len(
                    qtokens & _tokens(f"{chunk.title} {' '.join(chunk.section_path)} {chunk.text}")
                )
                if overlap:
                    scored[chunk.chunk_id] = overlap / max(len(qtokens), 1)
            seed_ids = sorted(scored, key=lambda cid: scored[cid], reverse=True)[:selected_top_k]
            for seed_id in list(seed_ids):
                for neighbor_id in graph.get(seed_id, ()):
                    scored.setdefault(neighbor_id, scored[seed_id] * 0.5)
            chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            selected = sorted(scored.items(), key=lambda item: item[1], reverse=True)[
                :selected_top_k
            ]
            results = [
                SearchResult(
                    chunk=chunk_by_id[cid],
                    score=score,
                    recall_score=score,
                    rank=rank,
                    retriever_id="graph-document-structure-v1",
                )
                for rank, (cid, score) in enumerate(selected, 1)
            ]
            return chunks, graph, seed_ids, results

        chunks, self.graph, seed_ids, results = await asyncio.to_thread(search_graph)
        self._trace("strategy.step", step=1, action="lexical_seed", nodes=len(chunks))
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        emit_trace(
            "retrieval",
            "retrieval.graph.completed",
            elapsed_ms=elapsed,
            payload={
                "graph_type": "document_structure",
                "nodes": len(chunks),
                "seeds": seed_ids,
                "expanded": len(results),
            },
        )
        self._trace(
            "strategy.stop",
            reason="no_lexical_seed" if not seed_ids else "graph_retrieval_completed",
            steps=1,
            candidates=len(results),
        )
        return RetrievalOutcome(
            results=results,
            timings_ms={"graph": elapsed, "total": elapsed},
            query=RetrievalQuery(text=question, filters=filters or RetrievalFilters()),
        )

    async def answer(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
        *,
        started: float | None = None,
    ):
        started = started or time.perf_counter()
        outcome = await self.retrieve(question, self.vanilla.settings.candidate_top_k, filters)
        return await self.answer_from_outcome(question, outcome, filters, started=started)

    async def answer_from_outcome(
        self,
        question: str,
        outcome: RetrievalOutcome,
        filters=None,
        *,
        started: float | None = None,
    ):
        return await self.vanilla.answer_from_outcome(question, outcome, filters, started=started)
