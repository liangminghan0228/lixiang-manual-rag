from __future__ import annotations

import threading
import time
from typing import Any

from app.models import RetrievalQuery, SearchResult
from app.settings import RerankerSettings
from app.tracing import emit_trace
from app.tracing.recorder import trace_level, trace_option
from app.tracing.serializers import serialize_candidate


class NoOpReranker:
    @property
    def component_id(self) -> str:
        return "noop-reranker-v1"

    @property
    def is_loaded(self) -> bool:
        return True

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        results = [
            result.model_copy(
                update={
                    "rank": rank,
                    "recall_score": result.recall_score or result.score,
                    "rerank_score": None,
                }
            )
            for rank, result in enumerate(candidates, start=1)
        ]
        emit_trace(
            "rerank",
            "rerank.completed",
            elapsed_ms=0.0,
            payload={
                "model": self.component_id,
                "candidate_count": len(candidates),
                "items": [serialize_candidate(result) for result in results],
            },
        )
        return results


class BgeReranker:
    def __init__(self, settings: RerankerSettings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def component_id(self) -> str:
        return f"bge-reranker:{self.settings.model}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(
                    self.settings.model,
                    use_fp16=self.settings.use_fp16,
                )
        return self._model

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        if not candidates:
            return []
        started = time.perf_counter()
        model = self._load_model()
        pairs = [(query.text, candidate.chunk.text) for candidate in candidates]
        with self._lock:
            raw_scores = model.compute_score(
                pairs,
                batch_size=self.settings.batch_size,
                max_length=self.settings.max_length,
                normalize=True,
            )
        scores = [raw_scores] if isinstance(raw_scores, float) else list(raw_scores)
        reranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        results = [
            candidate.model_copy(
                update={
                    "rank": rank,
                    "recall_score": candidate.recall_score or candidate.score,
                    "rerank_score": float(score),
                    "score": float(score),
                }
            )
            for rank, (candidate, score) in enumerate(reranked, start=1)
        ]
        before_ranks = {item.chunk.chunk_id: item.rank for item in candidates}
        items = []
        for result in results:
            item = serialize_candidate(result)
            item["before_rank"] = before_ranks[result.chunk.chunk_id]
            item["after_rank"] = result.rank
            item["rank_delta"] = before_ranks[result.chunk.chunk_id] - result.rank
            items.append(item)

        level = trace_level()
        sample_limit = trace_option("rerank_pair_sample_limit", 6)
        sample_ids = {result.chunk.chunk_id for result in results[:3]}
        if results:

            def rank_delta(item: SearchResult) -> int:
                return before_ranks[item.chunk.chunk_id] - item.rank

            sample_ids.add(max(results, key=rank_delta).chunk.chunk_id)
            sample_ids.add(min(results, key=rank_delta).chunk.chunk_id)
        if level == "full":
            sample_ids = {result.chunk.chunk_id for result in results}
            sample_limit = len(results)
        elif level == "summary":
            sample_ids = set()
        sampled_pairs = []
        for result in results:
            if result.chunk.chunk_id not in sample_ids or len(sampled_pairs) >= sample_limit:
                continue
            sampled_pairs.append(
                {
                    "chunk_id": result.chunk.chunk_id,
                    "query": query.text,
                    "candidate": (
                        result.chunk.text
                        if level == "full"
                        else serialize_candidate(result).get("excerpt", "")
                    ),
                    "candidate_chars": len(result.chunk.text),
                    "text_truncated_for_trace": len(result.chunk.text)
                    > trace_option("max_excerpt_chars", 800),
                }
            )
        emit_trace(
            "rerank",
            "rerank.completed",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            payload={
                "model": self.component_id,
                "candidate_count": len(candidates),
                "batch_size": self.settings.batch_size,
                "max_length": self.settings.max_length,
                "normalize": True,
                "items": items,
                "sampled_pairs": sampled_pairs,
            },
        )
        return results
