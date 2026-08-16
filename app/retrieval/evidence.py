from __future__ import annotations

from typing import Protocol

from app.models import EvidenceBundle, RetrievalQuery, SearchResult
from app.tracing import emit_trace


class EvidenceSelector(Protocol):
    @property
    def component_id(self) -> str: ...

    def select(
        self,
        query: RetrievalQuery,
        candidates: list[SearchResult],
    ) -> EvidenceBundle: ...


class DiversifiedEvidenceSelector:
    def __init__(
        self,
        *,
        top_k: int,
        min_score: float | None,
        per_topic_limit: int,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.per_topic_limit = per_topic_limit

    @property
    def component_id(self) -> str:
        return f"diversified-top-{self.top_k}-topic-{self.per_topic_limit}-v1"

    def select(
        self,
        query: RetrievalQuery,
        candidates: list[SearchResult],
    ) -> EvidenceBundle:
        selected: list[SearchResult] = []
        topic_counts: dict[str, int] = {}
        seen_hashes: set[str] = set()
        decisions: list[dict[str, str | int | float]] = []
        for candidate in candidates:
            reason = "selected"
            if len(selected) >= self.top_k:
                reason = "evidence_top_k_reached"
            if self.min_score is not None and candidate.score < self.min_score:
                reason = "below_min_score"
            elif candidate.chunk.content_hash in seen_hashes:
                reason = "duplicate_content_hash"
            else:
                count = topic_counts.get(candidate.chunk.topic_id, 0)
                if count >= self.per_topic_limit:
                    reason = "topic_limit_exceeded"
                elif len(selected) < self.top_k:
                    selected.append(candidate)
                    seen_hashes.add(candidate.chunk.content_hash)
                    topic_counts[candidate.chunk.topic_id] = count + 1
            decisions.append(
                {
                    "chunk_id": candidate.chunk.chunk_id,
                    "rank": candidate.rank,
                    "score": candidate.score,
                    "decision": "selected" if reason == "selected" else "rejected",
                    "reason": reason,
                }
            )
        emit_trace(
            "evidence",
            "evidence.completed",
            payload={
                "top_k": self.top_k,
                "min_score": self.min_score,
                "per_topic_limit": self.per_topic_limit,
                "selected_chunk_ids": [item.chunk.chunk_id for item in selected],
                "decisions": decisions,
            },
        )
        return EvidenceBundle(
            query=query,
            items=selected,
            rejected_reason=None if selected else "no_candidate_passed_evidence_policy",
        )
