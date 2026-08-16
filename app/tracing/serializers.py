from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Any

from app.models import SearchResult
from app.tracing.recorder import trace_level, trace_option


def serialize_candidate(result: SearchResult, *, include_text: bool = False) -> dict[str, Any]:
    limit = trace_option("max_excerpt_chars", 800)
    level = trace_level()
    payload: dict[str, Any] = {
        "chunk_id": result.chunk.chunk_id,
        "topic_id": result.chunk.topic_id,
        "title": result.chunk.title,
        "section_path": result.chunk.section_path,
        "source_url": result.chunk.source_url,
        "rank": result.rank,
        "score": result.score,
        "recall_score": result.recall_score,
        "rerank_score": result.rerank_score,
        "retriever_id": result.retriever_id,
    }
    if level != "summary":
        payload["excerpt"] = result.chunk.text[:limit]
        payload["text_truncated"] = len(result.chunk.text) > limit
    if include_text or level == "full":
        payload["text"] = result.chunk.text
    return payload


def vector_summary(vector: list[float]) -> dict[str, Any]:
    values = [float(value) for value in vector]
    if not values:
        return {"dimension": 0, "preview": [], "stats": {}}
    level = trace_level()
    configured_preview = trace_option("vector_preview_dimensions", 32)
    preview_size = (
        len(values)
        if level == "full"
        else min(configured_preview, 8 if level == "summary" else configured_preview)
    )
    top_absolute = sorted(
        enumerate(values),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:10]
    return {
        "dimension": len(values),
        "preview": values[:preview_size],
        "stats": {
            "min": min(values),
            "max": max(values),
            "mean": fmean(values),
            "std": pstdev(values),
            "l2_norm": math.sqrt(sum(value * value for value in values)),
        },
        "top_absolute_dimensions": [
            {"index": index, "value": value} for index, value in top_absolute
        ],
    }
