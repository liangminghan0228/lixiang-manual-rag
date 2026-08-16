from __future__ import annotations

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "rag_http_requests_total",
    "HTTP requests handled by the RAG API",
    ("method", "path", "status"),
)
HTTP_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "End-to-end HTTP request latency",
    ("method", "path"),
)
PIPELINE_LATENCY = Histogram(
    "rag_pipeline_stage_duration_seconds",
    "RAG pipeline latency by stage",
    ("stage",),
)
RAG_ERRORS = Counter(
    "rag_errors_total",
    "RAG request errors",
    ("operation", "exception"),
)


def observe_timings(timings_ms: dict[str, float]) -> None:
    for stage, milliseconds in timings_ms.items():
        PIPELINE_LATENCY.labels(stage=stage).observe(milliseconds / 1000.0)
