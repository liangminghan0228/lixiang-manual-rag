from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models import (
    Answer,
    IngestionReport,
    ManualCatalogEntry,
    RetrievalFilters,
    SearchResult,
)


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


class RetrieveResponse(BaseModel):
    results: list[SearchResult]
    timings_ms: dict[str, float]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


class ChatResponse(Answer):
    pass


class TraceCreateRequest(ChatRequest):
    trace_level: Literal["summary", "sampled", "full"] = "sampled"


class TraceCreateResponse(BaseModel):
    trace_id: str
    status: str
    events_url: str
    snapshot_url: str


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    embedding_provider: str
    embedding_loaded: bool
    generator: str
    generator_mock: bool
    reranker: str
    reranker_loaded: bool
    points_count: int


class ImportRequest(BaseModel):
    force_reembed: bool = False


class ImportJobResponse(BaseModel):
    job_id: str
    status: str
    report: IngestionReport | None = None
    error: str | None = None


class ManualStatus(BaseModel):
    catalog: ManualCatalogEntry
    status: str = "discovered"
    manual_id: str | None = None
    vehicle_model: str | None = None
    topics_parsed: int = 0
    chunks: int = 0
    error: str | None = None
