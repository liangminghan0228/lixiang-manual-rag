from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TopicRef(BaseModel):
    topic_id: str
    title: str
    breadcrumbs: list[str]
    source_url: str
    source_file: str


class ManualCatalogEntry(BaseModel):
    catalog_id: int
    name: str
    url: str
    publish_date: str
    version: str
    manual_key: str
    snapshot_id: str


class ImageRef(BaseModel):
    url: str
    alt: str = ""
    section_path: list[str] = Field(default_factory=list)


class Section(BaseModel):
    title: str
    level: int
    path: list[str]
    blocks: list[str] = Field(default_factory=list)
    images: list[ImageRef] = Field(default_factory=list)


class Document(BaseModel):
    document_id: str
    manual_id: str
    snapshot_id: str
    vehicle_model: str
    manual_key: str | None = None
    manual_name: str | None = None
    manual_version: str | None = None
    topic_id: str
    title: str
    breadcrumb: list[str]
    source_url: str
    sections: list[Section]
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    manual_id: str
    snapshot_id: str
    vehicle_model: str
    manual_key: str | None = None
    manual_name: str | None = None
    manual_version: str | None = None
    topic_id: str
    title: str
    text: str
    section_path: list[str]
    source_url: str
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalFilters(BaseModel):
    manual_id: str | None = None
    snapshot_id: str | None = None
    vehicle_model: str | None = None
    manual_keys: list[str] = Field(default_factory=list)
    manual_names: list[str] = Field(default_factory=list)
    topic_ids: list[str] = Field(default_factory=list)


class RetrievalQuery(BaseModel):
    text: str
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


class QueryPlan(RetrievalQuery):
    original_query: str
    queries: list[str] = Field(default_factory=list)
    fusion_strategy: str = "single_query"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    recall_score: float | None = None
    rerank_score: float | None = None
    retriever_id: str = "unknown"


class RetrievalOutcome(BaseModel):
    results: list[SearchResult]
    timings_ms: dict[str, float]
    query: QueryPlan | RetrievalQuery | None = None


class EvidenceBundle(BaseModel):
    query: RetrievalQuery
    items: list[SearchResult]
    rejected_reason: str | None = None


class Citation(BaseModel):
    index: int
    chunk_id: str
    title: str
    source_url: str
    section_path: list[str] = Field(default_factory=list)
    excerpt: str = ""


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    evidence: list[SearchResult]
    timings_ms: dict[str, float]
    refused: bool = False
    citation_validated: bool = False


class CrawlReport(BaseModel):
    manual_id: str
    vehicle_model: str
    snapshot_id: str
    source_url: str
    directory_refs: int
    unique_topics: int
    downloaded: int
    skipped: int
    failed: int
    topic_refs: list[TopicRef]
    failures: list[dict[str, str]] = Field(default_factory=list)


class IngestionReport(BaseModel):
    manual_id: str
    vehicle_model: str
    snapshot_id: str
    topics_total: int
    topics_parsed: int
    empty_topics: int
    chunks: int
    points_before: int
    points_after: int
    chunks_embedded: int = 0
    chunks_reused: int = 0
    chunks_upserted: int = 0
    embedding_cache_hits: int = 0
    stale_chunks_deleted: int = 0
    timings_ms: dict[str, float]


class BatchIngestionItem(BaseModel):
    catalog: ManualCatalogEntry
    status: str
    report: IngestionReport | None = None
    error: str | None = None


class BatchIngestionReport(BaseModel):
    catalog_url: str
    manuals_discovered: int
    manuals_selected: int
    manuals_completed: int
    manuals_failed: int
    topics_parsed: int
    chunks: int
    points_after: int
    elapsed_seconds: float
    items: list[BatchIngestionItem]
