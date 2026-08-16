from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


class DataSettings(BaseModel):
    source_url: str
    snapshot_id: str
    catalog_url: str = "https://manuals.lixiang.com/carlmodels_zh-cn_officialwebsite.json"
    manual_key: str | None = None
    catalog_id: int | None = None
    catalog_name: str | None = None
    catalog_version: str | None = None
    catalog_publish_date: str | None = None
    chunker: str = "heading"
    target_chars: int = 500
    overlap_chars: int = 80
    raw_dir: Path = Path("data/raw")
    normalized_dir: Path = Path("data/normalized")
    request_concurrency: int = 2
    requests_per_second: float = 2.0
    request_timeout_seconds: float = 20.0
    request_retries: int = 3
    incremental: bool = True
    revalidate_remote: bool = True
    embedding_content_only: bool = False
    embedding_cache_enabled: bool = False
    embedding_cache_path: Path = Path("artifacts/embedding-cache.sqlite3")


class EmbeddingSettings(BaseModel):
    provider: str = "bge_m3_local"
    model: str = "BAAI/bge-m3"
    batch_size: int = 8
    max_length: int = 512
    use_fp16: bool = False
    mock_dimension: int = 64


class VectorStoreSettings(BaseModel):
    provider: str = "qdrant"
    url: str = "http://localhost:6333"
    collection: str = "lixiang_mvp_bge_m3_v1"
    timeout_seconds: float = 10.0


class RerankerSettings(BaseModel):
    provider: str = "noop"
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 8
    max_length: int = 1024
    use_fp16: bool = False


class RetrievalSettings(BaseModel):
    provider: str = "dense"
    query_processor: str = "identity"
    top_k: int = 5
    candidate_top_k: int = 10
    evidence_top_k: int = 3
    min_score: float | None = None
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    evidence_selector: str = "diversified"
    per_topic_limit: int = 2
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60

    @model_validator(mode="after")
    def validate_top_k(self) -> RetrievalSettings:
        if self.top_k <= 0:
            raise ValueError("retrieval.top_k must be positive")
        if self.candidate_top_k < self.top_k:
            raise ValueError("retrieval.candidate_top_k must be >= top_k")
        if self.evidence_top_k <= 0 or self.evidence_top_k > self.candidate_top_k:
            raise ValueError("retrieval.evidence_top_k must be in [1, candidate_top_k]")
        if self.per_topic_limit <= 0:
            raise ValueError("retrieval.per_topic_limit must be positive")
        return self


class GenerationSettings(BaseModel):
    provider: str = "openrouter"
    model: str = "openrouter/free"
    mock_when_key_missing: bool = True
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    require_inline_citations: bool = True
    citation_repair_attempts: int = 1


class ExperimentSettings(BaseModel):
    id: str = "dense-bge-heading-v1"
    manifest_dir: Path = Path("reports/runs")


class MonitoringSettings(BaseModel):
    enabled: bool = True


class TracingSettings(BaseModel):
    enabled: bool = True
    ttl_seconds: int = Field(default=1800, ge=60)
    max_traces: int = Field(default=100, ge=1)
    max_events_per_trace: int = Field(default=200, ge=20)
    vector_preview_dimensions: int = Field(default=32, ge=1, le=128)
    rerank_pair_sample_limit: int = Field(default=6, ge=1, le=10)
    max_excerpt_chars: int = Field(default=800, ge=100, le=5000)


class AppSettings(BaseModel):
    data: DataSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    generation: GenerationSettings
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)


class RuntimeEnvironment(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str | None = None
    openrouter_model: str | None = None
    qdrant_url: str | None = None
    qdrant_collection: str | None = None


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(
    config_path: str | Path | None = None,
    *,
    apply_runtime_overrides: bool = True,
) -> tuple[AppSettings, RuntimeEnvironment]:
    selected = Path(config_path or os.getenv("APP_CONFIG", DEFAULT_CONFIG_PATH))
    if not selected.is_absolute():
        selected = PROJECT_ROOT / selected
    raw = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
    settings = AppSettings.model_validate(raw)
    runtime = RuntimeEnvironment()

    settings.data.raw_dir = _resolve_project_path(settings.data.raw_dir)
    settings.data.normalized_dir = _resolve_project_path(settings.data.normalized_dir)
    settings.data.embedding_cache_path = _resolve_project_path(settings.data.embedding_cache_path)
    settings.experiment.manifest_dir = _resolve_project_path(settings.experiment.manifest_dir)
    if apply_runtime_overrides:
        if runtime.qdrant_url:
            settings.vector_store.url = runtime.qdrant_url
        if runtime.qdrant_collection:
            settings.vector_store.collection = runtime.qdrant_collection
        if runtime.openrouter_model:
            settings.generation.model = runtime.openrouter_model
    return settings, runtime
