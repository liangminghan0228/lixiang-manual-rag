from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3.5-content-safety:free"


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


class QueryProcessorSettings(BaseModel):
    provider: str = "identity"
    model: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_queries: int = Field(default=4, ge=1, le=8)
    aliases: dict[str, str] = Field(default_factory=dict)
    expansions: dict[str, list[str]] = Field(default_factory=dict)


class RetrievalSettings(BaseModel):
    provider: str = "dense"
    query_processor: QueryProcessorSettings = Field(default_factory=QueryProcessorSettings)
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

    @field_validator("query_processor", mode="before")
    @classmethod
    def normalize_query_processor(cls, value: object) -> object:
        if isinstance(value, str):
            return {"provider": value}
        return value

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
    model: str = DEFAULT_OPENROUTER_MODEL
    mock_when_key_missing: bool = True
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    require_inline_citations: bool = True
    citation_repair_attempts: int = 1


class RagSettings(BaseModel):
    strategy: str = "vanilla"
    controller_model: str | None = None
    max_steps: int = Field(default=4, ge=1, le=12)


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
    rag: RagSettings = Field(default_factory=RagSettings)
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
    openrouter_model: str | None = DEFAULT_OPENROUTER_MODEL
    ragas_judge_api_key: str | None = None
    ragas_judge_model: str | None = DEFAULT_OPENROUTER_MODEL
    ragas_judge_base_url: str = "https://openrouter.ai/api/v1"
    query_optimizer_api_key: str | None = None
    query_optimizer_model: str | None = DEFAULT_OPENROUTER_MODEL
    rag_controller_model: str | None = DEFAULT_OPENROUTER_MODEL
    qdrant_url: str | None = None
    qdrant_collection: str | None = None


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _load_config_tree(path: Path, seen: set[Path] | None = None) -> dict[str, object]:
    resolved = path.resolve()
    visited = set(seen or ())
    if resolved in visited:
        raise ValueError(f"cyclic config inheritance detected at {resolved}")
    visited.add(resolved)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be an object: {resolved}")
    parent_value = raw.pop("extends", None)
    if parent_value is None:
        return raw
    if not isinstance(parent_value, str) or not parent_value.strip():
        raise ValueError(f"extends must be a non-empty path: {resolved}")
    parent = Path(parent_value)
    if not parent.is_absolute():
        parent = resolved.parent / parent
    return _deep_merge(_load_config_tree(parent, visited), raw)


def load_config_data(config_path: str | Path) -> dict[str, object]:
    """Load a YAML config with inheritance, without constructing runtime components."""
    selected = Path(config_path)
    if not selected.is_absolute():
        selected = PROJECT_ROOT / selected
    return _load_config_tree(selected)


def load_settings(
    config_path: str | Path | None = None,
    *,
    apply_runtime_overrides: bool = True,
) -> tuple[AppSettings, RuntimeEnvironment]:
    selected = Path(config_path or os.getenv("APP_CONFIG", DEFAULT_CONFIG_PATH))
    if not selected.is_absolute():
        selected = PROJECT_ROOT / selected
    raw = load_config_data(selected)
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
