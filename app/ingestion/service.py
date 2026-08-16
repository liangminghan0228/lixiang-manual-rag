from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from app.ingestion.embedding_cache import EmbeddingCache
from app.models import Chunk, Document, IngestionReport, RetrievalFilters
from app.wiring import Container, build_container

logger = logging.getLogger(__name__)


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "model_dump_json"):
                handle.write(record.model_dump_json())
            else:
                handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    temp.replace(path)


class IngestionService:
    def __init__(
        self,
        container: Container,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        self.container = container
        self.embedding_cache = embedding_cache
        if embedding_cache is None and container.settings.data.embedding_cache_enabled:
            self.embedding_cache = EmbeddingCache(container.settings.data.embedding_cache_path)

    def _embedding_text(self, chunk: Chunk) -> str:
        if not self.container.settings.data.embedding_content_only:
            return chunk.text
        return "\n".join(
            line
            for line in chunk.text.splitlines()
            if not line.startswith(("车型：", "手册版本："))
        )

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.container.embedder.component_id}:{digest}"

    async def run(self) -> IngestionReport:
        total_started = time.perf_counter()
        crawl_started = time.perf_counter()
        crawl_report = await self.container.crawler.crawl()
        crawl_ms = (time.perf_counter() - crawl_started) * 1000
        if crawl_report.failed:
            logger.warning("crawler completed with %s failures", crawl_report.failed)

        parse_started = time.perf_counter()
        raw_topic_dir = self.container.settings.data.raw_dir
        if self.container.settings.data.manual_key:
            raw_topic_dir /= self.container.settings.data.manual_key
        raw_topic_dir = raw_topic_dir / self.container.settings.data.snapshot_id / "topics"
        documents: list[Document] = []
        chunks: list[Chunk] = []
        empty_topics = 0
        for topic in crawl_report.topic_refs:
            topic_path = raw_topic_dir / topic.source_file
            if not topic_path.exists():
                empty_topics += 1
                continue
            document = self.container.parser.parse(
                topic_path.read_bytes(),
                topic,
                manual_id=crawl_report.manual_id,
                snapshot_id=crawl_report.snapshot_id,
                vehicle_model=crawl_report.vehicle_model,
            )
            if document is None:
                empty_topics += 1
                continue
            documents.append(document)
            catalog_metadata = {
                key: value
                for key, value in {
                    "catalog_id": self.container.settings.data.catalog_id,
                    "catalog_name": self.container.settings.data.catalog_name,
                    "catalog_version": self.container.settings.data.catalog_version,
                    "catalog_publish_date": self.container.settings.data.catalog_publish_date,
                    "manual_key": self.container.settings.data.manual_key,
                }.items()
                if value is not None
            }
            document.manual_key = self.container.settings.data.manual_key
            document.manual_name = self.container.settings.data.catalog_name
            document.manual_version = self.container.settings.data.catalog_version
            document.metadata.update(catalog_metadata)
            document_chunks = self.container.chunker.split(document)
            for chunk in document_chunks:
                chunk.metadata.update(catalog_metadata)
            chunks.extend(document_chunks)
        parse_ms = (time.perf_counter() - parse_started) * 1000
        if not chunks:
            raise RuntimeError("no chunks were produced; refusing to create an empty index")

        normalized_dir = self.container.settings.data.normalized_dir
        if self.container.settings.data.manual_key:
            normalized_dir /= self.container.settings.data.manual_key
        normalized_dir /= self.container.settings.data.snapshot_id
        _write_jsonl(normalized_dir / "documents.jsonl", documents)
        _write_jsonl(normalized_dir / "chunks.jsonl", chunks)

        points_before = self.container.vector_store.count()
        existing_hashes = (
            self.container.vector_store.existing_content_hashes(chunks)
            if self.container.settings.data.incremental
            else {}
        )
        pending_chunks = [
            chunk for chunk in chunks if existing_hashes.get(chunk.chunk_id) != chunk.content_hash
        ]
        chunks_reused = len(chunks) - len(pending_chunks)
        chunks_embedded = 0
        embedding_cache_hits = 0
        stale_chunk_ids: list[str] = []
        if self.container.settings.data.incremental and points_before:
            indexed = self.container.vector_store.all_chunks(
                RetrievalFilters(snapshot_id=self.container.settings.data.snapshot_id)
            )
            current_ids = {chunk.chunk_id for chunk in chunks}
            stale_chunk_ids = [
                chunk.chunk_id for chunk in indexed if chunk.chunk_id not in current_ids
            ]
        index_started = time.perf_counter()
        batch_size = self.container.settings.embedding.batch_size
        collection_ready = False
        for start in range(0, len(pending_chunks), batch_size):
            batch = pending_chunks[start : start + batch_size]
            texts = [self._embedding_text(chunk) for chunk in batch]
            if self.embedding_cache is None:
                vectors = await asyncio.to_thread(
                    self.container.embedder.embed_documents,
                    texts,
                )
                chunks_embedded += len(texts)
            else:
                keys = [self._cache_key(text) for text in texts]
                cached = self.embedding_cache.get_many(keys)
                missing = {
                    key: text for key, text in zip(keys, texts, strict=True) if key not in cached
                }
                generated: dict[str, list[float]] = {}
                if missing:
                    missing_vectors = await asyncio.to_thread(
                        self.container.embedder.embed_documents,
                        list(missing.values()),
                    )
                    generated = dict(zip(missing, missing_vectors, strict=True))
                    self.embedding_cache.put_many(generated)
                resolved = {**cached, **generated}
                vectors = [resolved[key] for key in keys]
                chunks_embedded += len(generated)
                embedding_cache_hits += len(keys) - len(generated)
            if not vectors:
                raise RuntimeError("embedder returned no vectors")
            if not collection_ready:
                await asyncio.to_thread(
                    self.container.vector_store.ensure_collection,
                    len(vectors[0]),
                )
                collection_ready = True
            await asyncio.to_thread(self.container.vector_store.upsert, batch, vectors)
        if stale_chunk_ids:
            await asyncio.to_thread(
                self.container.vector_store.delete_chunks,
                stale_chunk_ids,
            )
        index_ms = (time.perf_counter() - index_started) * 1000
        points_after = self.container.vector_store.count()
        total_ms = (time.perf_counter() - total_started) * 1000

        report = IngestionReport(
            manual_id=crawl_report.manual_id,
            vehicle_model=crawl_report.vehicle_model,
            snapshot_id=crawl_report.snapshot_id,
            topics_total=crawl_report.unique_topics,
            topics_parsed=len(documents),
            empty_topics=empty_topics,
            chunks=len(chunks),
            points_before=points_before,
            points_after=points_after,
            chunks_embedded=chunks_embedded,
            chunks_reused=chunks_reused,
            chunks_upserted=len(pending_chunks),
            embedding_cache_hits=embedding_cache_hits,
            stale_chunks_deleted=len(stale_chunk_ids),
            timings_ms={
                "crawl": round(crawl_ms, 3),
                "parse_chunk": round(parse_ms, 3),
                "embed_upsert": round(index_ms, 3),
                "total": round(total_ms, 3),
            },
        )
        self._write_run_manifest(report)
        return report

    def _write_run_manifest(self, report: IngestionReport) -> None:
        settings_payload = self.container.settings.model_dump(mode="json")
        serialized = json.dumps(settings_payload, ensure_ascii=False, sort_keys=True)
        config_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        finished_at = datetime.now(UTC)
        manifest = {
            "experiment_id": self.container.settings.experiment.id,
            "run_id": f"{finished_at.strftime('%Y%m%dT%H%M%SZ')}-{config_hash[:8]}",
            "finished_at": finished_at.isoformat(),
            "config_hash": config_hash,
            "snapshot_id": self.container.settings.data.snapshot_id,
            "components": {
                "chunker": self.container.chunker.component_id,
                "embedder": self.container.embedder.component_id,
                "vector_store": self.container.vector_store.component_id,
                "retriever": self.container.retriever.component_id,
                "reranker": self.container.reranker.component_id,
                "generator": self.container.generator.component_id,
            },
            "settings": settings_payload,
            "ingestion_report": report.model_dump(mode="json"),
        }
        output_dir = self.container.settings.experiment.manifest_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{manifest['run_id']}.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


async def _run(config: str | None) -> None:
    container = build_container(config)
    report = await IngestionService(container).run()
    print(report.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the configured Li Auto manual")
    parser.add_argument("--config", default=None, help="Path to YAML configuration")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()
