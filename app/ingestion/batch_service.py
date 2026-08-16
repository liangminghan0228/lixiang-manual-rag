from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from app.ingestion.catalog import fetch_catalog
from app.ingestion.crawler import LiXiangManualCrawler
from app.ingestion.embedding_cache import EmbeddingCache
from app.ingestion.service import IngestionService
from app.models import BatchIngestionItem, BatchIngestionReport, ManualCatalogEntry
from app.wiring import Container, build_container

logger = logging.getLogger(__name__)


def select_entries(
    entries: list[ManualCatalogEntry],
    include_pattern: str | None,
    limit: int | None,
) -> list[ManualCatalogEntry]:
    selected = entries
    if include_pattern:
        pattern = re.compile(include_pattern, re.IGNORECASE)
        selected = [
            entry
            for entry in selected
            if pattern.search(entry.name) or pattern.search(entry.manual_key)
        ]
    return selected[:limit] if limit else selected


def container_for_manual(base: Container, entry: ManualCatalogEntry) -> Container:
    settings = base.settings.model_copy(deep=True)
    settings.data.source_url = entry.url
    settings.data.snapshot_id = entry.snapshot_id
    settings.data.manual_key = entry.manual_key
    settings.data.catalog_id = entry.catalog_id
    settings.data.catalog_name = entry.name
    settings.data.catalog_version = entry.version
    settings.data.catalog_publish_date = entry.publish_date
    return replace(
        base,
        settings=settings,
        crawler=LiXiangManualCrawler(settings.data),
    )


class BatchIngestionService:
    def __init__(
        self,
        base_container: Container,
        *,
        include_pattern: str | None = None,
        limit: int | None = None,
        fail_fast: bool = False,
    ) -> None:
        self.base = base_container
        self.include_pattern = include_pattern
        self.limit = limit
        self.fail_fast = fail_fast
        self.embedding_cache = (
            EmbeddingCache(base_container.settings.data.embedding_cache_path)
            if base_container.settings.data.embedding_cache_enabled
            else None
        )

    async def run(self) -> BatchIngestionReport:
        started = time.perf_counter()
        entries = await fetch_catalog(
            self.base.settings.data.catalog_url,
            self.base.settings.data.request_timeout_seconds,
        )
        selected = select_entries(entries, self.include_pattern, self.limit)
        self._write_catalog(entries)
        items: list[BatchIngestionItem] = []
        for index, entry in enumerate(selected, start=1):
            logger.info("ingesting manual %s/%s: %s", index, len(selected), entry.name)
            try:
                report = await IngestionService(
                    container_for_manual(self.base, entry),
                    self.embedding_cache,
                ).run()
                items.append(BatchIngestionItem(catalog=entry, status="completed", report=report))
            except Exception as exc:  # noqa: BLE001 - batch must retain per-manual failures
                logger.exception("manual ingestion failed: %s", entry.name)
                items.append(
                    BatchIngestionItem(
                        catalog=entry,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                if self.fail_fast:
                    break
            self._write_progress(entries, selected, items, started)
        report = self._build_report(entries, selected, items, started)
        self._write_report(report)
        return report

    def _build_report(
        self,
        entries: list[ManualCatalogEntry],
        selected: list[ManualCatalogEntry],
        items: list[BatchIngestionItem],
        started: float,
    ) -> BatchIngestionReport:
        completed = [item for item in items if item.report is not None]
        return BatchIngestionReport(
            catalog_url=self.base.settings.data.catalog_url,
            manuals_discovered=len(entries),
            manuals_selected=len(selected),
            manuals_completed=len(completed),
            manuals_failed=sum(item.status == "failed" for item in items),
            topics_parsed=sum(item.report.topics_parsed for item in completed if item.report),
            chunks=sum(item.report.chunks for item in completed if item.report),
            points_after=self.base.vector_store.count(),
            elapsed_seconds=round(time.perf_counter() - started, 3),
            items=items,
        )

    def _write_catalog(self, entries: list[ManualCatalogEntry]) -> None:
        path = self.base.settings.data.raw_dir / "catalog.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([entry.model_dump() for entry in entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_progress(
        self,
        entries: list[ManualCatalogEntry],
        selected: list[ManualCatalogEntry],
        items: list[BatchIngestionItem],
        started: float,
    ) -> None:
        self._write_latest(self._build_report(entries, selected, items, started))

    def _write_latest(self, report: BatchIngestionReport) -> Path:
        output = self.base.settings.experiment.manifest_dir / "batch-latest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(output)
        return output

    def _write_report(self, report: BatchIngestionReport) -> None:
        output = self._write_latest(report)
        timestamped = output.with_name(f"batch-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
        timestamped.write_text(report.model_dump_json(indent=2), encoding="utf-8")


async def _run(args: argparse.Namespace) -> None:
    # A batch index is an experiment artifact. Its collection and model must
    # come from the selected YAML, not a single-model override left in .env.
    container = build_container(args.config, apply_runtime_overrides=False)
    report = await BatchIngestionService(
        container,
        include_pattern=args.include,
        limit=args.limit,
        fail_fast=args.fail_fast,
    ).run()
    print(report.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and ingest all official manuals")
    parser.add_argument("--config", default="configs/all-models.yaml")
    parser.add_argument("--include", default=None, help="Regex for catalog name/manual key")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
