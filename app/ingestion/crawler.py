from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from app.models import CrawlReport, TopicRef
from app.settings import DataSettings

logger = logging.getLogger(__name__)


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").replace("\u2028", " ").split()).strip()


def _direct_child(node: Tag, names: tuple[str, ...]) -> Tag | None:
    for child in node.find_all(recursive=False):
        if isinstance(child, Tag) and child.name in names:
            return child
    return None


def _topic_id_from_url(url: str) -> str:
    return Path(urlparse(url).path).stem


def parse_manual_index(index_html: str, index_url: str) -> tuple[str, str, list[TopicRef], int]:
    soup = BeautifulSoup(index_html, "lxml")
    menu = soup.select_one("#menu")
    vehicle_model = _normalize_text(menu.get("data-title") if isinstance(menu, Tag) else "")
    if not vehicle_model and soup.title:
        vehicle_model = _normalize_text(soup.title.get_text(" ", strip=True))
    manual_id = _normalize_text(menu.get("data-number") if isinstance(menu, Tag) else "")
    if not manual_id:
        manual_id = Path(urlparse(index_url).path).parent.name

    root = soup.select_one("ul#manual-nav") or soup.select_one("ul.map.bookmap")
    if not isinstance(root, Tag):
        raise ValueError("manual directory not found: expected ul#manual-nav")

    directory_refs = 0
    discovered: list[TopicRef] = []

    def walk(list_node: Tag, breadcrumbs: list[str]) -> None:
        nonlocal directory_refs
        for item in list_node.find_all("li", recursive=False):
            if not isinstance(item, Tag):
                continue
            direct = _direct_child(item, ("a", "div"))
            if direct is None:
                continue
            text_node = direct.select_one(".nav-text")
            label = _normalize_text(
                text_node.get_text(" ", strip=True)
                if isinstance(text_node, Tag)
                else direct.get_text(" ", strip=True)
            )
            next_breadcrumbs = [*breadcrumbs, label] if label else list(breadcrumbs)
            href = ""
            if direct.name == "a":
                href = _normalize_text(direct.get("data-content"))
                if not href:
                    candidate = _normalize_text(direct.get("href"))
                    href = candidate if candidate != "#" else ""
            if href:
                directory_refs += 1
                source_url = urljoin(index_url, href)
                discovered.append(
                    TopicRef(
                        topic_id=_topic_id_from_url(source_url),
                        title=label or _topic_id_from_url(source_url),
                        breadcrumbs=next_breadcrumbs,
                        source_url=source_url,
                        source_file=Path(urlparse(source_url).path).name,
                    )
                )
            child_list = item.find("ul", class_="sub-menu", recursive=False)
            if not isinstance(child_list, Tag):
                child_list = item.find("ul", recursive=False)
            if isinstance(child_list, Tag):
                walk(child_list, next_breadcrumbs)

    top_items = root.find_all("li", recursive=False)
    if len(top_items) == 1:
        top = top_items[0]
        direct = _direct_child(top, ("a", "div")) if isinstance(top, Tag) else None
        top_text = _normalize_text(direct.get_text(" ", strip=True) if direct else "")
        child = top.find("ul", recursive=False) if isinstance(top, Tag) else None
        if top_text == "用户手册" and isinstance(child, Tag):
            walk(child, [])
        else:
            walk(root, [])
    else:
        walk(root, [])

    unique: dict[str, TopicRef] = {}
    for topic in discovered:
        unique.setdefault(topic.source_url, topic)
    return manual_id, vehicle_model, list(unique.values()), directory_refs


@dataclass
class _RateLimiter:
    interval_seconds: float

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_allowed = time.monotonic() + self.interval_seconds


class LiXiangManualCrawler:
    def __init__(self, settings: DataSettings) -> None:
        self.settings = settings
        self.index_url = settings.source_url.split("?", 1)[0]
        self.output_dir = settings.raw_dir
        if settings.manual_key:
            self.output_dir /= settings.manual_key
        self.output_dir /= settings.snapshot_id
        self.topic_dir = self.output_dir / "topics"
        interval = 1.0 / max(settings.requests_per_second, 0.1)
        self.rate_limiter = _RateLimiter(interval)

    async def _get_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.settings.request_retries):
            try:
                await self.rate_limiter.wait()
                response = await client.get(url, headers=headers)
                if response.status_code != 304:
                    response.raise_for_status()
                return response
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                if attempt + 1 < self.settings.request_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_bytes(content)
        os.replace(temp_path, path)

    @staticmethod
    def _successful_manifest_entries(path: Path) -> dict[str, dict[str, object]]:
        successful: dict[str, dict[str, object]] = {}
        if not path.exists():
            return successful
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "ok" and item.get("source_url"):
                successful[str(item["source_url"])] = item
        return successful

    async def crawl(self) -> CrawlReport:
        self.topic_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.jsonl"
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        headers = {"User-Agent": "lixiang-qdrant-rag-learning-demo/0.1"}

        async with httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
        ) as client:
            index_response = await self._get_with_retry(client, self.index_url)
            index_bytes = index_response.content
            self._atomic_write(self.output_dir / "index.html", index_bytes)
            manual_id, vehicle_model, topics, directory_refs = parse_manual_index(
                index_response.text, self.index_url
            )

            previous_success = self._successful_manifest_entries(manifest_path)
            semaphore = asyncio.Semaphore(self.settings.request_concurrency)
            records: list[dict[str, str | int]] = []
            failures: list[dict[str, str]] = []
            downloaded = 0
            skipped = 0

            async def fetch_topic(topic: TopicRef) -> None:
                nonlocal downloaded, skipped
                target = self.topic_dir / topic.source_file
                previous = previous_success.get(topic.source_url)
                # Files are written atomically. Immutable snapshot mode can
                # therefore resume from a complete local file even if the
                # process stopped before flushing that page to manifest.jsonl.
                if target.exists() and not self.settings.revalidate_remote:
                    skipped += 1
                    return
                try:
                    conditional_headers: dict[str, str] = {}
                    if previous and target.exists():
                        if previous.get("etag"):
                            conditional_headers["If-None-Match"] = str(previous["etag"])
                        if previous.get("last_modified"):
                            conditional_headers["If-Modified-Since"] = str(
                                previous["last_modified"]
                            )
                    async with semaphore:
                        response = await self._get_with_retry(
                            client,
                            topic.source_url,
                            headers=conditional_headers or None,
                        )
                    if response.status_code == 304:
                        skipped += 1
                        records.append(
                            {
                                "status": "ok",
                                "topic_id": topic.topic_id,
                                "source_url": topic.source_url,
                                "source_file": topic.source_file,
                                "sha256": str(previous.get("sha256", "")),
                                "bytes": int(previous.get("bytes", target.stat().st_size)),
                                "etag": str(previous.get("etag", "")),
                                "last_modified": str(previous.get("last_modified", "")),
                                "revalidation": "not_modified",
                            }
                        )
                        return
                    content = response.content
                    self._atomic_write(target, content)
                    records.append(
                        {
                            "status": "ok",
                            "topic_id": topic.topic_id,
                            "source_url": topic.source_url,
                            "source_file": topic.source_file,
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "bytes": len(content),
                            "etag": response.headers.get("etag", ""),
                            "last_modified": response.headers.get("last-modified", ""),
                        }
                    )
                    downloaded += 1
                except Exception as exc:  # noqa: BLE001 - failure must be persisted per topic
                    message = f"{type(exc).__name__}: {exc}"
                    failures.append({"source_url": topic.source_url, "error": message})
                    records.append(
                        {
                            "status": "error",
                            "topic_id": topic.topic_id,
                            "source_url": topic.source_url,
                            "source_file": topic.source_file,
                            "error": message,
                        }
                    )

            await asyncio.gather(*(fetch_topic(topic) for topic in topics))
            if records:
                with manifest_path.open("a", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            metadata = {
                "manual_id": manual_id,
                "vehicle_model": vehicle_model,
                "snapshot_id": self.settings.snapshot_id,
                "source_url": self.index_url,
                "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
                "directory_refs": directory_refs,
                "unique_topics": len(topics),
                "topics": [topic.model_dump() for topic in topics],
            }
            self._atomic_write(
                self.output_dir / "metadata.json",
                json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
            )

        return CrawlReport(
            manual_id=manual_id,
            vehicle_model=vehicle_model,
            snapshot_id=self.settings.snapshot_id,
            source_url=self.index_url,
            directory_refs=directory_refs,
            unique_topics=len(topics),
            downloaded=downloaded,
            skipped=skipped,
            failed=len(failures),
            topic_refs=topics,
            failures=failures,
        )
