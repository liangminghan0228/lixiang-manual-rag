from __future__ import annotations

import json
from pathlib import PurePosixPath
from urllib.parse import urlparse

import httpx

from app.models import ManualCatalogEntry

ALLOWED_HOST = "manuals.lixiang.com"


def parse_catalog(payload: bytes) -> list[ManualCatalogEntry]:
    raw = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("manual catalog must be a JSON array")
    entries: list[ManualCatalogEntry] = []
    seen_urls: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("manual catalog entries must be objects")
        url = str(item.get("url", "")).split("?", 1)[0]
        parsed = urlparse(url)
        parts = PurePosixPath(parsed.path).parts
        if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
            raise ValueError(f"unsupported manual URL: {url}")
        if len(parts) < 5 or parts[-1] != "index.html":
            raise ValueError(f"unexpected manual URL layout: {url}")
        manual_key = parts[-3]
        snapshot_id = parts[-2]
        if not snapshot_id.isdigit():
            raise ValueError(f"invalid snapshot in manual URL: {url}")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        entries.append(
            ManualCatalogEntry(
                catalog_id=int(item["id"]),
                name=str(item["name"]),
                url=url,
                publish_date=str(item.get("publishDate", "")),
                version=str(item.get("version", "")),
                manual_key=manual_key,
                snapshot_id=snapshot_id,
            )
        )
    return entries


async def fetch_catalog(url: str, timeout_seconds: float = 20.0) -> list[ManualCatalogEntry]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
        raise ValueError(f"unsupported catalog URL: {url}")
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=True,
        headers={"User-Agent": "lixiang-qdrant-rag-learning-demo/0.2"},
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
    return parse_catalog(response.content)
